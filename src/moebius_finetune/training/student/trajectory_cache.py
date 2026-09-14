"""RAM-resident teacher-trajectory cache for MobileMoebius (TDD3 §8).

Motivation (measured in the 16-case smoke, TDD3 §8): teacher-forced
epsilon regression on ``q_sample`` states converges per-step but the
10-step DDIM trajectory drifts — the student never saw the states its
own integration visits. The fix used here: run the **teacher's** 20-step
DDIM rollout once per trajectory and train the student on the states the
teacher actually visited (the even positions are exactly the student's
10-step grid).

Storage contract (user decision 2026-09-14):

* **RAM only, never written to disk.** Everything lives in pinned CPU
  tensors and dies with the process.
* Only ``eps_T`` per step plus the initial noise are stored; the
  ``x_t`` state sequence is recomputed on the GPU from the stored
  epsilons via the deterministic eta=0 DDIM chain, which halves the
  footprint (~0.36 MB/trajectory bf16).
* Conditions (x0/ml/latent_mask/df) stay in the fp32 :class:`CaseBank`
  — built online (ZipDepth), never cached on disk either.
* At bf16 the full-manifest budget (40k cases × K=4 trajectories) is
  ~65 GB pinned RAM — within the 121 GB box without touching the 80 GB
  device-memory directive.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from ...students.schedule import StudentSchedule
from .mobile_distill import CaseBank, MobileDistillConfig, TrainStats

__all__ = [
    "TrajectoryRamCache",
    "TrajTrainConfig",
    "rollout_states",
    "recompute_states",
    "train_mobile_student_traj",
]


# ---------------------------------------------------------------------------
# Rollout: capture the teacher's epsilons on its own trajectory
# ---------------------------------------------------------------------------


@torch.no_grad()
def rollout_states(
    *,
    teacher_fn: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
    batch: Dict[str, torch.Tensor],
    noise: torch.Tensor,
    schedule: StudentSchedule,
) -> torch.Tensor:
    """Run the teacher's 20-step DDIM rollout for a batch of cases.

    ``batch`` holds GPU tensors ``[B, ...]`` (``x0`` unused here, needs
    ``ml``/``latent_mask``/``df``). ``noise`` is ``[B, 4, h, w]``. The
    teacher is queried at every grid step; the epsilons produced at the
    even positions (the student's 10 timesteps) are stacked.

    Returns ``[B, num_steps, 4, h, w]`` float32 epsilons — the distillation
    targets along the trajectory the teacher actually walked.
    """
    t_full = schedule.full_timesteps().to(batch["ml"].device)
    ab_full = schedule.alphas_bar_full.to(batch["ml"].device)
    final_ab = schedule.final_alpha_bar.to(batch["ml"].device)

    ml, latent_mask, df = batch["ml"], batch["latent_mask"], batch["df"]
    noisy = noise.to(device=ml.device, dtype=torch.float32)
    captured: List[Optional[torch.Tensor]] = [None] * schedule.num_steps

    for pos in range(t_full.shape[0]):
        t = int(t_full[pos].item())
        ab_t = ab_full[t].view(1, 1, 1, 1)
        if pos + 1 < t_full.shape[0]:
            ab_prev = ab_full[int(t_full[pos + 1].item())].view(1, 1, 1, 1)
        else:
            ab_prev = final_ab.view(1, 1, 1, 1)
        x9 = torch.cat([noisy, latent_mask, ml], dim=1)
        t_tensor = torch.tensor([t], dtype=torch.int64, device=ml.device).expand(
            noisy.shape[0]
        )
        with torch.autocast(device_type=ml.device.type, dtype=torch.bfloat16):
            eps = teacher_fn(x9, t_tensor, df)
        eps = eps.detach().float()
        if pos % 2 == 0:
            captured[pos // 2] = eps
        pred_x0 = (noisy - (1.0 - ab_t).sqrt() * eps) / ab_t.sqrt()
        noisy = ab_prev.sqrt() * pred_x0 + (1.0 - ab_prev).sqrt() * eps

    missing = [i for i, e in enumerate(captured) if e is None]
    if missing:
        raise RuntimeError(f"rollout missed positions {missing}")
    return torch.stack(captured, dim=1)


def recompute_states(
    *,
    eps_traj: torch.Tensor,
    noise: torch.Tensor,
    schedule: StudentSchedule,
) -> torch.Tensor:
    """Rebuild the visited ``x_t`` sequence from cached epsilons.

    ``eps_traj`` ``[B, S, 4, h, w]``, ``noise`` ``[B, 4, h, w]`` →
    ``[B, S, 4, h, w]`` where entry ``i`` is the state at
    ``schedule.timesteps[i]``. Deterministic: the same eta=0 DDIM chain
    the rollout used.
    """
    ab = schedule.alphas_bar.to(eps_traj.device, dtype=eps_traj.dtype)
    ab = ab.view(1, schedule.num_steps, 1, 1, 1)
    ab_prev = list(schedule.alphas_bar[1:].to(eps_traj.device, dtype=eps_traj.dtype))
    ab_prev.append(
        schedule.final_alpha_bar.to(eps_traj.device, dtype=eps_traj.dtype)
    )
    ab_prev = torch.stack(ab_prev).view(1, schedule.num_steps, 1, 1, 1)

    noisy = noise.to(dtype=eps_traj.dtype).unsqueeze(1)  # [B,1,4,h,w]
    states = []
    for i in range(schedule.num_steps):
        states.append(noisy[:, 0])
        eps_i = eps_traj[:, i].unsqueeze(1)
        pred_x0 = (noisy - (1.0 - ab[:, i : i + 1]).sqrt() * eps_i) / ab[
            :, i : i + 1
        ].sqrt()
        noisy = ab_prev[:, i : i + 1].sqrt() * pred_x0 + (
            1.0 - ab_prev[:, i : i + 1]
        ).sqrt() * eps_i
    return torch.stack(states, dim=1)


# ---------------------------------------------------------------------------
# RAM cache
# ---------------------------------------------------------------------------


class TrajectoryRamCache:
    """Pinned-CPU store of per-trajectory teacher epsilons.

    Layout: ``eps`` ``[N, S, 4, h, w]`` bf16, ``noise`` ``[N, 4, h, w]``
    bf16, ``case_idx`` ``[N]`` int64 (index into the CaseBank). Built
    incrementally via :meth:`extend` — chunks can be appended as they
    are rolled out. Nothing here is ever serialized to disk.
    """

    def __init__(
        self,
        schedule: StudentSchedule,
        latent_hw: int = 64,
        capacity: Optional[int] = None,
    ) -> None:
        """``capacity`` preallocates pinned buffers up-front.

        Preallocation avoids the 2× transient of ``torch.cat`` at
        finalize time when the total trajectory count is known (the
        full-manifest run pins ~55 GB — doubling that peak is not an
        option). Without a capacity the cache accumulates chunks and
        concatenates at :meth:`finalize`.
        """
        self.schedule = schedule
        self.latent_hw = latent_hw
        self._capacity = capacity
        self._filled = 0
        self._eps_p: Optional[torch.Tensor] = None
        self._noise_p: Optional[torch.Tensor] = None
        self._case_idx_p: Optional[torch.Tensor] = None
        self._eps_chunks: List[torch.Tensor] = []
        self._noise_chunks: List[torch.Tensor] = []
        self._case_chunks: List[torch.Tensor] = []
        if capacity is not None and capacity > 0:
            S = schedule.num_steps
            shape_eps = (capacity, S, 4, latent_hw, latent_hw)
            self._eps_p = torch.empty(shape_eps, dtype=torch.bfloat16, pin_memory=True)
            self._noise_p = torch.empty(
                (capacity, 4, latent_hw, latent_hw), dtype=torch.bfloat16, pin_memory=True
            )
            self._case_idx_p = torch.empty((capacity,), dtype=torch.int64, pin_memory=True)

    def extend(self, eps: torch.Tensor, noise: torch.Tensor, case_idx: torch.Tensor) -> None:
        """Append a rollout batch (CPU tensors).

        With a preallocated capacity the batch is written into the
        pinned buffer at the fill pointer; otherwise it accumulates
        until :meth:`finalize`.
        """
        if eps.dim() != 5 or eps.shape[1] != self.schedule.num_steps:
            raise ValueError(
                f"eps must be [B, {self.schedule.num_steps}, 4, h, w], got {tuple(eps.shape)}"
            )
        if eps.shape[0] != noise.shape[0] or noise.shape[0] != case_idx.shape[0]:
            raise ValueError("eps/noise/case_idx batch sizes disagree")
        if self._capacity is not None:
            n = eps.shape[0]
            if self._filled + n > self._capacity:
                raise OverflowError(
                    f"cache capacity {self._capacity} exceeded at fill={self._filled}+{n}"
                )
            s = self._filled
            self._eps_p[s : s + n].copy_(eps.detach().to("cpu", dtype=torch.bfloat16))
            self._noise_p[s : s + n].copy_(noise.detach().to("cpu", dtype=torch.bfloat16))
            self._case_idx_p[s : s + n].copy_(case_idx.detach().to("cpu", dtype=torch.int64))
            self._filled += n
            return
        self._eps_chunks.append(eps.detach().to("cpu", dtype=torch.bfloat16))
        self._noise_chunks.append(noise.detach().to("cpu", dtype=torch.bfloat16))
        self._case_chunks.append(case_idx.detach().to("cpu", dtype=torch.int64))

    def finalize(self) -> "TrajectoryRamCache":
        """Concatenate accumulated chunks and pin them (no-op when preallocated)."""
        if self._capacity is not None:
            if self._filled == 0:
                raise ValueError("cache is empty")
            return self
        if not self._eps_chunks:
            raise ValueError("cache is empty")
        eps = torch.cat(self._eps_chunks).contiguous()
        noise = torch.cat(self._noise_chunks).contiguous()
        case_idx = torch.cat(self._case_chunks).contiguous()
        self._eps_p = eps.pin_memory()
        self._noise_p = noise.pin_memory()
        self._case_idx_p = case_idx.pin_memory()
        self._filled = eps.shape[0]
        return self

    @property
    def num_traj(self) -> int:
        return self._filled

    def sample(
        self, n: int, generator: Optional[torch.Generator] = None
    ) -> Dict[str, torch.Tensor]:
        """Uniformly sample ``n`` trajectories (pinned, ready for H2D).

        Fancy indexing unpins, so results are copied into fresh pinned
        buffers to keep the whole path page-locked.
        """
        if self._eps_p is None:
            self.finalize()
        idx = torch.randint(
            0, self.num_traj, (n,), generator=generator, dtype=torch.int64
        )

        def _gather_pinned(src: torch.Tensor) -> torch.Tensor:
            out = torch.empty(
                src[idx].shape, dtype=src.dtype, pin_memory=True
            )
            out.copy_(src[idx])
            return out

        return {
            "eps": _gather_pinned(self._eps_p),
            "noise": _gather_pinned(self._noise_p),
            "case_idx": _gather_pinned(self._case_idx_p),
        }

    @property
    def ram_gb(self) -> float:
        if self._eps_p is None:
            return 0.0
        eps_bytes = self._filled * self.schedule.num_steps * 4 * self.latent_hw**2 * self._eps_p.element_size()
        noise_bytes = self._filled * 4 * self.latent_hw**2 * self._noise_p.element_size()
        return (eps_bytes + noise_bytes) / 1024**3


# ---------------------------------------------------------------------------
# Teacher-free training on cached trajectories
# ---------------------------------------------------------------------------


@dataclass
class TrajTrainConfig(MobileDistillConfig):
    """Distillation hyperparameters for the trajectory-cache loop.

    One optimizer update consumes ``microbatch`` trajectories × all 10
    visited states (the state dimension is free — the teacher is not
    called during training).
    """

    cosine_to: float = 1e-4


def _lr_at(step: int, total: int, cfg: TrajTrainConfig) -> float:
    if step < cfg.warmup:
        return cfg.lr * float(step + 1) / float(max(cfg.warmup, 1))
    progress = (step - cfg.warmup) / max(total - cfg.warmup, 1)
    return cfg.cosine_to + (cfg.lr - cfg.cosine_to) * 0.5 * (1.0 + math.cos(math.pi * progress))


def _gather_cases(
    bank: CaseBank, case_idx: torch.Tensor, device: torch.device
) -> Dict[str, torch.Tensor]:
    sel = [bank.entries[int(i)] for i in case_idx.tolist()]
    return {
        key: torch.cat([e[key] for e in sel], dim=0).to(device, non_blocking=True)
        for key in ("x0", "ml", "latent_mask", "df")
    }


def train_mobile_student_traj(
    *,
    student: nn.Module,
    bank: CaseBank,
    cache: TrajectoryRamCache,
    schedule: StudentSchedule,
    config: TrajTrainConfig,
    device: torch.device,
    output_dir: Optional[str] = None,
    log: Callable[[str], None] = print,
) -> List[TrainStats]:
    """Distill on cached teacher-trajectory states (no teacher at train time)."""
    if config.microbatch > cache.num_traj:
        raise ValueError(
            f"microbatch ({config.microbatch}) exceeds cache size ({cache.num_traj})"
        )
    torch.manual_seed(config.seed)
    student = student.to(device).train()
    opt = torch.optim.AdamW(student.parameters(), lr=config.lr)
    S = schedule.num_steps
    history: List[TrainStats] = []
    t_start = time.perf_counter()
    accum = 0

    for step in range(config.steps):
        lr_now = _lr_at(step, config.steps, config)
        for group in opt.param_groups:
            group["lr"] = lr_now

        sample = cache.sample(config.microbatch)
        cond = _gather_cases(bank, sample["case_idx"].cpu(), device)
        eps_traj = sample["eps"].to(device, non_blocking=True).float()
        noise = sample["noise"].to(device, non_blocking=True).float()
        states = recompute_states(eps_traj=eps_traj, noise=noise, schedule=schedule)

        B = states.shape[0]
        x_flat = states.reshape(B * S, *states.shape[2:])
        x11 = torch.cat(
            [
                x_flat,
                cond["latent_mask"].repeat_interleave(S, dim=0),
                cond["ml"].repeat_interleave(S, dim=0),
                cond["df"].repeat_interleave(S, dim=0),
            ],
            dim=1,
        )
        step_idx = torch.arange(S, device=device).repeat(B)
        pred = student(x11, step_idx)
        target = eps_traj.reshape(B * S, *eps_traj.shape[2:])
        loss = F.mse_loss(pred, target) / config.grad_accum
        loss.backward()
        accum += 1

        if accum >= config.grad_accum or step == config.steps - 1:
            params = [p for g in opt.param_groups for p in g["params"]]
            grad_norm = float(nn.utils.clip_grad_norm_(params, config.max_grad_norm))
            opt.step()
            opt.zero_grad(set_to_none=True)
            accum = 0

        if (step + 1) % config.log_every == 0 or step == 0:
            stats = TrainStats(
                step=step + 1,
                loss=float(loss.detach()) * config.grad_accum,
                lr=lr_now,
                elapsed_s=time.perf_counter() - t_start,
            )
            history.append(stats)
            config.history.append(stats)
            log(f"step={stats.step} loss={stats.loss:.6f} lr={stats.lr:.2e} gn={grad_norm:.4f}")
        if output_dir and (step + 1) % config.save_every == 0:
            _save_traj(student, schedule, config, step + 1, output_dir)

    if output_dir:
        _save_traj(student, schedule, config, config.steps, output_dir)
    return history


def _save_traj(
    student: nn.Module,
    schedule: StudentSchedule,
    config: TrajTrainConfig,
    step: int,
    output_dir: str,
) -> Path:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"student_step{step:07d}.pt"
    torch.save(
        {
            "model_state": {k: v.detach().cpu() for k, v in student.state_dict().items()},
            "step": step,
            "config": {
                "steps": config.steps,
                "microbatch": config.microbatch,
                "lr": config.lr,
                "cosine_to": config.cosine_to,
                "seed": config.seed,
                "objective": "trajectory-eps-MSE",
            },
            "schedule": {
                "timesteps": schedule.timesteps.tolist(),
                "scheduler_config": dict(schedule.scheduler_config),
            },
        },
        path,
    )
    return path
