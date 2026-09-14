"""Teacher-forced epsilon distillation for MobileMoebius (TDD3 §3).

Minimal training loop: for every microbatch the student and the
teacher each run **one** forward at the same noise level and their
epsilon predictions are aligned with plain MSE (decision S4). There is
no trajectory rollout and no teacher-output caching — every step draws
fresh ``(case, step_idx, noise)`` triples on the fly.

Input-contract note (mirrors ``training.teacher.finetune._train_step``
exactly so the teacher sees an identical distribution):

* 9-channel teacher input  = ``cat([x_t, latent_mask, masked_latent])``
  with ``latent_mask = interpolate(hole_mask, mode="nearest")``;
* the student's 11-channel input appends the §4.3 depth features
  ``[depth_mean, coverage]`` (``_default_depth_features``);
* the teacher is wrapped by a caller-supplied ``teacher_fn`` so tests
  can stub it without loading the 226M model.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
import torch
from torch import nn

from ...contracts import ConditionBatch
from ...data.grt_dataset import CaseSpec, Predictor, build_case
from ...students.schedule import StudentSchedule
from ...teachers.wrapper import _default_depth_features

__all__ = [
    "CaseBank",
    "MobileDistillConfig",
    "TrainStats",
    "build_case_bank",
    "train_mobile_student",
]


# ---------------------------------------------------------------------------
# Case bank (prebuilt conditions, overfit mode)
# ---------------------------------------------------------------------------


class CaseBank:
    """Prebuilt latents for a fixed case list (overfit / smoke mode).

    Each entry stores everything that does not depend on the noise or
    the schedule index: the clean latent ``x0``, the masked latent
    ``ml``, the nearest-pooled ``latent_mask`` and the depth features
    ``df`` — all at latent resolution, float32, on the target device.
    Building once is the point of overfit mode: the 16 cases are
    reused for every step, so ZipDepth and the VAE run exactly once.
    """

    def __init__(self, entries: List[Dict[str, torch.Tensor]]) -> None:
        if not entries:
            raise ValueError("CaseBank needs at least one entry")
        self.entries = entries

    def __len__(self) -> int:
        return len(self.entries)

    def gather(
        self, case_idx: torch.Tensor, device: torch.device
    ) -> Dict[str, torch.Tensor]:
        """Stack bank entries for a microbatch of case indices."""
        sel = [self.entries[int(i)] for i in case_idx.tolist()]
        return {
            key: torch.cat([e[key] for e in sel], dim=0).to(device)
            for key in ("x0", "ml", "latent_mask", "df")
        }

    @property
    def case_ids(self) -> List[str]:
        return [e["case_id"] for e in self.entries]


def _encode(
    vae: nn.Module, image_pm1: torch.Tensor, device: torch.device
) -> torch.Tensor:
    """VAE encode a [-1,1] image to the scaled fp32 latent (teacher convention)."""
    with torch.no_grad():
        vae_dtype = next(vae.parameters()).dtype
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            encoded = vae.encode(image_pm1.to(device=device, dtype=vae_dtype))
            latent = (
                encoded.latent_dist.mode()
                if hasattr(encoded, "latent_dist")
                else encoded.mode()
            )
        scale = getattr(vae.config, "scaling_factor", 0.13025)
    return (latent * float(scale)).to(dtype=torch.float32)


def build_case_bank(
    cases: Sequence[CaseSpec],
    *,
    image_dir: str,
    predictor: Predictor,
    vae: nn.Module,
    device: torch.device,
    size: int = 512,
) -> CaseBank:
    """Build the per-case conditioning bank (TDD3 §3, overfit mode)."""
    entries: List[Dict[str, torch.Tensor]] = []
    for case in cases:
        built = build_case(case, image_dir=image_dir, predictor=predictor, size=size)
        rgb = torch.from_numpy(np.ascontiguousarray(built["target"]))[None]
        hole = torch.from_numpy(np.ascontiguousarray(built["rgb_hole"]))[None]
        mask = torch.from_numpy(np.ascontiguousarray(built["hole_mask"]))[None]
        depth = torch.from_numpy(np.ascontiguousarray(built["depth_hole"]))[None]
        x0 = _encode(vae, 2.0 * rgb - 1.0, device)
        ml = _encode(vae, (2.0 * hole - 1.0) * (1.0 - mask), device)
        latent_mask = nn.functional.interpolate(
            mask.to(device), size=(size // 8, size // 8), mode="nearest"
        ).to(dtype=torch.float32)
        df = _default_depth_features(mask.to(device), depth.to(device))
        entries.append(
            {
                "case_id": case.case_id,
                "x0": x0.cpu(),
                "ml": ml.cpu(),
                "latent_mask": latent_mask.cpu(),
                "df": df.cpu(),
                "pixel_mask": mask.to(dtype=torch.float32),
                "target": rgb.to(dtype=torch.float32),
            }
        )
    return CaseBank(entries)


# ---------------------------------------------------------------------------
# Config / stats
# ---------------------------------------------------------------------------


@dataclass
class MobileDistillConfig:
    """Hyperparameters for the teacher-forced distillation loop."""

    steps: int = 1000
    microbatch: int = 16
    grad_accum: int = 1
    lr: float = 2e-4
    warmup: int = 100
    max_grad_norm: float = 1.0
    seed: int = 0
    save_every: int = 500
    log_every: int = 25
    weight_decay: float = 0.0
    history: List["TrainStats"] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.steps <= 0 or self.microbatch <= 0 or self.grad_accum <= 0:
            raise ValueError("steps, microbatch, grad_accum must be positive")
        if self.lr <= 0:
            raise ValueError(f"lr must be positive, got {self.lr}")


@dataclass
class TrainStats:
    step: int
    loss: float
    lr: float
    elapsed_s: float


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


def _lr_at(step: int, cfg: MobileDistillConfig) -> float:
    if step >= cfg.warmup:
        return cfg.lr
    return cfg.lr * float(step + 1) / float(max(cfg.warmup, 1))


def train_mobile_student(
    *,
    student: nn.Module,
    teacher_fn: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
    bank: CaseBank,
    schedule: StudentSchedule,
    config: MobileDistillConfig,
    device: torch.device,
    output_dir: Optional[str] = None,
    log: Callable[[str], None] = print,
) -> List[TrainStats]:
    """Run the teacher-forced distillation loop.

    Parameters
    ----------
    student
        ``MobileMoebius`` (or any module mapping ``[B, 11, h, w]`` +
        per-sample step indices to epsilon).
    teacher_fn
        ``(x9, timesteps, df) → epsilon`` — typically
        ``lambda x9, t, df: teacher.forward(x9, t, None, depth_features=df)``
        with the teacher in eval mode. Called under ``no_grad``.
    bank / schedule
        Prebuilt conditions and the fixed 10-step table.
    """
    if config.microbatch > len(bank):
        raise ValueError(
            f"microbatch ({config.microbatch}) exceeds bank size ({len(bank)}); "
            "for full-manifest training use an online provider instead of a bank"
        )
    torch.manual_seed(config.seed)
    student = student.to(device).train()
    opt = torch.optim.AdamW(
        student.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )

    history: List[TrainStats] = []
    t_start = time.perf_counter()
    accum_count = 0
    opt.zero_grad(set_to_none=True)

    for step in range(config.steps):
        lr_now = _lr_at(step, config)
        for group in opt.param_groups:
            group["lr"] = lr_now

        case_idx = torch.randint(0, len(bank), (config.microbatch,))
        batch = bank.gather(case_idx, device)
        step_idx = torch.randint(
            0, schedule.num_steps, (config.microbatch,), dtype=torch.int64, device=device
        )
        eps = torch.randn_like(batch["x0"])
        x_t = schedule.q_sample(batch["x0"], eps, step_idx.cpu())

        # Student forward (fp32) on the 11-channel input: 9ch ⊕ df.
        x11 = torch.cat([x_t, batch["latent_mask"], batch["ml"], batch["df"]], dim=1)
        pred = student(x11, step_idx)

        # Teacher forward (same x_t, same t_i) — no grad, bf16 autocast.
        # Full 9ch teacher input, exactly _train_step's assembly order.
        x9 = torch.cat([x_t.detach(), batch["latent_mask"], batch["ml"]], dim=1)
        t_cont = schedule.timestep_tensor(step_idx, device)
        with torch.no_grad():
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                eps_T = teacher_fn(x9, t_cont, batch["df"])
        target = eps_T.detach().to(dtype=torch.float32)

        loss = torch.nn.functional.mse_loss(pred, target) / config.grad_accum
        loss.backward()
        accum_count += 1

        if accum_count >= config.grad_accum or step == config.steps - 1:
            params = [p for g in opt.param_groups for p in g["params"]]
            grad_norm = float(nn.utils.clip_grad_norm_(params, config.max_grad_norm))
            opt.step()
            opt.zero_grad(set_to_none=True)
            accum_count = 0

        if (step + 1) % config.log_every == 0 or step == 0:
            stats = TrainStats(
                step=step + 1,
                loss=float(loss.detach()) * config.grad_accum,
                lr=lr_now,
                elapsed_s=time.perf_counter() - t_start,
            )
            history.append(stats)
            config.history.append(stats)
            log(
                f"step={stats.step} loss={stats.loss:.6f} lr={stats.lr:.2e} "
                f"gn={grad_norm:.4f}"
            )
        if output_dir and (step + 1) % config.save_every == 0:
            _save(student, schedule, config, step + 1, output_dir)

    if output_dir:
        _save(student, schedule, config, config.steps, output_dir)
    return history


def _save(
    student: nn.Module,
    schedule: StudentSchedule,
    config: MobileDistillConfig,
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
                "seed": config.seed,
            },
            "schedule": {
                "timesteps": schedule.timesteps.tolist(),
                "scheduler_config": dict(schedule.scheduler_config),
            },
        },
        path,
    )
    return path


# ---------------------------------------------------------------------------
# Inference: 10-step DDIM sampling with the student (mirrors cache.py)
# ---------------------------------------------------------------------------


@torch.no_grad()
def sample_mobile_student(
    *,
    student: nn.Module,
    bank_entry: Dict[str, torch.Tensor],
    schedule: StudentSchedule,
    initial_noise: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """Run the student's 10-step DDIM loop for one case.

    Mirrors ``training.teacher.cache._ddim_loop`` exactly (same 9ch
    assembly semantics — the student just consumes 11ch) and returns
    the final latent ``[1, 4, h, w]``.
    """
    student = student.to(device).eval()
    ml = bank_entry["ml"].to(device)
    latent_mask = bank_entry["latent_mask"].to(device)
    df = bank_entry["df"].to(device)
    noisy = initial_noise.to(device=device, dtype=torch.float32)
    for i in range(schedule.num_steps):
        x11 = torch.cat([noisy, latent_mask, ml, df], dim=1)
        step_idx = torch.tensor([i], dtype=torch.int64, device=device)
        eps = student(x11, step_idx)
        noisy = schedule.ddim_step(noisy, eps.float(), i)
    return noisy


@torch.no_grad()
def sample_teacher_reference(
    *,
    teacher_fn: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
    bank_entry: Dict[str, torch.Tensor],
    schedule: StudentSchedule,
    initial_noise: torch.Tensor,
    device: torch.device,
    alphas_bar_full: torch.Tensor,
) -> torch.Tensor:
    """Run the teacher's 20-step DDIM loop for one case.

    Uses the teacher's full (20-step) timesteps derived from the same
    scheduler config, mirroring ``_ddim_loop``. Provided here so the
    smoke comparison runs through one shared implementation of the
    DDIM update (``schedule.alphas_bar_full``).
    """
    t_full = torch.tensor(
        [951, 901, 851, 801, 751, 701, 651, 601, 551, 501, 451, 401, 351, 301, 251, 201, 151, 101, 51, 1],
        dtype=torch.int64,
    )
    # Derive from the config instead of hardcoding when possible.
    n_train = int(schedule.scheduler_config.get("num_train_timesteps", 1000))
    n_infer = int(schedule.scheduler_config.get("num_inference_steps", 20))
    offset = int(schedule.scheduler_config.get("steps_offset", 1))
    ratio = n_train // n_infer
    t_full = torch.arange(0, n_infer, dtype=torch.int64) * ratio + offset
    t_full = torch.flip(t_full, dims=[0]).contiguous()

    ml = bank_entry["ml"].to(device)
    latent_mask = bank_entry["latent_mask"].to(device)
    df = bank_entry["df"].to(device)
    ab_full = alphas_bar_full.to(device)

    noisy = initial_noise.to(device=device, dtype=torch.float32)
    for pos in range(t_full.shape[0]):
        t = int(t_full[pos].item())
        ab_t = ab_full[t].view(1, 1, 1, 1)
        if pos + 1 < t_full.shape[0]:
            ab_prev = ab_full[int(t_full[pos + 1].item())].view(1, 1, 1, 1)
        else:
            ab_prev = schedule.final_alpha_bar.to(device).view(1, 1, 1, 1)
        x9 = torch.cat([noisy, latent_mask, ml], dim=1)
        t_tensor = torch.tensor([t], dtype=torch.int64, device=device)
        eps = teacher_fn(x9, t_tensor, df).float()
        pred_x0 = (noisy - (1.0 - ab_t).sqrt() * eps) / ab_t.sqrt()
        noisy = ab_prev.sqrt() * pred_x0 + (1.0 - ab_prev).sqrt() * eps
    return noisy


#: Re-exported for callers that need the condition container type.
ConditionBatchType = ConditionBatch
