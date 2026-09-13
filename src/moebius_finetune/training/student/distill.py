"""One-step student distillation (TDD §7).

The trainer consumes a teacher cache list. Each entry is a dict with:

* ``case_id``: stable string
* ``seed``: integer
* ``initial_noise``: numpy ``[4, H/8, W/8]`` (the actual noise tensor)
* ``final_latent``: numpy ``[4, H/8, W/8]`` (teacher's terminal latent)
* ``teacher_rgb``: numpy ``[3, H, W]`` (the teacher's decoded RGB)
* ``scheduler_cfg``: any auxiliary dict (used as a key for cache
  invalidation; mismatched keys raise)
* ``rgb_hole``: numpy ``[3, H, W]`` (the known RGB in the hole region)
* ``hole_mask``: numpy ``[1, H, W]``
* ``depth_hole``: numpy ``[1, H, W]``
* ``target_rgb``: numpy ``[3, H, W]`` (the ground truth, optional)

While Agent B's real cache is missing, the agent uses
:class:`SyntheticTeacherCache` to generate a small synthetic cache
of the same schema. The smoke test marks this distinction in its
docstring.
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass, field
from typing import Iterable, List, Optional

import numpy as np
import torch
from torch import nn

from moebius_finetune.training.student.losses import distillation_loss


__all__ = ["DistillTrainer", "SyntheticTeacherCache", "distill_student"]


# ---------------------------------------------------------------------------
# Cache schema and key handling
# ---------------------------------------------------------------------------


CACHE_REQUIRED_KEYS = (
    "case_id",
    "seed",
    "initial_noise",
    "final_latent",
    "teacher_rgb",
    "scheduler_cfg",
    "rgb_hole",
    "hole_mask",
    "depth_hole",
)


@dataclass
class TeacherCacheEntry:
    """One cached teacher output, in plain numpy.

    The cache is the contract between Agent B and Agent C. Any change
    in its schema (added / removed keys) is a contract change.
    """

    case_id: str
    seed: int
    initial_noise: np.ndarray
    final_latent: np.ndarray
    teacher_rgb: np.ndarray
    scheduler_cfg: dict
    rgb_hole: np.ndarray
    hole_mask: np.ndarray
    depth_hole: np.ndarray
    target_rgb: Optional[np.ndarray] = None

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "TeacherCacheEntry":
        # Validate the schema.
        missing = [k for k in CACHE_REQUIRED_KEYS if k not in data]
        if missing:
            raise KeyError(
                f"Teacher cache entry missing required keys: {missing}"
            )
        return cls(
            case_id=str(data["case_id"]),
            seed=int(data["seed"]),
            initial_noise=np.asarray(data["initial_noise"], dtype=np.float32),
            final_latent=np.asarray(data["final_latent"], dtype=np.float32),
            teacher_rgb=np.asarray(data["teacher_rgb"], dtype=np.float32),
            scheduler_cfg=dict(data["scheduler_cfg"]),
            rgb_hole=np.asarray(data["rgb_hole"], dtype=np.float32),
            hole_mask=np.asarray(data["hole_mask"], dtype=np.float32),
            depth_hole=np.asarray(data["depth_hole"], dtype=np.float32),
            target_rgb=(
                np.asarray(data["target_rgb"], dtype=np.float32)
                if data.get("target_rgb") is not None
                else None
            ),
        )


def validate_cache_keys(
    cache: List[TeacherCacheEntry], expected_scheduler_cfg: dict
) -> None:
    """Raise if the cache's scheduler config doesn't match the run config.

    The TDD requires that ``cache key`` mismatches be detected and
    refused; this is the implementation. The comparison is on a
    subset of the scheduler config that the student cares about (the
    rest of the config can change without affecting the student).
    """
    for entry in cache:
        for k, v in expected_scheduler_cfg.items():
            if entry.scheduler_cfg.get(k) != v:
                raise ValueError(
                    f"cache key mismatch on case_id={entry.case_id}: "
                    f"field {k} expected {v!r}, got "
                    f"{entry.scheduler_cfg.get(k)!r}"
                )


# ---------------------------------------------------------------------------
# Synthetic cache
# ---------------------------------------------------------------------------


@dataclass
class SyntheticTeacherCache:
    """A small synthetic teacher cache, used only for the 20-step smoke.

    The synthetic cache generates random images, masks, depths and
    latents. It is *not* a faithful teacher output and the agent's
    docs explicitly say the smoke test is not a quality benchmark.
    """

    resolution: int = 64  # small so the smoke is fast
    latent_channels: int = 4
    num_cases: int = 2
    scheduler_cfg: dict = field(
        default_factory=lambda: {"num_steps": 20, "eta": 0.0, "strength": 1.0, "guidance": 1.0}
    )
    seed: int = 0

    def entries(self) -> List[TeacherCacheEntry]:
        rng = np.random.default_rng(self.seed)
        entries: List[TeacherCacheEntry] = []
        h = w = int(self.resolution)
        for i in range(self.num_cases):
            rgb = rng.random((3, h, w), dtype=np.float32)
            depth = rng.uniform(-1.0, 1.0, size=(1, h, w)).astype(np.float32)
            mask = np.zeros((1, h, w), dtype=np.float32)
            # Square hole in the middle.
            mask[0, h // 4 : 3 * h // 4, w // 4 : 3 * w // 4] = 1.0
            rgb_hole = rgb * (1.0 - mask)
            depth_hole = depth * (1.0 - mask)
            # Synthetic noise + latent.
            noise = rng.standard_normal(
                (self.latent_channels, h // 8, w // 8)
            ).astype(np.float32)
            final_latent = rng.standard_normal(
                (self.latent_channels, h // 8, w // 8)
            ).astype(np.float32)
            # Teacher RGB: noisy version of the GT, also hole-aware.
            teacher_rgb = rgb.copy()
            # Inside the hole, add some noise to the GT (mimics a
            # plausible teacher output without doing the real model).
            teacher_rgb = teacher_rgb * (1.0 - mask) + rng.random((3, h, w), dtype=np.float32) * mask
            entries.append(
                TeacherCacheEntry(
                    case_id=f"syn_{i}",
                    seed=int(self.seed + i),
                    initial_noise=noise,
                    final_latent=final_latent,
                    teacher_rgb=teacher_rgb,
                    scheduler_cfg=dict(self.scheduler_cfg),
                    rgb_hole=rgb_hole,
                    hole_mask=mask,
                    depth_hole=depth_hole,
                    target_rgb=rgb,
                )
            )
        return entries


# ---------------------------------------------------------------------------
# Distill trainer
# ---------------------------------------------------------------------------


@dataclass
class DistillTrainer:
    """A minimal in-place student distillation trainer.

    Mirrors :class:`training.codec.train.CodecTrainer`: no checkpointing,
    no rich logging, just a forward loop with the TDD loss and an
    AdamW optimizer.
    """

    student: nn.Module
    optimizer: torch.optim.Optimizer
    steps: int
    device: torch.device
    log_every: int = 5

    history: list = field(default_factory=list)
    final_loss: float = math.inf
    peak_memory_bytes: int = 0

    def _train_step_pixel(
        self, batch: TeacherCacheEntry
    ) -> dict:
        """One optimization step on a single cache entry (pixel student)."""
        from moebius_finetune.contracts import ConditionBatch
        from moebius_finetune.students.pixel import PixelStudentV0

        device = self.device
        condition = ConditionBatch(
            rgb_hole=batch.rgb_hole[None, ...],  # [1, 3, H, W]
            hole_mask=batch.hole_mask[None, ...],
            depth_hole=batch.depth_hole[None, ...],
            noise=batch.initial_noise[None, ...],
        )
        rgb = torch.from_numpy(np.ascontiguousarray(batch.rgb_hole)).to(
            device=device
        )[None]  # [1, 3, H, W]
        rgb_target = (
            torch.from_numpy(np.ascontiguousarray(batch.target_rgb)).to(device=device)[None]
            if batch.target_rgb is not None
            else None
        )
        teacher_rgb = torch.from_numpy(
            np.ascontiguousarray(batch.teacher_rgb)
        ).to(device=device)[None]
        mask = torch.from_numpy(np.ascontiguousarray(batch.hole_mask)).to(
            device=device
        )[None]
        depth = torch.from_numpy(np.ascontiguousarray(batch.depth_hole)).to(
            device=device
        )[None]
        noise = torch.from_numpy(np.ascontiguousarray(batch.initial_noise)).to(
            device=device
        )[None]
        # Concatenate to 5 channels.
        cond5 = torch.cat([rgb, mask, depth], dim=1)
        # The student's inpaint method runs in numpy and is not
        # differentiable, so we use the torch forward directly.
        self.optimizer.zero_grad(set_to_none=True)
        pred = self.student(cond5, noise)
        # Compose: known region from `rgb`, hole region from `pred`.
        composed = (1.0 - mask) * rgb + mask * pred
        info = distillation_loss(
            pred_rgb=composed,
            target_rgb=teacher_rgb,
            gt_rgb=rgb_target,
            target_latent=None,
            pred_latent=None,
            mask=mask,
        )
        info["loss"].backward()
        # Gradient clipping matches the TDD §7.2 setting.
        torch.nn.utils.clip_grad_norm_(self.student.parameters(), max_norm=1.0)
        self.optimizer.step()
        return {k: float(v.detach().item()) for k, v in info.items() if v is not None}

    def _train_step_latent(
        self, batch: TeacherCacheEntry
    ) -> dict:
        """One optimization step on a single cache entry (latent student)."""
        from moebius_finetune.contracts import ConditionBatch
        from moebius_finetune.students.latent import LatentStudentV0

        device = self.device
        rgb = torch.from_numpy(np.ascontiguousarray(batch.rgb_hole)).to(
            device=device
        )[None]
        mask = torch.from_numpy(np.ascontiguousarray(batch.hole_mask)).to(
            device=device
        )[None]
        depth = torch.from_numpy(np.ascontiguousarray(batch.depth_hole)).to(
            device=device
        )[None]
        noise = torch.from_numpy(np.ascontiguousarray(batch.initial_noise)).to(
            device=device
        )[None]
        teacher_rgb = torch.from_numpy(
            np.ascontiguousarray(batch.teacher_rgb)
        ).to(device=device)[None]
        target_latent = torch.from_numpy(
            np.ascontiguousarray(batch.final_latent)
        ).to(device=device)[None]
        condition = ConditionBatch(
            rgb_hole=batch.rgb_hole[None, ...],
            hole_mask=batch.hole_mask[None, ...],
            depth_hole=batch.depth_hole[None, ...],
            noise=batch.initial_noise[None, ...],
        )
        self.optimizer.zero_grad(set_to_none=True)
        # Build the 10-channel input, run the student net, and decode.
        student_input = self.student.build_student_input(condition)
        s_t = torch.from_numpy(np.ascontiguousarray(student_input)).to(device=device)
        pred_latent = self.student.net(s_t)
        # Decode via the codec.
        rgb_filled = self.student._filled_rgb(condition)
        rgb_t = torch.from_numpy(np.ascontiguousarray(rgb_filled)).to(device=device)
        _, skips = self.student.codec.encode(rgb_t)
        pred_rgb = self.student.codec.decode_latent(pred_latent, skips)
        composed = (1.0 - mask) * rgb + mask * pred_rgb
        info = distillation_loss(
            pred_rgb=composed,
            target_rgb=teacher_rgb,
            gt_rgb=(
                torch.from_numpy(np.ascontiguousarray(batch.target_rgb)).to(device=device)[None]
                if batch.target_rgb is not None else None
            ),
            target_latent=target_latent,
            pred_latent=pred_latent,
            mask=mask,
        )
        info["loss"].backward()
        torch.nn.utils.clip_grad_norm_(self.student.parameters(), max_norm=1.0)
        self.optimizer.step()
        return {k: float(v.detach().item()) for k, v in info.items() if v is not None}

    def fit(self, cache: List[TeacherCacheEntry]) -> "DistillTrainer":
        if not cache:
            raise ValueError("DistillTrainer.fit: empty cache")
        if torch.cuda.is_available() and self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
        for step_idx in range(self.steps):
            entry = cache[step_idx % len(cache)]
            # Detect which submodel we are training.
            from moebius_finetune.students.pixel import PixelStudentV0
            from moebius_finetune.students.latent import LatentStudentV0

            if isinstance(self.student, PixelStudentV0):
                info = self._train_step_pixel(entry)
            elif isinstance(self.student, LatentStudentV0):
                info = self._train_step_latent(entry)
            else:
                raise TypeError(
                    f"unsupported student type {type(self.student).__name__}"
                )
            self.history.append({"step": step_idx, **info})
            if (step_idx + 1) % self.log_every == 0 or step_idx == 0:
                if (
                    torch.cuda.is_available()
                    and self.device.type == "cuda"
                ):
                    self.peak_memory_bytes = max(
                        self.peak_memory_bytes,
                        int(torch.cuda.max_memory_allocated(self.device)),
                    )
        self.final_loss = self.history[-1]["loss"] if self.history else math.inf
        return self

    def report(self) -> dict:
        return {
            "steps": self.steps,
            "final_loss": self.final_loss,
            "peak_memory_bytes": self.peak_memory_bytes,
            "history": self.history,
        }


# ---------------------------------------------------------------------------
# Top-level entry point
# ---------------------------------------------------------------------------


def distill_student(
    student: nn.Module,
    teacher_cache: List[TeacherCacheEntry],
    cfg: Optional[dict] = None,
) -> DistillTrainer:
    """Run the student distillation loop and return the trainer.

    Parameters
    ----------
    student
        Either a :class:`PixelStudentV0` or a :class:`LatentStudentV0`.
    teacher_cache
        List of :class:`TeacherCacheEntry` (B's contract). Pass
        ``SyntheticTeacherCache().entries()`` if B is not ready.
    cfg
        Optional dict with training configuration:

        * ``lr`` (default 1e-4)
        * ``weight_decay`` (default 0.01)
        * ``max_grad_norm`` (default 1.0)
        * ``device`` (default: cuda if available else cpu)
        * ``steps`` (default 20)
        * ``expected_scheduler_cfg`` — dict of cache-key fields that
          must match each entry's ``scheduler_cfg``; if the cache
          doesn't match, a ``ValueError`` is raised.
    """
    cfg = cfg or {}
    lr = float(cfg.get("lr", 1e-4))
    weight_decay = float(cfg.get("weight_decay", 0.01))
    max_grad_norm = float(cfg.get("max_grad_norm", 1.0))
    device_str = cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_str)
    steps = int(cfg.get("steps", 20))
    expected = cfg.get("expected_scheduler_cfg")
    if expected is not None:
        validate_cache_keys(teacher_cache, expected)

    student = student.to(device)
    optimizer = torch.optim.AdamW(
        student.parameters(), lr=lr, weight_decay=weight_decay
    )

    trainer = DistillTrainer(
        student=student,
        optimizer=optimizer,
        steps=steps,
        device=device,
    )
    trainer.fit(teacher_cache)
    return trainer
