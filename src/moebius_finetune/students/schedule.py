"""Fixed 10-step schedule for the MobileMoebius student (TDD3 §2).

The student runs a DDIM trajectory that is the **even-index subsample**
of the teacher's 20-step DDIM schedule (same ``default_scheduler_config``
betas). Because ``DDIMScheduler.set_timesteps(10)`` produces a
*different* grid (``[901, ..., 1]`` with ``steps_offset=1``) than the
subsampled 20-step grid (``[951, 851, ..., 51]``), the table is built
explicitly here and the DDIM update is implemented by hand.

The hand-written update is numerically pinned against
``diffusers.DDIMScheduler.step`` in the tests (mid-trajectory pair and
the final step with ``set_alpha_to_one=False``), so the student's
inference loop uses exactly the same integrator semantics as the
teacher pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import torch

__all__ = ["StudentSchedule"]


@dataclass(frozen=True)
class StudentSchedule:
    """The fixed ``(t_i, ᾱ_i)`` table shared by training and inference.

    Attributes
    ----------
    timesteps
        Descending int64 tensor of the 10 teacher timesteps
        (``[951, 851, ..., 51]`` for the default 20-step config).
    alphas_bar
        ``ᾱ(t_i)`` for the 10 entries, float32, shape ``[10]``.
    final_alpha_bar
        The ᾱ the *last* DDIM step integrates towards. Mirrors the
        teacher scheduler's ``set_alpha_to_one=False`` convention
        (i.e. ``alphas_cumprod[0]``, not 1.0).
    alphas_bar_full
        The full ``[num_train_timesteps]`` cumprod table (kept for
        ``q_sample`` lookups and debugging).
    """

    timesteps: torch.Tensor
    alphas_bar: torch.Tensor
    final_alpha_bar: torch.Tensor
    alphas_bar_full: torch.Tensor
    scheduler_config: Mapping[str, Any] = field(repr=False, default_factory=dict)

    @property
    def num_steps(self) -> int:
        return int(self.timesteps.shape[0])

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_scheduler_config(cls, cfg: Mapping[str, Any]) -> "StudentSchedule":
        """Derive the 10-step table from the teacher's scheduler config.

        ``cfg`` is the mapping produced by
        :func:`moebius_finetune.training.teacher.cache.default_scheduler_config`
        (``scheduler_type='DDIM'``, ``num_inference_steps=20``,
        ``steps_offset=1``, ``set_alpha_to_one`` false-equivalent).
        """
        from diffusers import DDIMScheduler

        if str(cfg.get("scheduler_type", "DDIM")) != "DDIM":
            raise ValueError(
                f"only DDIM schedules are supported, got {cfg.get('scheduler_type')!r}"
            )
        n_infer = int(cfg["num_inference_steps"])
        if n_infer < 2 or n_infer % 2 != 0:
            raise ValueError(
                f"num_inference_steps must be an even number >= 2, got {n_infer}"
            )
        ddim = DDIMScheduler(
            num_train_timesteps=int(cfg["num_train_timesteps"]),
            beta_start=float(cfg["beta_start"]),
            beta_end=float(cfg["beta_end"]),
            beta_schedule=str(cfg["beta_schedule"]),
            clip_sample=False,
            set_alpha_to_one=False,
            steps_offset=int(cfg["steps_offset"]),
        )
        ddim.set_timesteps(n_infer)
        t_full = ddim.timesteps.detach().cpu().to(torch.int64)
        t_sub = t_full[::2].contiguous()  # descending, n_infer // 2 entries
        ab_full = ddim.alphas_cumprod.detach().cpu().to(torch.float32)
        ab_sub = ab_full[t_sub].contiguous()
        final_ab = ddim.final_alpha_cumprod.detach().cpu().to(torch.float32)
        if final_ab.ndim > 0:  # diffusers may store a 0-dim tensor
            final_ab = final_ab.reshape(())
        return cls(
            timesteps=t_sub,
            alphas_bar=ab_sub,
            final_alpha_bar=final_ab,
            alphas_bar_full=ab_full,
            scheduler_config=dict(cfg),
        )

    # ------------------------------------------------------------------
    # Training-side helpers
    # ------------------------------------------------------------------

    def q_sample(
        self, x0: torch.Tensor, eps: torch.Tensor, step_idx: torch.Tensor
    ) -> torch.Tensor:
        """``√ᾱ_i·x0 + √(1-ᾱ_i)·ε`` for per-sample schedule indices.

        ``step_idx`` is an int64 tensor of values in ``[0, num_steps)``.
        """
        if int(step_idx.min().item()) < 0 or int(step_idx.max().item()) >= self.num_steps:
            raise ValueError(
                f"step_idx out of range [0, {self.num_steps}): "
                f"min={int(step_idx.min().item())}, max={int(step_idx.max().item())}"
            )
        ab = self.alphas_bar.to(device=x0.device, dtype=x0.dtype)[step_idx]
        ab = ab.view(-1, 1, 1, 1)
        return ab.sqrt() * x0 + (1.0 - ab).sqrt() * eps

    def timestep_tensor(self, step_idx: torch.Tensor, device: torch.device) -> torch.Tensor:
        """Map schedule indices to the teacher's continuous timesteps."""
        return self.timesteps.to(device)[step_idx]

    # ------------------------------------------------------------------
    # Inference-side DDIM update (eta = 0)
    # ------------------------------------------------------------------

    def full_timesteps(self) -> torch.Tensor:
        """The teacher's full inference grid this table was subsampled from.

        For the default config this is the 20-step descending grid
        ``[951, 901, ..., 1]`` (``arange(n)·ratio + offset``, flipped).
        The trajectory rollout walks this grid and captures epsilons at
        the even positions (= :attr:`timesteps`).
        """
        cfg = self.scheduler_config
        n_train = int(cfg.get("num_train_timesteps", 1000))
        n_infer = int(cfg.get("num_inference_steps", 20))
        offset = int(cfg.get("steps_offset", 1))
        grid = torch.arange(0, n_infer, dtype=torch.int64) * (n_train // n_infer) + offset
        return torch.flip(grid, dims=[0]).contiguous()

    def ddim_step(
        self,
        x_t: torch.Tensor,
        eps_pred: torch.Tensor,
        step_idx: int,
    ) -> torch.Tensor:
        """One eta=0 DDIM update from schedule entry ``step_idx``.

        ``x_prev = √ᾱ_prev·x̂0 + √(1-ᾱ_prev)·ε`` with
        ``x̂0 = (x_t − √(1-ᾱ_t)·ε)/√ᾱ_t``. The previous ᾱ is the next
        table entry, or :attr:`final_alpha_bar` after the last entry.
        """
        if not 0 <= step_idx < self.num_steps:
            raise ValueError(f"step_idx {step_idx} outside [0, {self.num_steps})")
        ab_t = self.alphas_bar[step_idx].to(device=x_t.device, dtype=x_t.dtype)
        if step_idx + 1 < self.num_steps:
            ab_prev = self.alphas_bar[step_idx + 1].to(
                device=x_t.device, dtype=x_t.dtype
            )
        else:
            ab_prev = self.final_alpha_bar.to(device=x_t.device, dtype=x_t.dtype)
        ab_t = ab_t.view(-1, 1, 1, 1)
        ab_prev = ab_prev.view(-1, 1, 1, 1)
        pred_x0 = (x_t - (1.0 - ab_t).sqrt() * eps_pred) / ab_t.sqrt()
        return ab_prev.sqrt() * pred_x0 + (1.0 - ab_prev).sqrt() * eps_pred
