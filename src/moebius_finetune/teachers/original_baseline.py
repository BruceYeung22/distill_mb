"""Original 9-channel Moebius baseline, no depth adapter.

Used for the §9.2 alignment test: a depth-conditioned teacher with the
depth-adapter residual disabled (or with a zero-initialized adapter)
must produce a tensor bit-equal (FP32 ``max diff ≤ 1e-6``) to the
original 9-channel teacher given the same condition and noise.

The class mirrors :class:`DepthConditionedRemoval` so the comparison
is apples-to-apples; the only difference is that there is no adapter
to add.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
from torch import nn

from ..contracts import ConditionBatch, validate_condition
from .wrapper import DEFAULT_INPUT_IDS_HALF


class OriginalRemovalBaseline(nn.Module):
    """Single-condition forward for the unmodified 9-channel Moebius teacher."""

    def __init__(
        self,
        model: nn.Module,
        *,
        input_ids_half: int = DEFAULT_INPUT_IDS_HALF,
    ) -> None:
        super().__init__()
        if input_ids_half <= 0 or input_ids_half > model.num_embeddings:
            raise ValueError(
                f"input_ids_half must be in (0, num_embeddings="
                f"{model.num_embeddings}], got {input_ids_half}"
            )
        self.model = model
        self.input_ids_half = int(input_ids_half)

    def _make_input_ids(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.arange(
            0, self.input_ids_half, dtype=torch.int64, device=device
        ).unsqueeze(0).expand(batch_size, -1).contiguous()

    def forward(
        self,
        noisy_latents: torch.Tensor,
        timesteps: torch.Tensor,
        input_ids: Optional[torch.Tensor] = None,
        *,
        depth_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Run the unmodified 9-channel teacher.

        ``depth_features`` is accepted (and ignored) so the baseline is
        a drop-in for :class:`DepthConditionedRemoval` in callers that
        pass the kwarg unconditionally (e.g. the shared DDIM loop).
        """
        B = noisy_latents.shape[0]
        if input_ids is None:
            input_ids = self._make_input_ids(B, noisy_latents.device)
        noise_pred = self.model(noisy_latents, timesteps, input_ids)
        if hasattr(noise_pred, "sample"):
            return noise_pred.sample
        return noise_pred

    def predict_candidate(
        self,
        condition: ConditionBatch,
        noise: torch.Tensor,
        *,
        timesteps: Optional[torch.Tensor] = None,
        input_ids: Optional[torch.Tensor] = None,
        masked_latent: Optional[torch.Tensor] = None,
        depth_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Same signature as :func:`DepthConditionedRemoval.predict_candidate`."""
        validate_condition(condition)
        if masked_latent is None:
            raise ValueError(
                "OriginalRemovalBaseline.predict_candidate requires "
                "masked_latent; the baseline intentionally has no VAE."
            )

        rgb_hole = torch.from_numpy(np.ascontiguousarray(condition.rgb_hole))
        hole_mask = torch.from_numpy(np.ascontiguousarray(condition.hole_mask))
        if not isinstance(noise, torch.Tensor):
            noise = torch.from_numpy(np.ascontiguousarray(noise))

        device = noise.device
        rgb_hole = rgb_hole.to(device)
        hole_mask = hole_mask.to(device)
        noise = noise.to(device)
        masked_latent = masked_latent.to(device)

        B, _, H, W = hole_mask.shape
        latent_h, latent_w = H // 8, W // 8
        latent_mask = nn.functional.interpolate(
            hole_mask, size=(latent_h, latent_w), mode="nearest"
        )
        latent_input = torch.cat([noise, latent_mask, masked_latent], dim=1)

        if timesteps is None:
            timesteps = torch.zeros((B,), dtype=torch.int64, device=device)
        timesteps = timesteps.to(device)

        return self.forward(
            latent_input, timesteps, input_ids, depth_features=depth_features
        )


__all__ = ["OriginalRemovalBaseline"]
