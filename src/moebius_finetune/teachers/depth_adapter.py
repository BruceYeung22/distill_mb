"""Depth-branch adapter for the Moebius teacher.

TDD §5.1 specifies the depth branch:

    H/8 input 2 channels → Conv2d(2, 16, 3, padding=1) → ReLU →
    Conv2d(16, target_dim, 1) → residual add after the original
    ``conv_in``.

The final 1×1 projection is **zero-initialized** (weight and bias) so
that the depth-conditioned teacher is bit-equal to the original 9-channel
teacher at step 0. This is verified in :mod:`tests.teachers` with
``atol=1e-7`` in FP32.

The adapter does **not** introduce BN; it only stores its own
``state_dict`` so it can be saved/loaded independently of the main
Moebius backbone.
"""

from __future__ import annotations

from typing import Iterable, Optional

import torch
from torch import nn


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class DepthAdapterConfigError(ValueError):
    """Raised when the depth adapter is constructed with invalid args."""


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------


class DepthConditionAdapter(nn.Module):
    """Zero-initialized 2→hidden→target_dim adapter for depth features.

    Parameters
    ----------
    channels
        Number of input channels (depth features). The TDD spec uses
        ``2`` (coverage + depth-mean, see §4.3).
    hidden
        Hidden width for the 3×3 conv. The spec uses ``16``.
    target_dim
        Output channel count. The spec uses ``320`` to match
        ``block_out_channels[0]`` of the Moebius config (TDD §5.1).
    """

    def __init__(
        self,
        channels: int = 2,
        hidden: int = 16,
        target_dim: int = 320,
    ) -> None:
        super().__init__()
        if channels <= 0 or hidden <= 0 or target_dim <= 0:
            raise DepthAdapterConfigError(
                f"channels, hidden, target_dim must be positive, got "
                f"{(channels, hidden, target_dim)}"
            )

        self.channels = int(channels)
        self.hidden = int(hidden)
        self.target_dim = int(target_dim)

        self.spatial_conv = nn.Conv2d(
            self.channels, self.hidden, kernel_size=3, padding=1
        )
        self.act = nn.ReLU(inplace=False)
        self.channel_proj = nn.Conv2d(self.hidden, self.target_dim, kernel_size=1)

        self._zero_init_final_layer()
        # Cache the enabled flag so the wrapper can disable the branch
        # via context manager (TDD §5.1 ablation "no depth micro-tuning").
        self.enabled: bool = True

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def _zero_init_final_layer(self) -> None:
        """Zero weight and bias on the final 1×1 projection.

        This is the only non-default initialization in the adapter. We
        keep it in a method so tests can re-trigger it after manual
        weight edits (e.g. ``load_state_dict``).
        """
        with torch.no_grad():
            self.channel_proj.weight.zero_()
            self.channel_proj.bias.zero_()

    def reset_parameters(self) -> None:  # pragma: no cover - convenience
        """Re-initialize the adapter from scratch (PyTorch default + zero final)."""
        nn.init.kaiming_uniform_(self.spatial_conv.weight, a=5 ** 0.5)
        if self.spatial_conv.bias is not None:
            fan_in = self.spatial_conv.in_channels * self.spatial_conv.kernel_size[0] * self.spatial_conv.kernel_size[1]
            bound = 1.0 / (fan_in ** 0.5) if fan_in > 0 else 0.0
            nn.init.uniform_(self.spatial_conv.bias, -bound, bound)
        self._zero_init_final_layer()

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        low_res_depth_features: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the residual to add to the original ``conv_in`` output.

        Parameters
        ----------
        low_res_depth_features
            Float tensor of shape ``[B, channels, H/8, W/8]``.

        Returns
        -------
        torch.Tensor
            Residual of shape ``[B, target_dim, H/8, W/8]`` to be added
            element-wise to the main path. When the adapter is disabled
            (``self.enabled is False``) a zero tensor of the right
            shape/dtype/device is returned.
        """
        if not self.enabled:
            # Return zeros with the right shape/dtype/device without
            # touching the parameters (so this branch is also safe under
            # ``torch.no_grad``).
            B = low_res_depth_features.shape[0]
            H = low_res_depth_features.shape[2]
            W = low_res_depth_features.shape[3]
            return torch.zeros(
                (B, self.target_dim, H, W),
                dtype=low_res_depth_features.dtype,
                device=low_res_depth_features.device,
            )
        x = self.spatial_conv(low_res_depth_features)
        x = self.act(x)
        x = self.channel_proj(x)
        return x

    # ------------------------------------------------------------------
    # State dict helpers
    # ------------------------------------------------------------------

    def state_dict(self, *args, **kwargs):  # type: ignore[override]
        return super().state_dict(*args, **kwargs)

    def load_state_dict(
        self,
        state_dict: dict,
        strict: bool = True,
    ):  # type: ignore[override]
        return super().load_state_dict(state_dict, strict=strict)

    # ------------------------------------------------------------------
    # Context manager helpers
    # ------------------------------------------------------------------

    def disable(self) -> "DepthConditionAdapter":
        """Disable the adapter (residual becomes 0)."""
        self.enabled = False
        return self

    def enable(self) -> "DepthConditionAdapter":
        """Re-enable the adapter."""
        self.enabled = True
        return self


__all__ = ["DepthConditionAdapter", "DepthAdapterConfigError"]
