"""Depth-branch adapter for the Moebius teacher.

TDD2 §1 (``tdd/moebius-grt-depth-finetune-2026-09-12.md``, workspace
root — not tracked by git) specifies the depth branch as an inverted
pointwise / depthwise / pointwise block at H/8:

    input ``channels`` (2) → Conv2d(2, 64, 1) → ReLU →
    Conv2d(64, 64, 3, padding=1, groups=64) → BatchNorm2d(64) → ReLU →
    Conv2d(64, target_dim, 1)   # zero-initialized
    → residual add after the original ``conv_in`` output.

The final 1×1 projection is **zero-initialized** (weight and bias) so
that the depth-conditioned teacher is bit-equal to the original 9-channel
teacher at step 0. This is verified in :mod:`tests.teachers` with
``atol=1e-7`` in FP32.

BN conventions (batch=1 fine-tuning, TDD2 §5): the adapter module runs
in ``train()`` mode during fine-tuning so its BN running statistics
accumulate over steps. The Moebius backbone — including the BN inside
``conv_in`` — stays in ``eval()`` so the pretrained calibration is
never disturbed (see ``training.teacher.finetune._bn_eval``).
"""

from __future__ import annotations

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
    """Zero-initialized pw → dw+BN → pw adapter for depth features.

    Parameters
    ----------
    channels
        Number of input channels (depth features). TDD2 uses ``2``
        (depth-mean + coverage, see §4.3 of the old TDD).
    hidden
        Hidden width. The pointwise expansion, the depthwise groups and
        the BN width all use this value. TDD2 §1 uses ``64``.
    target_dim
        Output channel count. TDD2 uses ``320`` to match
        ``block_out_channels[0]`` / the ``conv_in`` output of the
        Moebius config.
    """

    def __init__(
        self,
        channels: int = 2,
        hidden: int = 64,
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

        # pw 1×1 expansion.
        self.pw_in = nn.Conv2d(self.channels, self.hidden, kernel_size=1, bias=True)
        self.act_in = nn.ReLU(inplace=False)
        # dw 3×3 (groups = hidden) + BN + ReLU.
        self.dw_conv = nn.Conv2d(
            self.hidden,
            self.hidden,
            kernel_size=3,
            padding=1,
            groups=self.hidden,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(self.hidden)
        self.act_dw = nn.ReLU(inplace=False)
        # pw 1×1 projection — zero-initialized (TDD2 §1).
        self.channel_proj = nn.Conv2d(self.hidden, self.target_dim, kernel_size=1, bias=True)

        self._zero_init_final_layer()
        # Cache the enabled flag so the wrapper can disable the branch
        # via context manager (ablation "no depth micro-tuning").
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
        nn.init.kaiming_uniform_(self.pw_in.weight, a=5 ** 0.5)
        if self.pw_in.bias is not None:
            nn.init.uniform_(self.pw_in.bias, -1.0 / self.pw_in.in_channels, 1.0 / self.pw_in.in_channels)
        nn.init.kaiming_normal_(self.dw_conv.weight, mode="fan_out", nonlinearity="relu")
        if self.bn.weight is not None:
            nn.init.ones_(self.bn.weight)
        if self.bn.bias is not None:
            nn.init.zeros_(self.bn.bias)
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
        x = self.pw_in(low_res_depth_features)
        x = self.act_in(x)
        x = self.dw_conv(x)
        x = self.bn(x)
        x = self.act_dw(x)
        return self.channel_proj(x)

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
