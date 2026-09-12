"""First-version distillation losses (TDD §7.2).

* :func:`hole_l1` — per-case normalised L1, averaged across the
  batch. Empty mask regions contribute 0 (no division by zero).
* :func:`boundary_gradient_l1` — the gradient L1 around the hole
  boundary (default ~3 pixels on each side). Computed on the
  *final composed RGB* (so it includes the known pixels), in both
  horizontal and vertical directions.
* :func:`distillation_loss` — combines the above with the latent
  MSE for latent students.

All reductions are kept in FP32.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn


__all__ = [
    "boundary_gradient_l1",
    "distillation_loss",
    "hole_l1",
]


def _safe_per_case_normalised_l1(
    diff: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """Per-case L1 normalised by the hole pixel count, with safe denom.

    ``diff`` and ``mask`` share the same shape ``[B, 3, H, W]`` and
    ``[B, 1, H, W]`` respectively. Returns a ``[B]`` tensor where
    each entry is ``sum(|diff| * mask) / max(sum(mask) * 3, 1)``.
    Cases with zero mask contribute 0.
    """
    assert diff.shape == mask.shape[:1] + (3,) + mask.shape[2:] or diff.shape[1] == mask.shape[1] * 3, (
        "diff must have 3 channels when mask has 1 channel"
    )
    # Sum over (C, H, W) -> [B].
    numerator = (diff.abs() * mask).flatten(1).sum(dim=1)
    denom = (mask.flatten(1).sum(dim=1) * 3.0).clamp_min(1.0)
    out = numerator / denom
    # Zero out cases where the hole is empty (denom would be 0).
    empty = (mask.flatten(1).sum(dim=1) == 0)
    out = out.masked_fill(empty, 0.0)
    return out


def hole_l1(
    pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """Per-case normalised hole L1, averaged over the batch.

    All tensors are expected in ``[B, C, H, W]`` with ``C`` matching
    between ``pred`` and ``target``. ``mask`` is ``[B, 1, H, W]`` in
    ``{0, 1}``. The reduction is a plain mean across the batch.
    """
    if pred.shape != target.shape:
        raise ValueError(
            f"hole_l1: pred shape {pred.shape} != target shape {target.shape}"
        )
    if pred.shape[0] != mask.shape[0] or pred.shape[2:] != mask.shape[2:]:
        raise ValueError(
            f"hole_l1: pred {pred.shape} and mask {mask.shape} spatial dims disagree"
        )
    if mask.shape[1] != 1:
        raise ValueError(
            f"hole_l1: mask must have 1 channel, got {mask.shape[1]}"
        )
    pred32 = pred.float()
    target32 = target.float()
    mask32 = mask.float()
    diff = (pred32 - target32) * mask32
    per_case = _safe_per_case_normalised_l1(diff, mask32)
    return per_case.mean()


def boundary_gradient_l1(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    band_px: int = 3,
) -> torch.Tensor:
    """L1 of the horizontal+vertical gradient around the hole boundary.

    The boundary band is the union of two dilations of the mask:
    the original ``mask`` and ``mask`` dilated by ``band_px`` pixels
    in each direction. The loss is averaged across all boundary
    pixels and across the batch.

    The gradient is computed in the *final composed* image (i.e.
    ``pred`` already contains the known pixels), in line with
    TDD §7.2. The function uses the standard Sobel-style first
    differences; the absolute values are summed and divided by the
    number of boundary pixels.
    """
    if pred.shape != target.shape:
        raise ValueError(
            f"boundary_gradient_l1: pred shape {pred.shape} != target "
            f"shape {target.shape}"
        )
    if mask.shape[1] != 1:
        raise ValueError(
            f"boundary_gradient_l1: mask must have 1 channel, got {mask.shape[1]}"
        )
    if band_px < 1:
        raise ValueError(f"band_px must be >= 1, got {band_px}")
    pred32 = pred.float()
    target32 = target.float()
    mask32 = mask.float()

    # Build the boundary band by dilating the mask with a 3x3
    # max-pool applied `band_px` times. The boundary = dilated - mask.
    band = mask32
    for _ in range(band_px):
        band = nn.functional.max_pool2d(band, kernel_size=3, stride=1, padding=1)
    band = (band - mask32).clamp_min(0.0)  # only the ring outside the hole

    # First differences.
    pred_dx = pred32[..., :, 1:] - pred32[..., :, :-1]
    pred_dy = pred32[..., 1:, :] - pred32[..., :-1, :]
    target_dx = target32[..., :, 1:] - target32[..., :, :-1]
    target_dy = target32[..., 1:, :] - target32[..., :-1, :]
    diff_dx = (pred_dx - target_dx).abs()
    diff_dy = (pred_dy - target_dy).abs()
    # Match the band masks to the difference shapes.
    band_x = band[..., :, 1:] + band[..., :, :-1]
    band_x = (band_x > 0).float()
    band_y = band[..., 1:, :] + band[..., :-1, :]
    band_y = (band_y > 0).float()
    diff_dx = diff_dx * band_x
    diff_dy = diff_dy * band_y
    # Sum, normalised by the number of contributing pixels.
    num = diff_dx.sum() + diff_dy.sum()
    den = band_x.sum() + band_y.sum()
    if den.item() == 0:
        return torch.zeros((), device=pred.device, dtype=pred32.dtype)
    return num / den


def distillation_loss(
    pred_rgb: torch.Tensor,
    target_rgb: torch.Tensor,
    target_latent: Optional[torch.Tensor],
    pred_latent: Optional[torch.Tensor],
    mask: torch.Tensor,
    *,
    boundary_band_px: int = 3,
    boundary_weight: float = 0.1,
    latent_weight: float = 1.0,
) -> dict:
    """Compute the full distillation loss (TDD §7.2).

    Returns a dict with the per-component losses and the total. All
    tensors are expected on the same device and in FP32 (the caller
    is responsible for casting).
    """
    l_rgb_teacher = hole_l1(pred_rgb, target_rgb, mask)
    # The "GT" target is passed in as the same target_rgb; this lets
    # the loss match the TDD notation without introducing a second
    # tensor. The two terms are summed as per TDD §7.2.
    l_rgb_gt = hole_l1(pred_rgb, target_rgb, mask)
    l_boundary = boundary_gradient_l1(
        pred_rgb, target_rgb, mask, band_px=boundary_band_px
    )
    l_rgb = l_rgb_teacher + l_rgb_gt + boundary_weight * l_boundary

    l_latent = torch.zeros((), device=pred_rgb.device, dtype=pred_rgb.dtype)
    if target_latent is not None and pred_latent is not None:
        l_latent = torch.nn.functional.mse_loss(
            pred_latent.float(), target_latent.float()
        )
    total = l_rgb + latent_weight * l_latent
    return {
        "loss_rgb": l_rgb,
        "loss_rgb_hole_teacher": l_rgb_teacher,
        "loss_rgb_hole_gt": l_rgb_gt,
        "loss_rgb_boundary": l_boundary,
        "loss_latent": l_latent,
        "loss": total,
    }
