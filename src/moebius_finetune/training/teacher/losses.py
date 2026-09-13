"""Epsilon-prediction losses for the depth-conditioned teacher.

TDD §5.3 defines:

* prediction target: epsilon,
* loss: hole_eps_MSE + 0.1 * known_eps_MSE,
* normalisation: per-case, divide by the number of hole pixels
  (and known pixels) before batch-averaging,
* empty mask contributes zero.

All reductions are computed in **float32** to keep the loss numerically
stable. The inputs are expected to be float32 already, but we cast
defensively so a stray fp16 model does not poison the gradient.
"""

from __future__ import annotations

import torch
from torch import Tensor


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class LossConfigError(ValueError):
    """Raised when a loss function is given a malformed input."""


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------


def _as_fp32(x: Tensor, name: str) -> Tensor:
    if not isinstance(x, Tensor):
        raise LossConfigError(f"{name} must be a torch.Tensor, got {type(x).__name__}")
    if x.dtype != torch.float32:
        return x.to(dtype=torch.float32)
    return x


def epsilon_mse(
    pred_eps: Tensor,
    target_eps: Tensor,
    mask: Tensor,
) -> Tensor:
    """MSE on epsilon predictions, per-case normalised on ``mask`` pixels.

    Parameters
    ----------
    pred_eps
        ``[B, ...]`` epsilon prediction.
    target_eps
        ``[B, ...]`` ground-truth epsilon (from the scheduler's
        ``add_noise``).
    mask
        ``[B, ...]`` float mask with values in ``[0, 1]``. The MSE is
        averaged over ``mask`` pixels per case, then averaged across
        the batch. Empty cases contribute ``0`` (no gradient).

    Returns
    -------
    torch.Tensor
        Scalar tensor. ``0`` (with ``requires_grad=False``) when every
        case has zero mask coverage.
    """
    pred_eps = _as_fp32(pred_eps, "pred_eps")
    target_eps = _as_fp32(target_eps, "target_eps")
    mask = _as_fp32(mask, "mask")

    if pred_eps.shape != target_eps.shape:
        raise LossConfigError(
            f"pred_eps shape {tuple(pred_eps.shape)} != target_eps "
            f"shape {tuple(target_eps.shape)}"
        )
    if mask.ndim != pred_eps.ndim:
        raise LossConfigError(
            f"mask ndim {mask.ndim} != pred_eps ndim {pred_eps.ndim}"
        )
    # Mask may have its channel dim == 1 (broadcast over channels) or
    # match pred_eps exactly. We reject any other mismatch.
    if mask.shape[0] != pred_eps.shape[0]:
        raise LossConfigError(
            f"mask batch dim {mask.shape[0]} != pred_eps batch dim "
            f"{pred_eps.shape[0]}"
        )
    if mask.shape[1] not in (1, pred_eps.shape[1]):
        raise LossConfigError(
            f"mask channel dim {mask.shape[1]} is not 1 or "
            f"{pred_eps.shape[1]}"
        )
    for d in range(2, pred_eps.ndim):
        if mask.shape[d] != pred_eps.shape[d]:
            raise LossConfigError(
                f"mask shape {tuple(mask.shape)} does not match "
                f"pred_eps shape {tuple(pred_eps.shape)} on dim {d}"
            )
    mask_b = mask.expand_as(pred_eps) if mask.shape[1] == 1 else mask

    diff2 = (pred_eps - target_eps) ** 2
    masked = diff2 * mask_b

    B = pred_eps.shape[0]
    # Per-case denominator: sum of mask over all non-batch dims.
    flat_mask = mask_b.reshape(B, -1).sum(dim=1)
    flat_diff = masked.reshape(B, -1).sum(dim=1)
    # Avoid division by zero: where flat_mask == 0 we return 0 and
    # never accumulate.
    valid = flat_mask > 0
    if not bool(valid.any()):
        return torch.zeros((), dtype=torch.float32, device=pred_eps.device)
    # Keep the complete B dimension.  Indexing to valid cases before
    # ``where`` makes mixed empty/non-empty batches shape-incompatible and
    # would also drop empty cases from the documented batch mean.
    per_case = torch.where(
        valid,
        flat_diff / flat_mask.clamp_min(1e-12),
        torch.zeros_like(flat_diff),
    )
    return per_case.mean()


def combined_epsilon_loss(
    pred_eps: Tensor,
    target_eps: Tensor,
    hole_mask: Tensor,
    *,
    known_weight: float = 0.1,
) -> Tensor:
    """hole_eps_MSE + ``known_weight`` * known_eps_MSE.

    The "known" region is the complement of ``hole_mask``. Both terms
    use the same per-case normalisation so big holes do not dominate
    small ones (TDD §5.3 / §7.2).
    """
    if known_weight < 0:
        raise LossConfigError(
            f"known_weight must be non-negative, got {known_weight}"
        )
    hole_term = epsilon_mse(pred_eps, target_eps, hole_mask)
    known_mask = (1.0 - hole_mask).clamp(0.0, 1.0)
    known_term = epsilon_mse(pred_eps, target_eps, known_mask)
    return hole_term + float(known_weight) * known_term


__all__ = ["combined_epsilon_loss", "epsilon_mse", "LossConfigError"]
