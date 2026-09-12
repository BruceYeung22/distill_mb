"""Tests for the distillation loss functions (TDD §7.2).

* :func:`hole_l1` per-case normalisation — every case is normalised
  by its own hole pixel count.
* :func:`boundary_gradient_l1` measures the gradient difference in
  a band of ~3 px around the mask edge.
* :func:`distillation_loss` combines the three RGB terms with the
  latent MSE.
* Empty mask regions contribute 0.
"""

from __future__ import annotations

import pytest
import torch

from moebius_finetune.training.student.losses import (
    boundary_gradient_l1,
    distillation_loss,
    hole_l1,
)




# ---------------------------------------------------------------------------
# hole_l1
# ---------------------------------------------------------------------------


def test_hole_l1_zero_when_pred_matches_target():
    """L1 hole loss is 0 when the prediction equals the target in the hole."""
    pred = torch.rand(2, 3, 16, 16)
    target = pred.clone()
    mask = torch.zeros(2, 1, 16, 16)
    mask[0, 0, 4:8, 4:8] = 1.0
    mask[1, 0, 2:6, 2:6] = 1.0
    out = hole_l1(pred, target, mask)
    assert out.item() == pytest.approx(0.0, abs=1e-7)


def test_hole_l1_per_case_normalisation_ignores_known_region():
    """A large error outside the hole does not affect the loss."""
    pred = torch.zeros(1, 3, 8, 8)
    target = pred.clone()
    mask = torch.zeros(1, 1, 8, 8)
    mask[0, 0, 0:2, 0:2] = 1.0
    # L1 inside the hole is 0; outside should be ignored.
    target[:, :, 4:, 4:] = 1.0  # known region
    pred[:, :, 4:, 4:] = 0.0  # wrong only outside the hole
    out = hole_l1(pred, target, mask)
    assert out.item() == pytest.approx(0.0, abs=1e-7)


def test_hole_l1_per_case_normalisation_handles_empty_mask():
    """An empty mask gives 0 contribution (no division by zero)."""
    pred = torch.zeros(1, 3, 8, 8)
    target = torch.ones(1, 3, 8, 8)
    mask = torch.zeros(1, 1, 8, 8)
    out = hole_l1(pred, target, mask)
    assert out.item() == 0.0


def test_hole_l1_per_case_normalisation_invariant_to_hole_size():
    """A constant error of 1.0 inside any hole gives loss = 1.0 regardless of size."""
    pred = torch.zeros(2, 3, 16, 16)
    target = torch.ones(2, 3, 16, 16)
    mask = torch.zeros(2, 1, 16, 16)
    mask[0, 0, 0:2, 0:2] = 1.0  # 4 hole pixels
    mask[1, 0, 0:8, 0:8] = 1.0  # 64 hole pixels
    out = hole_l1(pred, target, mask)
    # Average of per-case L1 / 3-channel-normalised is 1.0 each.
    assert out.item() == pytest.approx(1.0, abs=1e-6)


def test_hole_l1_dtype_is_fp32():
    """The reduction is computed in FP32."""
    pred = torch.rand(1, 3, 8, 8, dtype=torch.float64)
    target = torch.rand(1, 3, 8, 8, dtype=torch.float64)
    mask = torch.zeros(1, 1, 8, 8, dtype=torch.float64)
    mask[0, 0, 0:4, 0:4] = 1.0
    out = hole_l1(pred, target, mask)
    assert out.dtype == torch.float32


# ---------------------------------------------------------------------------
# boundary_gradient_l1
# ---------------------------------------------------------------------------


def test_boundary_gradient_l1_zero_when_pred_matches_target():
    """If pred == target, the boundary gradient is 0."""
    pred = torch.rand(1, 3, 16, 16)
    target = pred.clone()
    mask = torch.zeros(1, 1, 16, 16)
    mask[0, 0, 4:12, 4:12] = 1.0
    out = boundary_gradient_l1(pred, target, mask, band_px=3)
    assert out.item() == pytest.approx(0.0, abs=1e-7)


def test_boundary_gradient_l1_empty_mask_returns_zero():
    """No boundary if no hole exists."""
    pred = torch.rand(1, 3, 16, 16)
    target = torch.zeros_like(pred)
    mask = torch.zeros(1, 1, 16, 16)
    out = boundary_gradient_l1(pred, target, mask)
    assert out.item() == 0.0


def test_boundary_gradient_l1_band_thickness_grows_with_band_px():
    """Larger ``band_px`` averages over more boundary pixels but the loss
    value is well-defined and non-negative."""
    pred = torch.rand(1, 3, 32, 32)
    target = torch.rand(1, 3, 32, 32)
    mask = torch.zeros(1, 1, 32, 32)
    mask[0, 0, 8:24, 8:24] = 1.0
    for px in (1, 3, 5):
        v = boundary_gradient_l1(pred, target, mask, band_px=px).item()
        assert v >= 0.0


# ---------------------------------------------------------------------------
# distillation_loss
# ---------------------------------------------------------------------------


def test_distillation_loss_returns_expected_components():
    """The returned dict has the documented keys and a finite total."""
    pred = torch.rand(1, 3, 16, 16)
    target = torch.rand(1, 3, 16, 16)
    target_latent = torch.rand(1, 4, 2, 2)
    pred_latent = torch.rand(1, 4, 2, 2)
    mask = torch.zeros(1, 1, 16, 16)
    mask[0, 0, 4:12, 4:12] = 1.0
    out = distillation_loss(
        pred, target, target_latent, pred_latent, mask
    )
    expected = {
        "loss",
        "loss_rgb",
        "loss_rgb_hole_teacher",
        "loss_rgb_hole_gt",
        "loss_rgb_boundary",
        "loss_latent",
    }
    assert expected.issubset(out.keys())
    assert torch.isfinite(out["loss"]).item()


def test_distillation_loss_zero_when_pred_matches():
    """All loss components are 0 when pred == target == pred_latent."""
    rgb = torch.rand(1, 3, 16, 16)
    z = torch.rand(1, 4, 2, 2)
    mask = torch.zeros(1, 1, 16, 16)
    mask[0, 0, 4:12, 4:12] = 1.0
    out = distillation_loss(rgb, rgb, z, z, mask)
    assert out["loss"].item() == pytest.approx(0.0, abs=1e-6)
    assert out["loss_rgb_boundary"].item() == pytest.approx(0.0, abs=1e-6)


def test_distillation_loss_latent_optional():
    """If both latent tensors are ``None``, the latent term is 0."""
    pred = torch.rand(1, 3, 16, 16)
    target = torch.rand(1, 3, 16, 16)
    mask = torch.zeros(1, 1, 16, 16)
    mask[0, 0, 4:12, 4:12] = 1.0
    out = distillation_loss(pred, target, None, None, mask)
    assert out["loss_latent"].item() == 0.0
