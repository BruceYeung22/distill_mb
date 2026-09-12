"""Tests for the PixelStudent-v0 (TDD §6.2).

Covers:

* Synthetic forward at 128² and 512² (the contract resolution).
* Backward and gradient flow.
* 5-channel condition contract and noise-injection contract.
* Save/load round-trip.
* predict_candidate / inpaint composition with the contracts.inpaint
  helper (known region preserved).
* Hand-rolled MAC check against the budget tool.
"""

from __future__ import annotations

import math
import os
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from moebius_finetune.contracts import (
    ConditionBatch,
    inpaint as contracts_inpaint,
    validate_condition,
)
from moebius_finetune.data.synthetic import make_synthetic_batch
from moebius_finetune.deployment.budget import compute_algorithm_macs
from moebius_finetune.students import PixelStudentV0




# Use a small H for the forward/backward tests so they run in <1s.
SMALL_H = 128


def _batch(H: int = SMALL_H) -> ConditionBatch:
    return make_synthetic_batch(H=H, B=1, hole_spec="wide", seed=0)


# ---------------------------------------------------------------------------
# Construction & forward
# ---------------------------------------------------------------------------


def test_pixel_student_constructs_with_spec_defaults():
    model = PixelStudentV0()
    assert model.encoder_widths == (24, 48, 96)
    assert model.bottleneck_width == 160
    assert model.decoder_widths == (96, 48, 24)
    assert model.in_channels == 5
    assert model.out_channels == 3


def test_pixel_student_forward_at_128():
    """Synthetic forward at H=128 (still valid since H/8 = 16)."""
    model = PixelStudentV0()
    model.eval()
    batch = _batch(H=128)
    cond5 = np.concatenate(
        [batch.rgb_hole, batch.hole_mask, batch.depth_hole], axis=1
    )
    cond5_t = torch.from_numpy(cond5)
    noise_t = torch.from_numpy(batch.noise)
    with torch.no_grad():
        out = model(cond5_t, noise_t)
    assert out.shape == (1, 3, 128, 128)
    assert torch.isfinite(out).all()


def test_pixel_student_forward_at_512():
    """Full-resolution (512²) synthetic forward."""
    model = PixelStudentV0()
    model.eval()
    batch = make_synthetic_batch(H=512, B=1, hole_spec="wide", seed=0)
    cond5 = np.concatenate(
        [batch.rgb_hole, batch.hole_mask, batch.depth_hole], axis=1
    )
    cond5_t = torch.from_numpy(cond5)
    noise_t = torch.from_numpy(batch.noise)
    with torch.no_grad():
        out = model(cond5_t, noise_t)
    assert out.shape == (1, 3, 512, 512)
    assert torch.isfinite(out).all()


def test_pixel_student_rejects_wrong_condition_channels():
    model = PixelStudentV0()
    model.eval()
    bad_cond = torch.randn(1, 4, 128, 128)  # 4 instead of 5
    noise = torch.randn(1, 4, 16, 16)
    with pytest.raises(ValueError):
        model(bad_cond, noise)


def test_pixel_student_rejects_wrong_noise_spatial():
    model = PixelStudentV0()
    model.eval()
    cond = torch.randn(1, 5, 128, 128)
    bad_noise = torch.randn(1, 4, 8, 8)  # should be 16x16 at H=128
    with pytest.raises(ValueError):
        model(cond, bad_noise)


# ---------------------------------------------------------------------------
# Backward / gradient flow
# ---------------------------------------------------------------------------


def test_pixel_student_backward_at_128():
    model = PixelStudentV0()
    model.train()
    batch = _batch(H=128)
    cond5 = np.concatenate(
        [batch.rgb_hole, batch.hole_mask, batch.depth_hole], axis=1
    )
    cond5_t = torch.from_numpy(cond5)
    noise_t = torch.from_numpy(batch.noise)
    out = model(cond5_t, noise_t)
    target = torch.zeros_like(out)
    loss = ((out - target) ** 2).mean()
    loss.backward()
    # Every trainable parameter should have a non-empty gradient.
    n_with_grad = 0
    n_total = 0
    for p in model.parameters():
        if p.requires_grad:
            n_total += 1
            if p.grad is not None and torch.isfinite(p.grad).all():
                n_with_grad += 1
    assert n_total > 0
    assert n_with_grad == n_total


def test_pixel_student_noise_proj_receives_gradient():
    """The noise-projection 1x1 conv is updated by the gradient."""
    model = PixelStudentV0()
    model.train()
    batch = _batch(H=128)
    cond5 = np.concatenate(
        [batch.rgb_hole, batch.hole_mask, batch.depth_hole], axis=1
    )
    cond5_t = torch.from_numpy(cond5)
    noise_t = torch.from_numpy(batch.noise)
    out = model(cond5_t, noise_t)
    out.sum().backward()
    np_grad = model.noise_proj.weight.grad
    assert np_grad is not None
    assert torch.isfinite(np_grad).all()
    assert np_grad.abs().sum() > 0


# ---------------------------------------------------------------------------
# predict_candidate / inpaint (numpy-level integration with contracts)
# ---------------------------------------------------------------------------


def test_pixel_student_predict_candidate_shape():
    model = PixelStudentV0()
    model.eval()
    batch = _batch(H=128)
    candidate = model.predict_candidate(batch, batch.noise)
    assert candidate.shape == batch.rgb_hole.shape
    assert candidate.dtype == np.float32
    assert np.isfinite(candidate).all()


def test_pixel_student_inpaint_preserves_known_region():
    """The known region of the output equals the input rgb_hole (clamped)."""
    model = PixelStudentV0()
    model.eval()
    batch = make_synthetic_batch(H=128, B=1, hole_spec="wide", seed=1)
    out = model.inpaint(batch, batch.noise)
    assert out.shape == batch.rgb_hole.shape
    # Outside the hole the output must equal rgb_hole exactly.
    mask = batch.hole_mask  # [1, 1, H, W]
    known = mask == 0
    np.testing.assert_array_equal(
        out[:, :, :, :][np.broadcast_to(known, out.shape)],
        batch.rgb_hole[np.broadcast_to(known, batch.rgb_hole.shape)],
    )


def test_pixel_student_inpaint_clamps_to_unit_range():
    model = PixelStudentV0()
    model.eval()
    # Use a "full" hole so the inpaint returns only the (clamped) prediction.
    batch = make_synthetic_batch(H=128, B=1, hole_spec="full", seed=0)
    out = model.inpaint(batch, batch.noise)
    assert out.min() >= 0.0
    assert out.max() <= 1.0


def test_pixel_student_inpaint_empty_mask_returns_input():
    """Empty-mask inpaint must equal the input (no clamp needed)."""
    model = PixelStudentV0()
    model.eval()
    batch = make_synthetic_batch(H=128, B=1, hole_spec="empty", seed=0)
    out = model.inpaint(batch, batch.noise)
    np.testing.assert_array_equal(out, batch.rgb_hole)


# ---------------------------------------------------------------------------
# Save / load round-trip
# ---------------------------------------------------------------------------


def test_pixel_student_save_load_roundtrip(tmp_path: Path):
    model = PixelStudentV0()
    model.eval()
    batch = _batch(H=128)
    # Capture the candidate before saving.
    before = model.predict_candidate(batch, batch.noise)
    # Save and load.
    ckpt = tmp_path / "pixel.pt"
    torch.save(model.state_dict(), ckpt)
    model2 = PixelStudentV0()
    model2.load_state_dict(torch.load(ckpt, map_location="cpu"))
    model2.eval()
    after = model2.predict_candidate(batch, batch.noise)
    np.testing.assert_allclose(before, after, atol=1e-6)


# ---------------------------------------------------------------------------
# Budget hand-roll check
# ---------------------------------------------------------------------------


def test_pixel_student_algorithm_macs_within_budget():
    """Pixel MACs must fit the 8 GMACs global budget (TDD §8.2)."""
    model = PixelStudentV0()
    report = compute_algorithm_macs(model, (1, 5, 512, 512))
    gmacs = report["total_macs_g"]
    assert gmacs <= 8.0, f"pixel {gmacs:.3f} GMACs > 8 GMACs budget"
    # The spec's initial hand-roll is 3.07 GMACs. The actual programmatic
    # count differs by <10% (TDD explicitly notes the hand-roll is
    # approximate). We allow a wider band for the actual count.
    assert 2.76 <= gmacs <= 3.40, (
        f"pixel {gmacs:.3f} GMACs is outside the ±10% band of the spec's "
        f"3.07 GMACs hand-roll. The spec is an approximation; the "
        f"programmatic count is the ground truth."
    )


def test_pixel_student_weight_count_close_to_spec():
    """Pixel weight count must be within ±10% of the spec's 0.92M."""
    model = PixelStudentV0()
    report = compute_algorithm_macs(model, (1, 5, 512, 512))
    n_params = report["total_weight_params"]
    # 0.92M ±10% = [0.828M, 1.012M]
    assert 0.828e6 <= n_params <= 1.012e6, (
        f"pixel weight count {n_params:,} is outside ±10% of 0.92M"
    )
