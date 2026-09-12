"""Tests for the synthetic fixtures."""

from __future__ import annotations

import numpy as np
import pytest

from moebius_finetune.contracts import (
    ConditionBatch,
    ConditionContractError,
    inpaint,
    validate_condition,
)
from moebius_finetune.data.synthetic import make_synthetic_batch


@pytest.mark.parametrize(
    "hole_spec", ["thin", "wide", "full", "empty"]
)
def test_synthetic_batch_honours_contract(hole_spec):
    batch = make_synthetic_batch(H=512, B=1, hole_spec=hole_spec, seed=0)
    assert isinstance(batch, ConditionBatch)
    # Should not raise
    validate_condition(batch)


@pytest.mark.parametrize("hole_spec", ["thin", "wide", "full", "empty"])
def test_synthetic_batch_shapes(hole_spec):
    H = 512
    B = 2
    batch = make_synthetic_batch(H=H, B=B, hole_spec=hole_spec)
    assert batch.rgb_hole.shape == (B, 3, H, H)
    assert batch.hole_mask.shape == (B, 1, H, H)
    assert batch.depth_hole.shape == (B, 1, H, H)
    assert batch.noise.shape == (B, 4, H // 8, H // 8)
    assert batch.rgb_hole.dtype == np.float32
    assert batch.hole_mask.dtype == np.float32
    assert batch.depth_hole.dtype == np.float32
    assert batch.noise.dtype == np.float32


def test_thin_hole_is_a_strip():
    batch = make_synthetic_batch(H=64, B=1, hole_spec="thin")
    mask = batch.hole_mask[0, 0]
    # All mask columns should be either 0 (background) or 1 (vertical strip)
    col_sums = mask.sum(axis=0)
    non_zero = col_sums[col_sums > 0]
    assert (non_zero == 64).all(), "thin hole should be a full-height vertical strip"
    # Background columns
    assert (col_sums[col_sums == 0]).size > 0


def test_wide_hole_is_a_rectangle():
    batch = make_synthetic_batch(H=64, B=1, hole_spec="wide")
    mask = batch.hole_mask[0, 0]
    # The hole should be a contiguous rectangle
    rows_with_hole = np.where(mask.any(axis=1))[0]
    cols_with_hole = np.where(mask.any(axis=0))[0]
    assert len(rows_with_hole) > 0
    assert len(cols_with_hole) > 0
    # All pixels in the bounding box should be in the mask
    sub = mask[
        rows_with_hole.min() : rows_with_hole.max() + 1,
        cols_with_hole.min() : cols_with_hole.max() + 1,
    ]
    assert sub.all()


def test_full_hole_covers_everything():
    batch = make_synthetic_batch(H=64, B=1, hole_spec="full")
    assert batch.hole_mask.sum() == 64 * 64
    assert (batch.rgb_hole == 0).all()
    assert (batch.depth_hole == 0).all()


def test_empty_hole_has_no_mask():
    batch = make_synthetic_batch(H=64, B=1, hole_spec="empty")
    assert batch.hole_mask.sum() == 0


def test_hole_pixels_are_zero_in_rgb_and_depth():
    batch = make_synthetic_batch(H=128, B=1, hole_spec="wide", seed=42)
    mask_broadcast = np.broadcast_to(batch.hole_mask, batch.rgb_hole.shape)
    # Pixels in the hole must be 0 in rgb and depth
    assert float(np.abs(batch.rgb_hole * batch.hole_mask).max()) == 0.0
    assert float(np.abs(batch.depth_hole * batch.hole_mask).max()) == 0.0
    # And the inverse: background pixels in rgb are non-zero (unless
    # the random draw was exactly 0, which is essentially impossible)
    bg = 1.0 - batch.hole_mask
    assert (batch.rgb_hole * bg).max() > 0.0


def test_inpaint_preserves_known_region():
    """When fed to inpaint, the known region must be preserved exactly."""
    batch = make_synthetic_batch(H=64, B=1, hole_spec="thin", seed=0)
    candidate = np.full_like(batch.rgb_hole, 0.42)
    out = inpaint(batch, candidate)
    mask = batch.hole_mask
    mask_broadcast = np.broadcast_to(mask, batch.rgb_hole.shape)
    # Hole region: equals the candidate (0.42)
    np.testing.assert_allclose(out[mask_broadcast == 1], 0.42, atol=1e-6)
    # Known region: equals the original rgb_hole
    np.testing.assert_array_equal(out[mask_broadcast == 0], batch.rgb_hole[mask_broadcast == 0])


def test_synthetic_batch_rejects_non_multiple_of_8():
    with pytest.raises(ValueError):
        make_synthetic_batch(H=63, hole_spec="thin")


def test_synthetic_batch_unknown_hole_spec():
    with pytest.raises(ValueError):
        make_synthetic_batch(H=64, hole_spec="bogus")  # type: ignore[arg-type]


def test_signed_depth_default_is_signed():
    batch = make_synthetic_batch(H=64, B=1, hole_spec="empty", signed_depth=True)
    # depth_hole has both positive and negative values
    assert batch.depth_hole.min() < 0.0
    assert batch.depth_hole.max() > 0.0


def test_noise_is_deterministic_for_seed():
    a = make_synthetic_batch(H=64, B=1, hole_spec="thin", seed=123)
    b = make_synthetic_batch(H=64, B=1, hole_spec="thin", seed=123)
    np.testing.assert_array_equal(a.noise, b.noise)


def test_noise_changes_with_seed():
    a = make_synthetic_batch(H=64, B=1, hole_spec="thin", seed=0)
    b = make_synthetic_batch(H=64, B=1, hole_spec="thin", seed=1)
    assert not np.array_equal(a.noise, b.noise)
