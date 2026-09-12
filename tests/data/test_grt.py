"""GRT pure-function tests.

These cover the four bit-exact hand-crafted scenarios from the TDD
(§9.1) and the tie-survival rule (Hami paper). Reference outputs were
captured from Hami's torch implementation via
``tests/data/reference_grt_compute.py`` and stored in
``tests/data/hami_grt_reference.json``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from moebius_finetune.contracts import Direction
from moebius_finetune.data.grt import (
    apply_p99_normalization,
    compute_grt_mask,
    disparity_to_signed_normalized,
)


REF_PATH = Path(__file__).parent / "hami_grt_reference.json"
REF = json.loads(REF_PATH.read_text(encoding="utf-8"))


def _mask_array(name: str, kind: str) -> np.ndarray:
    entry = REF[name]
    return np.asarray(entry[kind], dtype=bool)


def _disparity(name: str) -> np.ndarray:
    entry = REF[name]
    return np.asarray(entry["disparity"], dtype=np.float32)


# ---------------------------------------------------------------------------
# 1. Two-plane: a far plane on the left, a near plane on the right.
# ---------------------------------------------------------------------------


def test_two_plane_r():
    H, W = REF["two_plane"]["shape"]
    got = compute_grt_mask((H, W), _disparity("two_plane"), Direction.R2L, dmax_px=1)
    np.testing.assert_array_equal(got, _mask_array("two_plane", "R"))


def test_two_plane_l():
    H, W = REF["two_plane"]["shape"]
    got = compute_grt_mask((H, W), _disparity("two_plane"), Direction.L2R, dmax_px=1)
    np.testing.assert_array_equal(got, _mask_array("two_plane", "L"))


# ---------------------------------------------------------------------------
# 2. Flat depth (all equal): only the OOB columns are holes, ties survive.
# ---------------------------------------------------------------------------


def test_flat_ties_r():
    H, W = REF["flat_ties"]["shape"]
    got = compute_grt_mask((H, W), _disparity("flat_ties"), Direction.R2L, dmax_px=1)
    np.testing.assert_array_equal(got, _mask_array("flat_ties", "R"))


def test_flat_ties_l():
    H, W = REF["flat_ties"]["shape"]
    got = compute_grt_mask((H, W), _disparity("flat_ties"), Direction.L2R, dmax_px=1)
    np.testing.assert_array_equal(got, _mask_array("flat_ties", "L"))


def test_perfectly_flat_zero_disparity_has_no_holes():
    """A perfectly flat disparity of 0 (all 0) produces no holes.
    Tests the explicit tie-survival semantics with a single disparity value.
    """
    d = np.zeros((4, 4), dtype=np.float32)
    for direction in (Direction.R2L, Direction.L2R):
        got = compute_grt_mask((4, 4), d, direction, dmax_px=1)
        assert not got.any(), f"flat d=0 {direction} should have no holes, got {got.sum()}"


# ---------------------------------------------------------------------------
# 3. Out-of-bounds: source pixels that project past the right edge are oob.
# ---------------------------------------------------------------------------


def test_out_of_bounds_r():
    H, W = REF["out_of_bounds"]["shape"]
    got = compute_grt_mask((H, W), _disparity("out_of_bounds"), Direction.R2L, dmax_px=1)
    np.testing.assert_array_equal(got, _mask_array("out_of_bounds", "R"))


def test_out_of_bounds_l():
    H, W = REF["out_of_bounds"]["shape"]
    got = compute_grt_mask((H, W), _disparity("out_of_bounds"), Direction.L2R, dmax_px=1)
    np.testing.assert_array_equal(got, _mask_array("out_of_bounds", "L"))


# ---------------------------------------------------------------------------
# 4. Gradient ramp L/R flip equivalence.
# ---------------------------------------------------------------------------


def test_gradient_ramp_r():
    H, W = REF["gradient_ramp"]["shape"]
    got = compute_grt_mask((H, W), _disparity("gradient_ramp"), Direction.R2L, dmax_px=1)
    np.testing.assert_array_equal(got, _mask_array("gradient_ramp", "R"))


def test_gradient_ramp_l():
    H, W = REF["gradient_ramp"]["shape"]
    got = compute_grt_mask((H, W), _disparity("gradient_ramp"), Direction.L2R, dmax_px=1)
    np.testing.assert_array_equal(got, _mask_array("gradient_ramp", "L"))


def test_l_r_flip_relationship():
    """The L mask equals hflip(R(hflip(d))) — Hami's L/R convention.

    Important: this is NOT the same as hflip(R(d)) for asymmetric
    disparity fields. We verify the documented L convention.
    """
    for name in ("two_plane", "out_of_bounds", "gradient_ramp"):
        d = _disparity(name)
        H, W = d.shape
        d_hflip = np.flip(d, axis=-1)
        r_on_hflip = compute_grt_mask((H, W), d_hflip, Direction.R2L, dmax_px=1)
        l = compute_grt_mask((H, W), d, Direction.L2R, dmax_px=1)
        np.testing.assert_array_equal(l, np.flip(r_on_hflip, axis=-1))


def test_l_r_flip_inverse_for_symmetric_d():
    """For left-right symmetric disparity, the L mask should be the
    horizontal flip of the R mask.
    """
    d = np.array([
        [0.0, 1.0, 2.0, 3.0, 3.0, 2.0, 1.0, 0.0],
        [0.0, 1.0, 2.0, 3.0, 3.0, 2.0, 1.0, 0.0],
        [0.0, 1.0, 2.0, 3.0, 3.0, 2.0, 1.0, 0.0],
        [0.0, 1.0, 2.0, 3.0, 3.0, 2.0, 1.0, 0.0],
    ], dtype=np.float32)
    H, W = d.shape
    r = compute_grt_mask((H, W), d, Direction.R2L, dmax_px=1)
    l = compute_grt_mask((H, W), d, Direction.L2R, dmax_px=1)
    np.testing.assert_array_equal(l, np.flip(r, axis=-1))


# ---------------------------------------------------------------------------
# 5. Single-point single-pixel occlusion.
# ---------------------------------------------------------------------------


def test_single_point_r():
    H, W = REF["single_point"]["shape"]
    got = compute_grt_mask((H, W), _disparity("single_point"), Direction.R2L, dmax_px=1)
    np.testing.assert_array_equal(got, _mask_array("single_point", "R"))


def test_single_point_l():
    H, W = REF["single_point"]["shape"]
    got = compute_grt_mask((H, W), _disparity("single_point"), Direction.L2R, dmax_px=1)
    np.testing.assert_array_equal(got, _mask_array("single_point", "L"))


# ---------------------------------------------------------------------------
# 6. Input validation.
# ---------------------------------------------------------------------------


def test_negative_disparity_raises():
    d = np.zeros((4, 4), dtype=np.float32)
    d[0, 0] = -0.1
    with pytest.raises(ValueError):
        compute_grt_mask((4, 4), d, Direction.R2L, dmax_px=1)


def test_non_finite_disparity_raises():
    d = np.zeros((4, 4), dtype=np.float32)
    d[1, 1] = float("nan")
    with pytest.raises(ValueError):
        compute_grt_mask((4, 4), d, Direction.R2L, dmax_px=1)


def test_shape_mismatch_raises():
    d = np.zeros((3, 4), dtype=np.float32)
    with pytest.raises(ValueError):
        compute_grt_mask((4, 4), d, Direction.R2L, dmax_px=1)


def test_dmax_zero_raises():
    d = np.zeros((4, 4), dtype=np.float32)
    with pytest.raises(ValueError):
        compute_grt_mask((4, 4), d, Direction.R2L, dmax_px=0)


def test_invalid_occlusion_test_raises():
    d = np.zeros((4, 4), dtype=np.float32)
    with pytest.raises(ValueError):
        compute_grt_mask((4, 4), d, Direction.R2L, dmax_px=1, occlusion_test="loose")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# 7. p99 normalisation (matches Hami.normalize_disparity).
# ---------------------------------------------------------------------------


def test_p99_normalization_uniform():
    out = apply_p99_normalization(np.full((4, 4), 2.0, dtype=np.float32))
    assert out.shape == (4, 4)
    assert out.dtype == np.float32
    # p99 == 2.0 for uniform, so result == 1.0 everywhere
    np.testing.assert_array_equal(out, np.ones_like(out))


def test_p99_normalization_zeros():
    out = apply_p99_normalization(np.zeros((4, 4), dtype=np.float32))
    # p99 == 0, <= eps; fall back to max==0; all zero
    np.testing.assert_array_equal(out, np.zeros((4, 4), dtype=np.float32))


def test_p99_normalization_tiny_uses_max():
    """When p99 is below eps but max is not, the max is used as anchor."""
    d = np.zeros((4, 4), dtype=np.float32)
    d[0, 0] = 1e-3
    out = apply_p99_normalization(d)
    # p99 is 0 (only one non-zero value, < 1% quantile), but max is 1e-3
    # so anchor = 1e-3, result = d / 1e-3 = [0, 1.0 at d[0,0]]
    assert out[0, 0] == pytest.approx(1.0)
    assert out.sum() == pytest.approx(1.0)


def test_p99_normalization_clamp_to_one():
    d = np.array([[0.5, 1.0, 2.0, 3.0]], dtype=np.float32)
    out = apply_p99_normalization(d)
    # p99 of [0.5, 1.0, 2.0, 3.0] ≈ 2.97
    assert out.max() <= 1.0
    assert out.min() >= 0.0


def test_disparity_to_signed_l_neg():
    d = np.array([[0.5, 1.0]], dtype=np.float32)
    out = disparity_to_signed_normalized(d, p99=1.0)
    # After p99 norm: [0.5, 1.0]
    np.testing.assert_array_equal(out, np.array([[0.5, 1.0]], dtype=np.float32))


def test_disparity_to_signed_unsupported_convention_raises():
    d = np.zeros((2, 2), dtype=np.float32)
    with pytest.raises(ValueError):
        disparity_to_signed_normalized(d, p99=1.0, sign_convention="R_neg_L_pos")


def test_disparity_to_signed_clamps_negative():
    d = np.array([[-0.5, 1.0]], dtype=np.float32)
    with pytest.raises(ValueError):
        disparity_to_signed_normalized(d, p99=1.0)


def test_disparity_to_signed_non_finite_raises():
    d = np.array([[1.0, float("nan")]], dtype=np.float32)
    with pytest.raises(ValueError):
        disparity_to_signed_normalized(d, p99=1.0)
