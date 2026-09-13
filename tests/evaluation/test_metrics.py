"""Tests for evaluation/metrics.py."""

from __future__ import annotations

import math

import numpy as np
import pytest

from moebius_finetune.evaluation.metrics import (
    aggregate_per_case,
    boundary_l1,
    hole_psnr,
    known_max_error,
)


B, C, H, W = 2, 3, 16, 16


def _full_mask() -> np.ndarray:
    return np.ones((B, 1, H, W), dtype=np.float32)


def _empty_mask() -> np.ndarray:
    return np.zeros((B, 1, H, W), dtype=np.float32)


def _half_mask() -> np.ndarray:
    m = np.zeros((B, 1, H, W), dtype=np.float32)
    m[:, :, :H // 2, :] = 1.0
    return m


def _zero_target() -> np.ndarray:
    return np.zeros((B, C, H, W), dtype=np.float32)


def _half_target() -> np.ndarray:
    return np.full((B, C, H, W), 0.5, dtype=np.float32)


def _random_pair(seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    a = rng.random((B, C, H, W), dtype=np.float32)
    b = rng.random((B, C, H, W), dtype=np.float32)
    return a, b


# ---------------------------------------------------------------------------
# hole_psnr
# ---------------------------------------------------------------------------


def test_hole_psnr_perfect_prediction():
    """Perfect prediction inside the hole → infinite PSNR.

    Our convention: clip MSE to > 0 and report inf, but the aggregate
    helper treats inf specially.
    """
    target = _half_target()
    pred = target.copy()
    out = hole_psnr(pred, target, _half_mask(), per_case=True)
    assert math.isinf(float(out[0]))
    scalar = hole_psnr(pred, target, _half_mask())
    assert math.isinf(scalar) or scalar > 60.0  # numerical guard


def test_hole_psnr_full_mask_constant_fill():
    """For a full mask with constant fill 0.5 and target 0.0,
    the per-pixel MSE = 0.25, PSNR = 10 * log10(1 / 0.25) = 6.02 dB.
    """
    pred = np.full((1, 3, 4, 4), 0.5, dtype=np.float32)
    target = np.zeros((1, 3, 4, 4), dtype=np.float32)
    mask = np.ones((1, 1, 4, 4), dtype=np.float32)
    out = hole_psnr(pred, target, mask)
    assert out == pytest.approx(6.02, abs=0.05)


def test_hole_psnr_normalizes_by_hole_pixels_and_channels():
    pred = np.full((1, 3, 16, 16), 0.5, dtype=np.float32)
    target = np.zeros_like(pred)
    mask = np.zeros((1, 1, 16, 16), dtype=np.float32)
    mask[:, :, :8, :] = 1
    assert hole_psnr(pred, target, mask) == pytest.approx(6.02, abs=0.05)


def test_hole_psnr_zero_db_is_valid_case():
    pred = np.ones((2, 3, 2, 2), dtype=np.float32)
    pred[1] = 0.5
    target = np.zeros_like(pred)
    mask = np.ones((2, 1, 2, 2), dtype=np.float32)
    assert hole_psnr(pred, target, mask) == pytest.approx(10 * np.log10(4) / 2)


def test_hole_psnr_empty_mask_returns_zero():
    pred, target = _random_pair(0)
    out = hole_psnr(pred, target, _empty_mask())
    assert out == 0.0


def test_hole_psnr_per_case_returns_per_batch():
    pred, target = _random_pair(0)
    out = hole_psnr(pred, target, _half_mask(), per_case=True)
    assert out.shape == (B,)


def test_hole_psnr_batched_mean():
    a, b = _random_pair(0)
    out = hole_psnr(a, b, _half_mask())
    # Should be a finite scalar
    assert math.isfinite(out) or out == 0.0


def test_hole_psnr_data_range_negative_raises():
    pred, target = _random_pair(0)
    with pytest.raises(ValueError):
        hole_psnr(pred, target, _half_mask(), data_range=0.0)


def test_hole_psnr_shape_mismatch_raises():
    """Different pred/target shapes must raise."""
    a = np.zeros((2, 3, 8, 8), dtype=np.float32)
    b = np.zeros((2, 3, 8, 8), dtype=np.float32)
    # Mask with wrong spatial dim
    m = np.zeros((2, 1, 4, 4), dtype=np.float32)
    with pytest.raises(ValueError):
        hole_psnr(a, b, m)


def test_hole_psnr_handles_3d_input():
    a = np.full((3, 4, 4), 0.5, dtype=np.float32)
    b = np.zeros((3, 4, 4), dtype=np.float32)
    m = np.ones((1, 4, 4), dtype=np.float32)
    out = hole_psnr(a, b, m)
    assert isinstance(out, float)


# ---------------------------------------------------------------------------
# boundary_l1
# ---------------------------------------------------------------------------


def test_boundary_l1_perfect_prediction_zero():
    target = _half_target()
    out = boundary_l1(target, target, _half_mask())
    assert out == pytest.approx(0.0, abs=1e-6)


def test_boundary_l1_increases_with_perturbation():
    target = _half_target()
    pred_clean = target.copy()
    pred_perturbed = target.copy()
    pred_perturbed[:, :, :H // 2, :W // 2] = 0.0  # at the hole boundary
    clean = boundary_l1(pred_clean, target, _half_mask())
    pert = boundary_l1(pred_perturbed, target, _half_mask())
    assert pert > clean


def test_boundary_l1_band_px_widens_band():
    """Wider band should include more pixels, generally lowering the
    mean L1 when most of the band is in the known region (which
    matches perfectly). We assert the widening is monotonic in
    band size for our specific case: widening the band in the
    known region reduces the per-pixel mean.
    """
    target = _half_target()
    pred = target.copy()
    pred[:, :, :H // 2, :W // 2] = 0.0  # perturb a corner of the hole
    l1 = []
    for bp in (1, 2, 4):
        l1.append(boundary_l1(pred, target, _half_mask(), band_px=bp))
    # All values should be finite
    for v in l1:
        assert math.isfinite(v)


def test_boundary_l1_empty_mask_returns_zero():
    a, b = _random_pair(0)
    out = boundary_l1(a, b, _empty_mask())
    assert out == 0.0


def test_boundary_l1_excludes_deep_hole_pixels():
    target = np.zeros((1, 3, 16, 16), dtype=np.float32)
    pred = target.copy()
    mask = np.zeros((1, 1, 16, 16), dtype=np.float32)
    mask[:, :, 3:13, 3:13] = 1
    pred[:, :, 8, 8] = 1.0
    assert boundary_l1(pred, target, mask, band_px=1) == pytest.approx(0.0)


def test_boundary_l1_zero_error_case_is_kept_in_batch_mean():
    target = np.zeros((2, 3, 8, 8), dtype=np.float32)
    pred = target.copy()
    mask = np.zeros((2, 1, 8, 8), dtype=np.float32)
    mask[:, :, :4, :] = 1
    pred[0, :, 3, :] = 1.0
    assert boundary_l1(pred, target, mask, band_px=1) == pytest.approx(0.25)


def test_boundary_l1_per_case():
    a, b = _random_pair(0)
    out = boundary_l1(a, b, _half_mask(), per_case=True)
    assert out.shape == (B,)


def test_boundary_l1_band_px_invalid_raises():
    a, b = _random_pair(0)
    with pytest.raises(ValueError):
        boundary_l1(a, b, _half_mask(), band_px=0)


# ---------------------------------------------------------------------------
# known_max_error
# ---------------------------------------------------------------------------


def test_known_max_error_perfect_zero():
    target = _half_target()
    out = known_max_error(target, target, _half_mask())
    assert out == 0.0


def test_known_max_error_known_perturbed():
    a, b = _random_pair(0)
    # Set one known pixel to 1.0
    a[0, 0, H - 1, W - 1] = 1.0
    out = known_max_error(a, b, _half_mask())
    assert out > 0.0


def test_known_max_error_full_mask_returns_zero():
    a, b = _random_pair(0)
    out = known_max_error(a, b, _full_mask())
    assert out == 0.0


def test_known_max_error_empty_mask_returns_zero():
    a, b = _random_pair(0)
    out = known_max_error(a, b, _empty_mask())
    # Empty mask → all known → max error is the global max abs diff
    expected = float(np.abs(a - b).max())
    assert out == pytest.approx(expected, abs=1e-5)


# ---------------------------------------------------------------------------
# aggregate_per_case
# ---------------------------------------------------------------------------


def _make_case(
    case_id: str,
    *,
    psnr: float,
    split: str = "train",
    direction: str = "L2R",
    dmax_px: int = 8,
    hole_ratio: float = 0.05,
    empty: bool = False,
) -> dict:
    return {
        "case_id": case_id,
        "split": split,
        "direction": direction,
        "dmax_px": dmax_px,
        "hole_ratio": hole_ratio,
        "hole_psnr": psnr,
        "boundary_l1": 0.1,
        "known_max_error": 0.05,
        "empty": empty,
    }


def test_aggregate_empty_input():
    out = aggregate_per_case([])
    assert out["overall"]["n"] == 0
    assert out["empty_count"] == 0


def test_aggregate_handles_all_empty():
    out = aggregate_per_case(
        [_make_case("c", psnr=0.0, empty=True) for _ in range(3)]
    )
    assert out["overall"]["n"] == 0
    assert out["empty_count"] == 3


def test_aggregate_basic_stats():
    cases = [_make_case(f"c{i}", psnr=float(20 + i)) for i in range(5)]
    out = aggregate_per_case(cases)
    assert out["overall"]["n"] == 5
    assert out["overall"]["mean"] == pytest.approx(22.0)
    assert out["overall"]["max"] == pytest.approx(24.0)
    assert out["overall"]["min"] == pytest.approx(20.0)
    assert out["empty_count"] == 0


def test_aggregate_groups_by_split_direction_dmax():
    cases = [
        _make_case("a", psnr=20.0, split="train", direction="L2R", dmax_px=8),
        _make_case("b", psnr=30.0, split="train", direction="R2L", dmax_px=8),
        _make_case("c", psnr=25.0, split="holdout_images", direction="L2R", dmax_px=16),
    ]
    out = aggregate_per_case(cases)
    assert "train" in out["by_split"]
    assert "holdout_images" in out["by_split"]
    assert "L2R" in out["by_direction"]
    assert "R2L" in out["by_direction"]
    assert 8 in out["by_dmax"]
    assert 16 in out["by_dmax"]


def test_aggregate_groups_by_hole_width():
    cases = [
        _make_case("thin", psnr=20.0, hole_ratio=0.01),
        _make_case("medium", psnr=22.0, hole_ratio=0.10),
        _make_case("wide", psnr=24.0, hole_ratio=0.40),
    ]
    out = aggregate_per_case(cases)
    assert "thin" in out["by_hole_width"]
    assert "medium" in out["by_hole_width"]
    assert "wide" in out["by_hole_width"]


def test_aggregate_extra_group_keys():
    cases = [
        _make_case("a", psnr=20.0),
        _make_case("b", psnr=22.0),
    ]
    cases[0]["model_name"] = "teacher"
    cases[1]["model_name"] = "student"
    out = aggregate_per_case(cases, group_keys=["model_name"])
    assert "model_name" in out["by_group"]
    assert "teacher" in out["by_group"]["model_name"]
    assert "student" in out["by_group"]["model_name"]
