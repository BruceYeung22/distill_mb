"""Tests for depth_normalization.py."""

from __future__ import annotations

import numpy as np
import pytest

from moebius_finetune.contracts import DepthNormalization
from moebius_finetune.data.depth_normalization import (
    DepthNormalizationApplier,
    apply_depth_normalization,
)


def test_from_disparity_records_p99():
    d = np.array([[0.5, 1.0, 2.0, 4.0]], dtype=np.float32)
    applier = DepthNormalizationApplier.from_disparity(d)
    # np.quantile uses linear interpolation; p99 of [0.5, 1.0, 2.0, 4.0]
    # lands at index 2.97 → 2.0 + 0.97 * (4.0 - 2.0) = 3.94.
    assert applier.normalization.p99 == pytest.approx(3.94, abs=1e-3)
    assert applier.normalization.sign_convention == "L_neg_R_pos"


def test_from_disparity_rejects_non_finite():
    d = np.array([[1.0, float("nan")]], dtype=np.float32)
    with pytest.raises(ValueError):
        DepthNormalizationApplier.from_disparity(d)


def test_from_disparity_rejects_negative():
    d = np.array([[1.0, -0.1]], dtype=np.float32)
    with pytest.raises(ValueError):
        DepthNormalizationApplier.from_disparity(d)


def test_from_disparity_rejects_other_sign_convention():
    d = np.array([[1.0]], dtype=np.float32)
    with pytest.raises(ValueError):
        DepthNormalizationApplier.from_disparity(d, sign_convention="R_neg_L_pos")


def test_apply_l_neg():
    """L direction = sign = -1."""
    d = np.array([[0.5, 1.0, 2.0]], dtype=np.float32)
    applier = DepthNormalizationApplier.from_disparity(d)
    out = applier.apply(d, direction_sign=-1)
    # After p99 ≈ 1.96 (of [0.5, 1.0, 2.0]), d/p99 ≈ [0.255, 0.510, 1.020→1.0]
    np.testing.assert_allclose(out[0, :2], [-0.255, -0.510], atol=1e-2)
    assert out[0, 2] == pytest.approx(-1.0, abs=1e-6)


def test_apply_r_pos():
    d = np.array([[0.5, 1.0, 2.0]], dtype=np.float32)
    applier = DepthNormalizationApplier.from_disparity(d)
    out = applier.apply(d, direction_sign=1)
    np.testing.assert_allclose(out[0, :2], [0.255, 0.510], atol=1e-2)
    assert out[0, 2] == pytest.approx(1.0, abs=1e-6)


def test_apply_clamps_to_unit():
    """Values above p99 are clipped to +1 (or -1 for L)."""
    d = np.array([[10.0, 100.0]], dtype=np.float32)
    applier = DepthNormalizationApplier.from_disparity(d)
    out_pos = applier.apply(d, direction_sign=1)
    assert out_pos.max() <= 1.0
    out_neg = applier.apply(d, direction_sign=-1)
    assert out_neg.min() >= -1.0


def test_apply_invalid_direction_sign():
    d = np.array([[1.0]], dtype=np.float32)
    applier = DepthNormalizationApplier.from_disparity(d)
    with pytest.raises(ValueError):
        applier.apply(d, direction_sign=2)


def test_apply_with_tiny_p99_returns_zeros():
    """When p99 is below epsilon, apply returns zeros."""
    d = np.array([[0.0, 1e-9, 1e-9]], dtype=np.float32)
    applier = DepthNormalizationApplier.from_disparity(d)
    out = applier.apply(d, direction_sign=1)
    np.testing.assert_array_equal(out, np.zeros_like(d))


def test_apply_batched_input():
    d2d = np.array([[0.5, 1.0, 2.0]], dtype=np.float32)  # (1, 3)
    d4d = np.broadcast_to(d2d[None, None, :, :], (2, 1, 1, 3)).astype(np.float32)
    applier = DepthNormalizationApplier.from_disparity(d2d)
    out = applier.apply(d4d, direction_sign=1)
    assert out.shape == (2, 1, 1, 3)
    # p99 of [0.5, 1.0, 2.0] is roughly 1.96, so d/p99 ~= [0.255, 0.510, 1.0]
    np.testing.assert_allclose(out[0, 0, 0, :2], [0.255, 0.510], atol=1e-2)
    assert out[0, 0, 0, 2] == pytest.approx(1.0, abs=1e-6)


def test_apply_depth_normalization_function():
    dn = DepthNormalization(p99=2.0)
    d = np.array([[0.0, 1.0, 2.0, 3.0]], dtype=np.float32)
    out = apply_depth_normalization(d, dn, direction_sign=1)
    np.testing.assert_allclose(out, [[0.0, 0.5, 1.0, 1.0]], atol=1e-6)


def test_apply_depth_normalization_zero_p99():
    dn = DepthNormalization(p99=0.0)
    d = np.array([[0.0, 1.0]], dtype=np.float32)
    out = apply_depth_normalization(d, dn)
    np.testing.assert_array_equal(out, np.zeros_like(d))


def test_repr_is_informative():
    d = np.zeros((4, 4), dtype=np.float32)
    applier = DepthNormalizationApplier.from_disparity(d)
    s = repr(applier)
    assert "p99" in s
    assert "shape=(4, 4)" in s
