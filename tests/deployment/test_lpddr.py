"""Tests for the LPDDR4X bandwidth model (TDD §8.3)."""

from __future__ import annotations

import pytest

from moebius_finetune.deployment.lpddr import BandwidthModel, latency_lower_bound




def test_300mb_at_10gbps_is_30ms():
    """300 MB (decimal) at 10 GB/s → 30 ms (TDD §8.3 baseline)."""
    # 300 MB = 300 * 1e6 bytes (decimal, per TDD §8.1).
    v = latency_lower_bound(300 * 1_000_000, effective_gbps=10.0)
    assert v == pytest.approx(30.0, abs=1e-6)


def test_300mb_at_15gbps_is_20ms():
    v = latency_lower_bound(300 * 1_000_000, effective_gbps=15.0)
    assert v == pytest.approx(20.0, abs=1e-6)


def test_300mb_at_20gbps_is_15ms():
    v = latency_lower_bound(300 * 1_000_000, effective_gbps=20.0)
    assert v == pytest.approx(15.0, abs=1e-6)


def test_zero_bytes_is_zero_ms():
    assert latency_lower_bound(0, effective_gbps=10.0) == 0.0


def test_invalid_bandwidth_raises():
    with pytest.raises(ValueError):
        latency_lower_bound(1024, effective_gbps=0.0)
    with pytest.raises(ValueError):
        latency_lower_bound(1024, effective_gbps=-1.0)


def test_negative_bytes_raises():
    with pytest.raises(ValueError):
        latency_lower_bound(-1, effective_gbps=10.0)


def test_bandwidth_model_default_peak_is_34_1():
    bw = BandwidthModel()
    assert bw.peak_gbps == pytest.approx(34.1, abs=1e-6)


def test_bandwidth_model_default_scenarios():
    bw = BandwidthModel()
    out = bw.latency_scenarios(300 * 1024 * 1024)
    assert set(out.keys()) == {"10gbps", "15gbps", "20gbps"}


def test_bandwidth_model_three_scenarios_match_spec():
    """300 MB (decimal) → 30 / 20 / 15 ms at 10 / 15 / 20 GB/s."""
    bw = BandwidthModel()
    out = bw.latency_scenarios(300 * 1_000_000)
    assert out["10gbps"]["latency_ms"] == pytest.approx(30.0, abs=1e-6)
    assert out["15gbps"]["latency_ms"] == pytest.approx(20.0, abs=1e-6)
    assert out["20gbps"]["latency_ms"] == pytest.approx(15.0, abs=1e-6)
