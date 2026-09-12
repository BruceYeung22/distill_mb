"""Tests for the fixed-seed evaluator."""

from __future__ import annotations

import math

import numpy as np
import pytest

from moebius_finetune.contracts import (
    ConditionBatch,
    ConditionContractError,
    SampleManifest,
    Split,
    Direction,
    CoordFrame,
    DepthNormalization,
)
from moebius_finetune.evaluation.evaluator import (
    FixedEvaluator,
    evaluate_manifest,
    run_dummy_predictor,
)


# ---------------------------------------------------------------------------
# run_dummy_predictor
# ---------------------------------------------------------------------------


def _make_batch(H: int = 64, B: int = 1, hole_spec: str = "thin") -> ConditionBatch:
    from moebius_finetune.data.synthetic import make_synthetic_batch
    return make_synthetic_batch(H=H, B=B, hole_spec=hole_spec, seed=0)


def test_dummy_predictor_returns_correct_shape():
    batch = _make_batch()
    out = run_dummy_predictor(batch, seed=0)
    assert out.shape == batch.rgb_hole.shape
    assert out.dtype == np.float32


def test_dummy_predictor_values_in_unit_range():
    batch = _make_batch()
    out = run_dummy_predictor(batch, seed=0)
    assert out.min() >= 0.0
    assert out.max() <= 1.0


def test_dummy_predictor_changes_with_seed():
    batch = _make_batch()
    a = run_dummy_predictor(batch, seed=0)
    b = run_dummy_predictor(batch, seed=1)
    assert not np.allclose(a, b)


# ---------------------------------------------------------------------------
# FixedEvaluator with synthetic fallback
# ---------------------------------------------------------------------------


def test_evaluator_synthetic_returns_required_keys():
    ev = FixedEvaluator(
        predictor=run_dummy_predictor,
        manifest=None,
        seed=0,
        n_seeds=3,
        H=64,
        B=1,
        hole_spec="thin",
    )
    out = ev.run()
    assert "per_case" in out
    assert "aggregate" in out
    assert "stability" in out
    assert isinstance(out["per_case"], list)
    assert len(out["per_case"]) == 1


def test_evaluator_synthetic_per_case_record():
    ev = FixedEvaluator(
        predictor=run_dummy_predictor,
        H=64,
        hole_spec="thin",
    )
    out = ev.run()
    case = out["per_case"][0]
    assert case["case_id"] == "__synthetic__"
    assert "hole_psnr" in case
    assert "boundary_l1" in case
    assert "known_max_error" in case
    assert "hole_ratio" in case
    assert "empty" in case


def test_evaluator_empty_hole_is_marked():
    ev = FixedEvaluator(
        predictor=run_dummy_predictor,
        H=64,
        hole_spec="empty",
    )
    out = ev.run()
    case = out["per_case"][0]
    assert case["empty"] is True
    assert case["hole_psnr"] == 0.0


def test_evaluator_stability_for_synthetic():
    ev = FixedEvaluator(
        predictor=run_dummy_predictor,
        H=64,
        n_seeds=4,
    )
    out = ev.run()
    stab = out["stability"]
    assert stab["n_seeds"] == 4
    assert len(stab["psnr_per_seed"]) == 4
    # Mean and std must be finite
    assert math.isfinite(stab["psnr_mean"])
    assert math.isfinite(stab["psnr_std"])
    assert stab["psnr_std"] >= 0.0


def test_evaluator_rejects_non_finite_predictor():
    def bad_predictor(batch, *, seed):
        out = np.zeros_like(batch.rgb_hole)
        out[0, 0, 0, 0] = float("nan")
        return out

    ev = FixedEvaluator(predictor=bad_predictor, H=64)
    with pytest.raises(ConditionContractError):
        ev.run()


def test_evaluator_rejects_wrong_shape_predictor():
    def bad_predictor(batch, *, seed):
        return np.zeros((batch.rgb_hole.shape[0], 1, 8, 8), dtype=np.float32)

    ev = FixedEvaluator(predictor=bad_predictor, H=64)
    with pytest.raises(ConditionContractError):
        ev.run()


def test_evaluator_rejects_wrong_dtype_predictor():
    def bad_predictor(batch, *, seed):
        return np.zeros(batch.rgb_hole.shape, dtype=np.float64)

    ev = FixedEvaluator(predictor=bad_predictor, H=64)
    with pytest.raises(ConditionContractError):
        ev.run()


# ---------------------------------------------------------------------------
# FixedEvaluator with manifest
# ---------------------------------------------------------------------------


def _make_manifest_sample(case_id: str = "0001__00_d08L") -> SampleManifest:
    return SampleManifest(
        case_id=case_id,
        source_id="0001",
        split=Split.TRAIN,
        coord_frame=CoordFrame.SOURCE,
        rgb_path="rgb.png",
        depth_path="depth.npy",
        mask_path="mask.png",
        target_path="target.png",
        direction=Direction.L2R,
        dmax_px=8,
        depth_normalization=DepthNormalization(p99=0.5),
    )


def test_evaluator_runs_with_manifest():
    manifest = [_make_manifest_sample("a"), _make_manifest_sample("b")]
    ev = FixedEvaluator(
        predictor=run_dummy_predictor, manifest=manifest, H=64, n_seeds=3,
    )
    out = ev.run()
    assert len(out["per_case"]) == 2
    assert {c["case_id"] for c in out["per_case"]} == {"a", "b"}
    # Each case has its own stability record
    assert {c["case_id"] for c in out["stability"]} == {"a", "b"}


def test_evaluate_manifest_convenience():
    manifest = [_make_manifest_sample()]
    out = evaluate_manifest(
        manifest, run_dummy_predictor, H=64, n_seeds=2,
    )
    assert "per_case" in out
    assert len(out["per_case"]) == 1


def test_evaluator_aggregate_includes_per_split_groups():
    manifest = [
        _make_manifest_sample("a"),
        SampleManifest(
            case_id="b",
            source_id="0002",
            split=Split.HOLDOUT_IMAGES,
            coord_frame=CoordFrame.SOURCE,
            rgb_path="rgb_b.png",
            depth_path="depth_b.npy",
            mask_path="mask_b.png",
            target_path="target_b.png",
            direction=Direction.R2L,
            dmax_px=16,
            depth_normalization=DepthNormalization(p99=0.3),
        ),
    ]
    out = evaluate_manifest(manifest, run_dummy_predictor, H=64, n_seeds=2)
    agg = out["aggregate"]
    assert "by_split" in agg
    assert "train" in agg["by_split"]
    assert "holdout_images" in agg["by_split"]
