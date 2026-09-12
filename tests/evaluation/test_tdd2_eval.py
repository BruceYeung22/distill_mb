"""Tests for TDD2 §5 evaluation: hole_l1 / global_l1 / masked lpips / pick_cases."""

from __future__ import annotations

import numpy as np
import pytest

from moebius_finetune.evaluation.lpips_masked import masked_lpips
from moebius_finetune.evaluation.metrics import global_l1, hole_l1
from moebius_finetune.evaluation.teacher_eval import aggregate_rows, pick_cases


B, C, H, W = 2, 3, 16, 16


def _pair(seed=0):
    rng = np.random.default_rng(seed)
    return (
        rng.random((B, C, H, W), dtype=np.float32),
        rng.random((B, C, H, W), dtype=np.float32),
        (rng.random((B, 1, H, W)) > 0.5).astype(np.float32),
    )


def test_hole_l1_matches_torch_semantics():
    import torch

    from moebius_finetune.training.student.losses import hole_l1 as torch_hole_l1

    pred, tgt, mask = _pair(0)
    a = hole_l1(pred, tgt, mask)
    b = float(torch_hole_l1(torch.from_numpy(pred), torch.from_numpy(tgt), torch.from_numpy(mask)))
    assert abs(a - b) < 1e-6


def test_hole_l1_empty_mask_is_zero():
    pred, tgt, _ = _pair(1)
    empty = np.zeros((B, 1, H, W), np.float32)
    assert hole_l1(pred, tgt, empty) == 0.0


def test_global_l1_exact():
    pred = np.zeros((1, 3, 4, 4), np.float32)
    tgt = np.ones((1, 3, 4, 4), np.float32)
    assert global_l1(pred, tgt) == 1.0


def test_masked_lpips_known_region_is_neutralized():
    """Compositing must make the known region identical in both images."""

    captured = {}

    def stub(img0, img1):
        captured["img0"] = img0.detach().cpu().numpy()
        captured["img1"] = img1.detach().cpu().numpy()
        import torch

        return torch.ones(img0.shape[0], 1, 1, 1, device=img0.device)

    pred, tgt, mask = _pair(2)
    value = masked_lpips(pred, tgt, mask, model=stub, device="cpu")
    # The stub returns 1.0 for every case → batch mean is exactly 1.
    assert value == 1.0
    # Known region: pred was composited to target → identical in both.
    known = np.broadcast_to((1.0 - mask) > 0, captured["img0"].shape)
    assert np.allclose(captured["img0"][known], captured["img1"][known])
    # Hole region: pred kept its own values.
    hole = np.broadcast_to(mask > 0, captured["img0"].shape)
    assert np.allclose(
        captured["img0"][hole], (pred * 2.0 - 1.0)[hole]
    )


def test_pick_cases_deterministic_subset():
    from moebius_finetune.data.grt_dataset import CaseSpec

    cases = [
        CaseSpec(f"id{i:03d}", 16, "L2R", "holdout_images") for i in range(100)
    ]
    a = pick_cases(cases, 10, seed=0)
    b = pick_cases(cases, 10, seed=0)
    c = pick_cases(cases, 10, seed=1)
    assert [x.case_id for x in a] == [x.case_id for x in b]
    assert [x.case_id for x in a] != [x.case_id for x in c]
    assert len(a) == 10


def test_aggregate_rows_composite():
    rows = [
        {
            "hole_l1": 0.1, "hole_lpips": 0.2, "global_l1": 0.05,
            "hole_ratio": 0.1, "known_max_error": 0.0,
        },
        {
            "hole_l1": 0.2, "hole_lpips": 0.4, "global_l1": 0.15,
            "hole_ratio": 0.3, "known_max_error": 0.0,
        },
    ]
    s = aggregate_rows(rows)
    assert s["n"] == 2
    assert abs(s["hole_l1"] - 0.15) < 1e-9
    assert s["known_max_error"] == 0.0
    # S = 1*0.3 + 3*0.15 + 1*0.10
    assert abs(s["S"] - (0.3 + 0.45 + 0.10)) < 1e-9
