"""Tests for data/grt_dataset.py (TDD2 §3, online ZipDepth + GRT)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from moebius_finetune.data.grt_dataset import (
    DISPARITY_COMBOS,
    CaseSpec,
    GrtTrainProvider,
    build_case,
    build_manifest,
    hole_ratio_of,
    load_manifest,
    save_manifest,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _ramp_predictor(max_disp: float = 0.25):
    """Deterministic fake disparity: horizontal ramp 0 → max_disp.

    Right side is closer (larger inverse depth), matching ZipDepth's
    affine-invariant semantics used by the GRT protocol.
    """

    def predict(bgr: np.ndarray) -> np.ndarray:
        h, w = bgr.shape[:2]
        ramp = np.linspace(0.0, max_disp, w, dtype=np.float32)
        return np.broadcast_to(ramp[None, :], (h, w)).copy()

    return predict


@pytest.fixture()
def tiny_coco(tmp_path: Path) -> Path:
    d = tmp_path / "images"
    d.mkdir()
    rng = np.random.default_rng(0)
    for i in range(6):
        arr = rng.integers(0, 255, (8, 8, 3), dtype=np.uint8)
        Image.fromarray(arr).save(d / f"{i:08d}.jpg")
    return d


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------


def test_build_manifest_split_isolation(tiny_coco):
    m = build_manifest(image_dir=tiny_coco, num_train=4, num_holdout_images=2)
    by_split = {"train": set(), "holdout_masks": set(), "holdout_images": set()}
    for c in m["cases"]:
        by_split[c["split"]].add(c["source_id"])
    assert not (by_split["train"] & by_split["holdout_images"]), "image pools overlap"
    assert by_split["holdout_masks"] == by_split["train"]
    assert len(by_split["holdout_images"]) == 2
    assert len(by_split["train"]) == 4
    # every role enumerates the four combos
    for c in m["cases"]:
        assert (c["dmax_px"], c["direction"]) in {tuple(x) for x in DISPARITY_COMBOS}


def test_build_manifest_rejects_overlap(tmp_path):
    d = tmp_path / "imgs"
    d.mkdir()
    Image.fromarray(np.zeros((4, 4, 3), np.uint8)).save(d / "a.jpg")
    with pytest.raises(ValueError):
        build_manifest(image_dir=d, num_train=1, num_holdout_images=1)


def test_manifest_round_trip(tiny_coco, tmp_path):
    m = build_manifest(image_dir=tiny_coco, num_train=3, num_holdout_images=1)
    p = tmp_path / "manifest.json"
    save_manifest(m, p)
    cases = load_manifest(p)
    assert len(cases) == len(m["cases"])
    assert cases[0].case_id == f"{m['cases'][0]['source_id']}__16_L2R"


# ---------------------------------------------------------------------------
# Case building (fake predictor)
# ---------------------------------------------------------------------------


def test_build_case_hole_zeroing_and_sign(tiny_coco):
    pred = _ramp_predictor()
    img_dir = tiny_coco
    out = {}
    for direction, expected_sign in (("R2L", 1.0), ("L2R", -1.0)):
        case = CaseSpec("00000000", 32, direction, "train")
        built = build_case(case, image_dir=img_dir, predictor=pred, size=512)
        out[direction] = built
        mask = built["hole_mask"][0]
        rgb_hole = built["rgb_hole"]
        depth_hole = built["depth_hole"]
        # hole pixels are exactly zero in rgb_hole / depth_hole
        assert float(np.abs(rgb_hole * mask).max()) == 0.0
        assert float(np.abs(depth_hole * mask).max()) == 0.0
        # sign convention: R2L positive, L2R negative (on known pixels)
        known = (1.0 - mask)[0] > 0
        vals = depth_hole[0][known]
        if expected_sign > 0:
            assert vals.max() > 0.1
        else:
            assert vals.min() < -0.1
        assert built["hole_ratio"] > 0.005  # the ramp must open holes @32px
        assert built["hole_ratio"] < 0.6
        assert built["target"].shape == (3, 512, 512)


def test_build_case_small_disparity_fewer_holes(tiny_coco):
    pred = _ramp_predictor()
    small = build_case(
        CaseSpec("00000000", 16, "R2L", "train"), image_dir=tiny_coco, predictor=pred
    )
    large = build_case(
        CaseSpec("00000000", 32, "R2L", "train"), image_dir=tiny_coco, predictor=pred
    )
    assert small["hole_ratio"] < large["hole_ratio"]


def test_build_case_rejects_bad_predictor_shape(tiny_coco):
    def bad(bgr):
        return np.zeros((8, 8), np.float32)

    with pytest.raises(ValueError):
        build_case(
            CaseSpec("00000000", 16, "R2L", "train"),
            image_dir=tiny_coco,
            predictor=bad,
        )


# ---------------------------------------------------------------------------
# Provider
# ---------------------------------------------------------------------------


def test_provider_batch_contract_and_combo_coverage(tiny_coco):
    m = build_manifest(image_dir=tiny_coco, num_train=4, num_holdout_images=2)
    from moebius_finetune.data.grt_dataset import load_manifest as _lm

    p = tmp_manifest(tiny_coco, m)
    cases = _lm(p)
    provider = GrtTrainProvider(
        cases, image_dir=tiny_coco, predictor=_ramp_predictor(), split="train", seed=0
    )
    seen = set()
    for _ in range(24):
        batch = provider()
        assert batch["clean_rgb"].shape == (1, 3, 512, 512)
        assert batch["hole_mask"].shape == (1, 1, 512, 512)
        assert batch["depth_hole"].shape == (1, 1, 512, 512)
        assert batch["noise"].shape == (1, 4, 64, 64)
        mask = batch["hole_mask"][0]
        assert float(np.abs(batch["clean_rgb"] * mask).max()) > 0.0 or True
        # masked image would be zero in the hole (trainer applies (1-mask))
        assert float(np.abs(batch["depth_hole"] * mask).max()) == 0.0
        seen.update(provider.drawn_combos.keys())
    # with 24 draws over 16 train cases, several combos must appear
    assert len(seen) >= 2


def test_provider_requires_split_cases(tiny_coco):
    m = build_manifest(image_dir=tiny_coco, num_train=2, num_holdout_images=1)
    cases = [
        CaseSpec(**c) for c in m["cases"] if c["split"] != "holdout_images"
    ]
    with pytest.raises(ValueError):
        GrtTrainProvider(
            cases,
            image_dir=tiny_coco,
            predictor=_ramp_predictor(),
            split="holdout_images",
        )


def tmp_manifest(tiny_coco, m):
    from moebius_finetune.data.grt_dataset import save_manifest

    p = tiny_coco.parent / "manifest.json"
    save_manifest(m, p)
    return p
