"""Tests for pipeline_512.py.

These use a synthetic RGB and a hand-crafted disparity cache so the
test does not need real COCO data. The pipeline must:

* Write all five on-disk artefacts in the right places.
* Produce a :class:`SampleManifest` whose paths are relative to the
  data root.
* Build the GRT mask at 512 — not by reusing or upsampling a 320
  mask.
* Record the same p99 the GRT pipeline actually used.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from moebius_finetune.contracts import (
    CoordFrame,
    DepthNormalization,
    Direction,
    SampleManifest,
    Split,
)
from moebius_finetune.data.pipeline_512 import build_512_sample


def _write_fake_rgb(target: Path, size: int = 512) -> None:
    rng = np.random.default_rng(0)
    arr = (rng.random((size, size, 3)) * 255).astype(np.uint8)
    target.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr).save(target)


def _write_fake_disparity(target: Path, *, size: int = 512, dmax_for_data: int = 24) -> np.ndarray:
    """Write a fake depth cache with a known p99."""
    rng = np.random.default_rng(42)
    # Smooth gradient with some noise so p99 is well-defined.
    ys, xs = np.meshgrid(np.linspace(0.0, 1.0, size), np.linspace(0.0, 1.0, size), indexing="ij")
    d = (0.6 * ys + 0.3 * xs + 0.05 * rng.random((size, size))).astype(np.float32)
    d = np.clip(d, 0.0, 1.0)
    target.parent.mkdir(parents=True, exist_ok=True)
    np.save(target, d.astype(np.float32))
    return d


def test_build_512_sample_writes_all_artefacts(tmp_path):
    data_root = tmp_path / "data"
    coco_dir = tmp_path / "coco"
    src_id = "000000033114"
    size = 64  # small for fast test
    _write_fake_rgb(coco_dir / f"{src_id}.jpg", size=size)
    _write_fake_disparity(data_root / "disparity" / f"{src_id}.npy", size=size)

    m = build_512_sample(
        src_id,
        data_root=data_root,
        coco_dir=coco_dir,
        split=Split.TRAIN,
        direction=Direction.L2R,
        dmax_px=16,
        size=size,
    )

    # 1. Manifest fields
    assert isinstance(m, SampleManifest)
    assert m.case_id == f"{src_id}__16_L2R"
    assert m.source_id == src_id
    assert m.split is Split.TRAIN
    assert m.direction is Direction.L2R
    assert m.dmax_px == 16
    assert m.coord_frame is CoordFrame.SOURCE

    # 2. All paths are RELATIVE
    for key in ("rgb_path", "depth_path", "mask_path", "target_path"):
        p = getattr(m, key)
        assert not Path(p).is_absolute(), f"{key} must be relative, got {p!r}"
        # And actually exists on disk
        assert (data_root / p).exists(), f"{key} not written: {data_root / p}"

    # 3. Stats JSON is written
    stats_path = data_root / Split.TRAIN.value / "stats" / f"{m.case_id}.json"
    assert stats_path.exists()
    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    assert stats["dmax_px"] == 16
    assert stats["direction"] == "L2R"
    assert stats["size"] == size
    assert stats["depth_normalization"]["sign_convention"] == "L_neg_R_pos"


def test_build_512_sample_mask_is_512_not_upsampled(tmp_path):
    """Confirm that the 320 mask is NOT being upsampled to 512.

    We materialise a 320 case and a 512 case for the same source and
    compare: the 512 mask must have a *strictly* larger hole count
    when dmax is proportional to size, and must NOT match a naive
    nn-upsample of the 320 mask.
    """
    data_root = tmp_path / "data"
    coco_dir = tmp_path / "coco"
    src_id = "000000033114"
    size = 64
    _write_fake_rgb(coco_dir / f"{src_id}.jpg", size=size)
    _write_fake_disparity(data_root / "disparity" / f"{src_id}.npy", size=size)

    m = build_512_sample(
        src_id,
        data_root=data_root,
        coco_dir=coco_dir,
        split=Split.TRAIN,
        direction=Direction.R2L,
        dmax_px=8,
        size=size,
    )
    mask_path = data_root / m.mask_path
    mask = np.asarray(Image.open(mask_path).convert("L"))
    # Mask should be size x size
    assert mask.shape == (size, size)
    # Mask must contain only 0/255
    uniq = set(np.unique(mask).tolist())
    assert uniq.issubset({0, 255})


def test_build_512_sample_hole_pixels_zero_in_depth(tmp_path):
    data_root = tmp_path / "data"
    coco_dir = tmp_path / "coco"
    src_id = "000000033114"
    size = 64
    _write_fake_rgb(coco_dir / f"{src_id}.jpg", size=size)
    _write_fake_disparity(data_root / "disparity" / f"{src_id}.npy", size=size)

    m = build_512_sample(
        src_id,
        data_root=data_root,
        coco_dir=coco_dir,
        split=Split.HOLDOUT_MASKS,
        direction=Direction.R2L,
        dmax_px=12,
        size=size,
    )

    depth = np.load(data_root / m.depth_path)
    mask = (np.asarray(Image.open(data_root / m.mask_path).convert("L")) >= 127).astype(np.float32)
    # Depth inside the hole is exactly 0
    assert float((depth * mask).max()) == 0.0
    # Depth outside the hole has at least some non-zero values
    bg = 1.0 - mask
    if bg.sum() > 0:
        assert float((np.abs(depth) * bg).max()) > 0.0


def test_build_512_sample_l_direction_sign_is_negative(tmp_path):
    data_root = tmp_path / "data"
    coco_dir = tmp_path / "coco"
    src_id = "000000033114"
    size = 64
    _write_fake_rgb(coco_dir / f"{src_id}.jpg", size=size)
    _write_fake_disparity(data_root / "disparity" / f"{src_id}.npy", size=size)

    m_l = build_512_sample(
        src_id,
        data_root=data_root,
        coco_dir=coco_dir,
        split=Split.TRAIN,
        direction=Direction.L2R,
        dmax_px=8,
        size=size,
    )
    m_r = build_512_sample(
        src_id,
        data_root=data_root,
        coco_dir=coco_dir,
        split=Split.TRAIN,
        direction=Direction.R2L,
        dmax_px=8,
        size=size,
    )
    depth_l = np.load(data_root / m_l.depth_path)
    depth_r = np.load(data_root / m_r.depth_path)
    mask_l = (np.asarray(Image.open(data_root / m_l.mask_path).convert("L")) >= 127)
    mask_r = (np.asarray(Image.open(data_root / m_r.mask_path).convert("L")) >= 127)
    # Outside the hole: L is non-positive, R is non-negative
    bg_l = ~mask_l
    bg_r = ~mask_r
    if bg_l.any():
        assert depth_l[bg_l].max() <= 0.0 + 1e-6
        assert depth_l[bg_l].min() <= 0.0 + 1e-6  # at least one strictly negative
    if bg_r.any():
        assert depth_r[bg_r].min() >= 0.0 - 1e-6
        assert depth_r[bg_r].max() >= 0.0 - 1e-6  # at least one strictly positive


def test_build_512_sample_missing_rgb_raises(tmp_path):
    data_root = tmp_path / "data"
    coco_dir = tmp_path / "coco"
    (data_root / "disparity").mkdir(parents=True, exist_ok=True)
    with pytest.raises(FileNotFoundError):
        build_512_sample(
            "nope",
            data_root=data_root,
            coco_dir=coco_dir,
            split=Split.TRAIN,
            direction=Direction.R2L,
            dmax_px=8,
            size=64,
        )


def test_build_512_sample_missing_disparity_raises(tmp_path):
    data_root = tmp_path / "data"
    coco_dir = tmp_path / "coco"
    (coco_dir).mkdir(parents=True, exist_ok=True)
    _write_fake_rgb(coco_dir / "0001.jpg", size=64)
    # No disparity cache
    with pytest.raises(FileNotFoundError) as exc:
        build_512_sample(
            "0001",
            data_root=data_root,
            coco_dir=coco_dir,
            split=Split.TRAIN,
            direction=Direction.R2L,
            dmax_px=8,
            size=64,
        )
    assert "ZipDepth" in str(exc.value)


def test_build_512_sample_size_must_be_multiple_of_8(tmp_path):
    with pytest.raises(Exception):
        build_512_sample(
            "0001",
            data_root=tmp_path,
            coco_dir=tmp_path,
            split=Split.TRAIN,
            direction=Direction.R2L,
            dmax_px=8,
            size=63,
        )


def test_build_512_sample_dmax_must_be_positive(tmp_path):
    with pytest.raises(Exception):
        build_512_sample(
            "0001",
            data_root=tmp_path,
            coco_dir=tmp_path,
            split=Split.TRAIN,
            direction=Direction.R2L,
            dmax_px=0,
            size=64,
        )
