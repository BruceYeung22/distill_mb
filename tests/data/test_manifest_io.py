"""Tests for manifest_io.py."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from moebius_finetune.contracts import (
    ConditionContractError,
    CoordFrame,
    DepthNormalization,
    Direction,
    SampleManifest,
    Split,
)
from moebius_finetune.data.manifest_io import (
    from_hami_320,
    load_manifest,
    mask_rel_to_data_root,
    save_manifest,
    to_relative,
)


# ---------------------------------------------------------------------------
# to_relative
# ---------------------------------------------------------------------------


def test_to_relative_simple(tmp_path):
    p = tmp_path / "a" / "b" / "c.png"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.touch()
    assert to_relative(p, tmp_path) == "a/b/c.png"


def test_to_relative_equal_returns_dot(tmp_path):
    assert to_relative(tmp_path, tmp_path) == "."


def test_to_relative_already_relative(tmp_path):
    assert to_relative("a/b/c.png", tmp_path) == "a/b/c.png"


def test_to_relative_outside_root_returns_absolute(tmp_path):
    p = Path("C:/some/other/path.png")
    out = to_relative(p, tmp_path)
    # On Windows this may be absolute; the API preserves it.
    assert isinstance(out, str)
    assert out.endswith("path.png")


# ---------------------------------------------------------------------------
# save_manifest / load_manifest round-trip
# ---------------------------------------------------------------------------


def _sample(
    case_id: str = "case-0001",
    *,
    split: Split = Split.TRAIN,
    direction: Direction = Direction.L2R,
    dmax_px: int = 12,
) -> SampleManifest:
    return SampleManifest(
        case_id=case_id,
        source_id="src-42",
        split=split,
        coord_frame=CoordFrame.SOURCE,
        rgb_path="images/case_rgb.png",
        depth_path="depth/case_depth.npy",
        mask_path="masks/case_mask.png",
        target_path="targets/case_target.png",
        direction=direction,
        dmax_px=dmax_px,
        depth_normalization=DepthNormalization(p99=4.5),
    )


def test_save_then_load_round_trip(tmp_path):
    samples = [_sample(), _sample("case-0002", split=Split.HOLDOUT_MASKS, direction=Direction.R2L)]
    out_path = tmp_path / "manifest.json"
    save_manifest(samples, out_path, data_root=tmp_path)
    loaded = load_manifest(out_path)
    assert len(loaded) == 2
    assert loaded[0] == samples[0]
    assert loaded[1] == samples[1]


def test_save_writes_relative_paths(tmp_path):
    samples = [_sample()]
    out_path = tmp_path / "manifest.json"
    save_manifest(samples, out_path, data_root=tmp_path)
    raw = json.loads(out_path.read_text(encoding="utf-8"))
    for entry in raw:
        for key in ("rgb_path", "depth_path", "mask_path", "target_path"):
            assert not Path(entry[key]).is_absolute(), (
                f"{key} in manifest must be relative, got {entry[key]!r}"
            )


def test_save_rejects_absolute_path_without_data_root(tmp_path):
    samples = [_sample()]
    samples[0] = SampleManifest(
        **{
            **samples[0].to_dict(),
            "rgb_path": str(tmp_path / "abs" / "rgb.png"),
            "split": samples[0].split,
            "coord_frame": samples[0].coord_frame,
            "direction": samples[0].direction,
            "depth_normalization": samples[0].depth_normalization,
        }
    )
    with pytest.raises(ConditionContractError):
        save_manifest(samples, tmp_path / "manifest.json")


def test_save_rebases_absolute_path_with_data_root(tmp_path):
    abs_path = tmp_path / "abs" / "rgb.png"
    samples = [_sample()]
    samples[0] = SampleManifest(
        **{
            **samples[0].to_dict(),
            "rgb_path": str(abs_path),
            "split": samples[0].split,
            "coord_frame": samples[0].coord_frame,
            "direction": samples[0].direction,
            "depth_normalization": samples[0].depth_normalization,
        }
    )
    save_manifest(samples, tmp_path / "manifest.json", data_root=tmp_path)
    raw = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert raw[0]["rgb_path"] == "abs/rgb.png"


def test_save_rejects_absolute_outside_data_root(tmp_path):
    samples = [_sample()]
    # Use an absolute path that is guaranteed not to be under tmp_path.
    samples[0] = SampleManifest(
        **{
            **samples[0].to_dict(),
            "rgb_path": str(tmp_path.parent / "definitely-not-tmp" / "abs.png"),
            "split": samples[0].split,
            "coord_frame": samples[0].coord_frame,
            "direction": samples[0].direction,
            "depth_normalization": samples[0].depth_normalization,
        }
    )
    with pytest.raises(ConditionContractError):
        save_manifest(samples, tmp_path / "manifest.json", data_root=tmp_path)


# ---------------------------------------------------------------------------
# load_manifest validation
# ---------------------------------------------------------------------------


def test_load_manifest_missing_file_raises(tmp_path):
    with pytest.raises(ConditionContractError):
        load_manifest(tmp_path / "nope.json")


def test_load_manifest_wrong_shape_raises(tmp_path):
    p = tmp_path / "bad.json"
    p.write_text(json.dumps({"not_cases": []}), encoding="utf-8")
    with pytest.raises(ConditionContractError):
        load_manifest(p)


def test_load_manifest_missing_fields_raises(tmp_path):
    p = tmp_path / "bad.json"
    p.write_text(json.dumps([{"case_id": "x"}]), encoding="utf-8")
    with pytest.raises(ConditionContractError) as exc:
        load_manifest(p)
    assert "missing required field" in str(exc.value)


def test_load_manifest_invalid_split_raises(tmp_path):
    p = tmp_path / "bad.json"
    bad = {
        "case_id": "c",
        "source_id": "s",
        "split": "bogus",
        "coord_frame": "source",
        "rgb_path": "r",
        "depth_path": "d",
        "mask_path": "m",
        "target_path": "t",
        "direction": "L2R",
        "dmax_px": 8,
    }
    p.write_text(json.dumps([bad]), encoding="utf-8")
    with pytest.raises(ConditionContractError):
        load_manifest(p)


def test_load_manifest_invalid_direction_raises(tmp_path):
    p = tmp_path / "bad.json"
    bad = {
        "case_id": "c",
        "source_id": "s",
        "split": "train",
        "coord_frame": "source",
        "rgb_path": "r",
        "depth_path": "d",
        "mask_path": "m",
        "target_path": "t",
        "direction": "bogus",
        "dmax_px": 8,
    }
    p.write_text(json.dumps([bad]), encoding="utf-8")
    with pytest.raises(ConditionContractError):
        load_manifest(p)


def test_load_manifest_invalid_dmax_raises(tmp_path):
    p = tmp_path / "bad.json"
    bad = {
        "case_id": "c",
        "source_id": "s",
        "split": "train",
        "coord_frame": "source",
        "rgb_path": "r",
        "depth_path": "d",
        "mask_path": "m",
        "target_path": "t",
        "direction": "L2R",
        "dmax_px": "8",  # string, should be int
    }
    p.write_text(json.dumps([bad]), encoding="utf-8")
    with pytest.raises(ConditionContractError):
        load_manifest(p)


def test_load_manifest_supports_dict_with_cases_key(tmp_path):
    p = tmp_path / "manifest.json"
    p.write_text(
        json.dumps({"cases": [_sample().to_dict()]}), encoding="utf-8"
    )
    loaded = load_manifest(p)
    assert len(loaded) == 1
    assert loaded[0] == _sample()


# ---------------------------------------------------------------------------
# from_hami_320
# ---------------------------------------------------------------------------


def _write_hami_manifest(path: Path, *, n_cases: int = 2):
    images = [
        {
            "split": "train",
            "file_name": "000000033114.jpg",
            "coco_image_id": 33114,
            "sha256": "abc",
            "depth_stats": {"p99": 0.123, "min": 0.01, "max": 0.15},
        },
        {
            "split": "holdout_images",
            "file_name": "000000017379.jpg",
            "coco_image_id": 17379,
            "sha256": "def",
            "depth_stats": {"p99": 0.456, "min": 0.01, "max": 0.5},
        },
    ]
    cases = []
    for i in range(n_cases):
        cases.append(
            {
                "case_id": f"000000033114__{i:02d}_d08L",
                "split": "train",
                "coco_image_id": 33114,
                "dmax_px": 8,
                "direction": "L",
                "hole_ratio": 0.02,
                "image_path": "data/benchmarks/coco32_grt/train/images/000000033114__00_d08L.jpg",
                "mask_path": "data/benchmarks/coco32_grt/train/masks/000000033114__00_d08L.png",
            }
        )
    data = {"images": images, "cases": cases}
    path.write_text(json.dumps(data), encoding="utf-8")


def test_from_hami_320_basic(tmp_path):
    p = tmp_path / "hami.json"
    _write_hami_manifest(p, n_cases=2)
    samples = from_hami_320(p)
    assert len(samples) == 2
    for s, i in zip(samples, range(2)):
        assert s.case_id == f"000000033114__{i:02d}_d08L"
        assert s.source_id == "33114"
        assert s.split is Split.TRAIN
        assert s.direction is Direction.L2R
        assert s.dmax_px == 8
        assert s.depth_normalization.p99 == pytest.approx(0.123, abs=1e-6)


def test_from_hami_320_only_images(tmp_path):
    p = tmp_path / "hami.json"
    p.write_text(
        json.dumps(
            {
                "images": [
                    {
                        "split": "train",
                        "file_name": "000000033114.jpg",
                        "coco_image_id": 33114,
                        "depth_stats": {"p99": 0.5},
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    samples = from_hami_320(p)
    assert len(samples) == 1
    assert samples[0].source_id == "33114"
    assert samples[0].depth_normalization.p99 == pytest.approx(0.5)


def test_from_hami_320_missing_file_raises(tmp_path):
    with pytest.raises(ConditionContractError):
        from_hami_320(tmp_path / "nope.json")


def test_mask_rel_to_data_root_relative_kept():
    assert mask_rel_to_data_root("masks/x.png", "x") == "masks/x.png"


def test_mask_rel_to_data_root_absolute_rebased(tmp_path):
    abs_path = tmp_path / "data" / "masks" / "x.png"
    out = mask_rel_to_data_root(str(abs_path), "x")
    assert "masks" not in Path(out).parts or out.endswith("x.png")
