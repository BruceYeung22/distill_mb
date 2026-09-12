"""Tests for splits.py."""

from __future__ import annotations

from pathlib import Path

import pytest

from moebius_finetune.contracts import ConditionContractError, Split
from moebius_finetune.data.splits import (
    SPLIT_SOURCES,
    assert_isolated,
    build_default_splits,
    split_sources_from_directory,
)


# ---------------------------------------------------------------------------
# split_sources_from_directory
# ---------------------------------------------------------------------------


def test_split_sources_from_directory_returns_stems(tmp_path):
    d = tmp_path / "imgs"
    d.mkdir()
    (d / "0001.jpg").write_bytes(b"")
    (d / "0002.jpg").write_bytes(b"")
    (d / "ignore.png").write_bytes(b"")
    sources = split_sources_from_directory(d)
    assert sources == {"0001", "0002"}


def test_split_sources_from_directory_missing_returns_empty(tmp_path):
    sources = split_sources_from_directory(tmp_path / "nope")
    assert sources == set()


def test_split_sources_from_directory_custom_suffix(tmp_path):
    d = tmp_path / "imgs"
    d.mkdir()
    (d / "0001.png").write_bytes(b"")
    (d / "0002.jpg").write_bytes(b"")
    sources = split_sources_from_directory(d, suffix=".png")
    assert sources == {"0001"}


# ---------------------------------------------------------------------------
# assert_isolated
# ---------------------------------------------------------------------------


def test_assert_isolated_passes_for_valid_layout():
    splits = {
        Split.TRAIN: {"a", "b", "c"},
        Split.HOLDOUT_MASKS: {"a", "b"},  # subset
        Split.HOLDOUT_IMAGES: {"d", "e"},  # disjoint
    }
    assert_isolated(splits)  # does not raise


def test_assert_isolated_raises_on_overlap():
    splits = {
        Split.TRAIN: {"a", "b", "c"},
        Split.HOLDOUT_MASKS: {"a"},
        Split.HOLDOUT_IMAGES: {"c"},  # overlaps with train
    }
    with pytest.raises(ConditionContractError) as exc:
        assert_isolated(splits)
    assert "holdout_images" in str(exc.value)
    assert "share" in str(exc.value)


def test_assert_isolated_raises_on_holdout_masks_leak():
    splits = {
        Split.TRAIN: {"a", "b"},
        Split.HOLDOUT_MASKS: {"a", "b", "c"},  # c is not in train
        Split.HOLDOUT_IMAGES: set(),
    }
    with pytest.raises(ConditionContractError) as exc:
        assert_isolated(splits)
    assert "holdout_masks" in str(exc.value)


# ---------------------------------------------------------------------------
# build_default_splits
# ---------------------------------------------------------------------------


def _build_hami_layout(root: Path) -> None:
    """Materialise a Hami-style 320 benchmark with combo-suffixed images."""
    for split in ("train", "holdout_masks", "holdout_images"):
        for combo_idx in range(2):
            (root / split / "images").mkdir(parents=True, exist_ok=True)
            (root / split / "images" / f"000000033114__{combo_idx:02d}_d08L.jpg").write_bytes(b"")


def test_build_default_splits_collects_unique_sources(tmp_path):
    _build_hami_layout(tmp_path)
    out = build_default_splits(hami_benchmark_root=tmp_path)
    # All 3 splits have the same combo files in this test fixture
    assert out[Split.TRAIN] == {"000000033114"}
    assert out[Split.HOLDOUT_MASKS] == {"000000033114"}
    assert out[Split.HOLDOUT_IMAGES] == {"000000033114"}


def test_build_default_splits_updates_module_global(tmp_path):
    _build_hami_layout(tmp_path)
    build_default_splits(hami_benchmark_root=tmp_path)
    assert "000000033114" in SPLIT_SOURCES[Split.TRAIN]


def test_build_default_splits_missing_root_returns_empty(tmp_path):
    out = build_default_splits(hami_benchmark_root=tmp_path / "nope")
    assert out == {Split.TRAIN: set(), Split.HOLDOUT_MASKS: set(), Split.HOLDOUT_IMAGES: set()}
