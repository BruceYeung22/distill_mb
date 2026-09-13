"""Tests for the comparison figure renderer."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from moebius_finetune.evaluation.plots import _ensure_image, plot_comparison_grid


def _full_item(h: int = 32, w: int = 32) -> dict:
    return {
        "input": (np.random.default_rng(0).random((3, h, w)) * 255).astype(np.uint8),
        "mask": np.zeros((1, h, w), dtype=np.float32),
        "depth": np.random.default_rng(1).random((h, w)).astype(np.float32),
        "target": (np.random.default_rng(2).random((3, h, w)) * 255).astype(np.uint8),
        "baseline": (np.random.default_rng(3).random((3, h, w)) * 255).astype(np.uint8),
        "teacher": (np.random.default_rng(4).random((3, h, w)) * 255).astype(np.uint8),
        "student": (np.random.default_rng(5).random((3, h, w)) * 255).astype(np.uint8),
        "error": {
            "pred": np.random.default_rng(6).random((3, h, w)).astype(np.float32),
            "target": np.random.default_rng(7).random((3, h, w)).astype(np.float32),
        },
    }


def test_plot_comparison_grid_creates_file(tmp_path: Path):
    out = tmp_path / "grid.png"
    res = plot_comparison_grid([_full_item()], out)
    assert res == out
    assert out.exists()
    assert out.stat().st_size > 0


def test_plot_comparison_grid_multiple_rows(tmp_path: Path):
    out = tmp_path / "grid.png"
    items = [_full_item(), _full_item(h=24, w=24)]
    res = plot_comparison_grid(items, out)
    assert res is not None
    assert res.exists()


def test_plot_comparison_grid_handles_missing_columns(tmp_path: Path):
    """When a row is missing some columns, the renderer should still
    succeed and write a blank tile for the missing ones.
    """
    out = tmp_path / "grid.png"
    item = _full_item()
    item.pop("teacher", None)
    item.pop("student", None)
    res = plot_comparison_grid([item], out)
    assert res is not None
    assert out.exists()


def test_plot_comparison_grid_empty_items(tmp_path: Path):
    out = tmp_path / "grid.png"
    res = plot_comparison_grid([], out)
    assert res is not None
    assert out.exists()


def test_plot_comparison_grid_error_dict(tmp_path: Path):
    out = tmp_path / "grid.png"
    item = _full_item()
    item["error"] = {
        "pred": np.random.default_rng(0).random((3, 32, 32)).astype(np.float32),
        "target": np.random.default_rng(1).random((3, 32, 32)).astype(np.float32),
    }
    res = plot_comparison_grid([item], out)
    assert res is not None


def test_error_tile_dict_is_rendered_as_error_image():
    pred = np.zeros((3, 2, 2), dtype=np.float32)
    target = np.ones_like(pred)
    img = _ensure_image({"pred": pred, "target": target}, "error")
    assert img.shape == (2, 2, 3)
    assert not np.all(img == 128)  # not the renderer's blank fallback tile


def test_error_tile_accepts_pre_rendered_array():
    rendered = np.full((8, 8, 3), [10, 20, 30], dtype=np.uint8)
    np.testing.assert_array_equal(_ensure_image(rendered, "error"), rendered)


def test_plot_comparison_grid_bad_value_returns_none(tmp_path: Path, capsys):
    out = tmp_path / "grid.png"
    item = _full_item()
    item["input"] = "not a numpy array"  # bad value
    res = plot_comparison_grid([item], out)
    # Should still return a path; the renderer writes a blank tile
    # for the bad column.
    assert res is not None
    assert out.exists()
