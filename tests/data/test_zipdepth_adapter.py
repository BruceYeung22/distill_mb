"""Tests for the ZipDepth cache adapter."""

from __future__ import annotations

import numpy as np
import pytest

from moebius_finetune.data.zipdepth_adapter import (
    ZipDepthAdapter,
    disparity_path_for,
    load_disparity,
)


def test_disparity_path_for_prefers_npy_when_pt_missing(tmp_path):
    # Create only .npy
    (tmp_path / "disparity").mkdir()
    np.save(tmp_path / "disparity" / "0001.npy", np.zeros((4, 4), dtype=np.float32))
    p = disparity_path_for("0001", data_root=tmp_path, prefer_torch=True)
    assert p.suffix == ".npy"
    assert p.exists()


def test_disparity_path_for_prefers_torch_when_present(tmp_path):
    (tmp_path / "disparity").mkdir()
    np.save(tmp_path / "disparity" / "0001.npy", np.zeros((4, 4), dtype=np.float32))
    # Create a fake .pt file (the adapter only checks existence, not content)
    (tmp_path / "disparity" / "0001.pt").write_bytes(b"")
    p = disparity_path_for("0001", data_root=tmp_path, prefer_torch=True)
    assert p.suffix == ".pt"


def test_disparity_path_for_default_to_pt(tmp_path):
    (tmp_path / "disparity").mkdir()
    p = disparity_path_for("0001", data_root=tmp_path)
    assert p.suffix == ".pt"
    assert not p.exists()


def test_load_disparity_npy(tmp_path):
    (tmp_path / "disparity").mkdir()
    arr = np.arange(16, dtype=np.float32).reshape(4, 4)
    np.save(tmp_path / "disparity" / "0001.npy", arr)
    loaded = load_disparity(tmp_path / "disparity" / "0001.npy")
    np.testing.assert_array_equal(loaded, arr)
    assert loaded.dtype == np.float32


def test_load_disparity_strips_leading_batch_dim(tmp_path):
    (tmp_path / "disparity").mkdir()
    arr = np.arange(16, dtype=np.float32).reshape(1, 4, 4)
    np.save(tmp_path / "disparity" / "0001.npy", arr)
    loaded = load_disparity(tmp_path / "disparity" / "0001.npy")
    assert loaded.shape == (4, 4)


def test_load_disparity_strips_trailing_channel_dim(tmp_path):
    (tmp_path / "disparity").mkdir()
    arr = np.arange(16, dtype=np.float32).reshape(4, 4, 1)
    np.save(tmp_path / "disparity" / "0001.npy", arr)
    loaded = load_disparity(tmp_path / "disparity" / "0001.npy")
    assert loaded.shape == (4, 4)


def test_load_disparity_unsupported_shape_raises(tmp_path):
    (tmp_path / "disparity").mkdir()
    arr = np.zeros((2, 2, 2, 2), dtype=np.float32)
    np.save(tmp_path / "disparity" / "0001.npy", arr)
    with pytest.raises(FileNotFoundError):
        load_disparity(tmp_path / "disparity" / "0001.npy")


def test_load_disparity_missing_file_raises_with_hint(tmp_path):
    p = tmp_path / "disparity" / "0001.npy"
    with pytest.raises(FileNotFoundError) as exc:
        load_disparity(p)
    msg = str(exc.value)
    assert "ZipDepth" in msg
    assert "0001.npy" in msg


def test_load_disparity_unknown_suffix_raises(tmp_path):
    p = tmp_path / "disparity" / "0001.bin"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"")
    with pytest.raises(FileNotFoundError):
        load_disparity(p)


def test_load_disparity_disabled_torch_on_pt(tmp_path):
    """When the only file is .pt but torch is disabled, raise."""
    p = tmp_path / "disparity" / "0001.pt"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"")
    with pytest.raises(FileNotFoundError):
        load_disparity(p, allow_torch=False)


def test_zipdepth_adapter_load(tmp_path):
    (tmp_path / "disparity").mkdir()
    arr = np.full((8, 8), 0.5, dtype=np.float32)
    np.save(tmp_path / "disparity" / "abc.npy", arr)
    adapter = ZipDepthAdapter(data_root=tmp_path)
    loaded = adapter.load("abc")
    np.testing.assert_array_equal(loaded, arr)


def test_zipdepth_adapter_path_for(tmp_path):
    adapter = ZipDepthAdapter(data_root=tmp_path)
    p = adapter.path_for("xyz")
    assert p == tmp_path / "disparity" / "xyz.pt"


def test_zipdepth_adapter_custom_subdir(tmp_path):
    (tmp_path / "depth_cache").mkdir()
    arr = np.zeros((4, 4), dtype=np.float32)
    np.save(tmp_path / "depth_cache" / "x.npy", arr)
    adapter = ZipDepthAdapter(data_root=tmp_path, subdir="depth_cache")
    loaded = adapter.load("x")
    assert loaded.shape == (4, 4)
