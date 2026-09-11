"""Synthetic tests for :mod:`moebius_finetune.contracts`.

Pure-numpy + stdlib only. No torch, diffusers, transformers, onnx or
rknn is imported. Coverage corresponds to the verification list in the
stage-0 brief.
"""

from __future__ import annotations

import importlib
import sys
import types
from dataclasses import asdict

import numpy as np
import pytest

import moebius_finetune.contracts as contracts
from moebius_finetune.contracts import (
    ConditionBatch,
    ConditionContractError,
    CoordFrame,
    DepthNormalization,
    Direction,
    SampleManifest,
    Split,
    inpaint,
    to_torch,
    validate_condition,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


B, H, W = 2, 16, 24  # H,W multiples of 8, small for fast tests


def _valid_batch(
    *,
    batch_size: int = B,
    height: int = H,
    width: int = W,
    seed: int = 0,
    hole_mask: np.ndarray | None = None,
    rgb_hole: np.ndarray | None = None,
    depth_hole: np.ndarray | None = None,
    noise: np.ndarray | None = None,
) -> ConditionBatch:
    """Build a valid batch with the hole region strictly zeroed."""
    rng = np.random.default_rng(seed)

    if hole_mask is None:
        hole_mask = np.zeros((batch_size, 1, height, width), dtype=np.float32)
        # Add a square hole in the first sample, leave the second one
        # empty so we exercise both empty-mask and non-empty-mask paths.
        hole_mask[0, 0, 4:8, 6:10] = 1.0

    if rgb_hole is None:
        rgb_hole = rng.random((batch_size, 3, height, width), dtype=np.float32)
    # Zero out the hole region in rgb and depth.
    rgb_hole = rgb_hole * (1.0 - hole_mask)

    if depth_hole is None:
        # L_neg_R_pos signed inverse depth in roughly [-1, 1].
        depth_hole = rng.uniform(-1.0, 1.0, size=(batch_size, 1, height, width)).astype(
            np.float32
        )
    depth_hole = depth_hole * (1.0 - hole_mask)

    if noise is None:
        noise = rng.standard_normal((batch_size, 4, height // 8, width // 8)).astype(
            np.float32
        )

    return ConditionBatch(
        rgb_hole=rgb_hole.astype(np.float32, copy=False),
        hole_mask=hole_mask.astype(np.float32, copy=False),
        depth_hole=depth_hole.astype(np.float32, copy=False),
        noise=noise.astype(np.float32, copy=False),
    )


# ---------------------------------------------------------------------------
# 1. Construct a valid batch (with hole; hole pixels must be 0)
# ---------------------------------------------------------------------------


def test_construct_valid_batch_succeeds():
    batch = _valid_batch()
    validate_condition(batch)
    # Hole pixels of rgb and depth are strictly 0.
    mask = batch.hole_mask  # [B, 1, H, W]
    mask_broadcast = np.broadcast_to(mask, batch.rgb_hole.shape)
    assert np.all(batch.rgb_hole[mask_broadcast == 1] == 0.0)
    assert np.all(batch.depth_hole[mask == 1] == 0.0)
    # Noise shape matches the 8x downsampled spec.
    assert batch.noise.shape == (B, 4, H // 8, W // 8)


# ---------------------------------------------------------------------------
# 2. rgb_hole non-zero inside the hole -> ConditionContractError
# ---------------------------------------------------------------------------


def test_rgb_hole_nonzero_in_hole_raises():
    base = _valid_batch()
    bad = ConditionBatch(
        rgb_hole=base.rgb_hole.copy(),
        hole_mask=base.hole_mask,
        depth_hole=base.depth_hole,
        noise=base.noise,
    )
    # Inject a non-zero value into a hole pixel.
    bad.rgb_hole[0, 0, 5, 7] = 0.42
    with pytest.raises(ConditionContractError):
        validate_condition(bad)


# ---------------------------------------------------------------------------
# 3. depth_hole non-zero inside the hole -> ConditionContractError
# ---------------------------------------------------------------------------


def test_depth_hole_nonzero_in_hole_raises():
    base = _valid_batch()
    bad = ConditionBatch(
        rgb_hole=base.rgb_hole,
        hole_mask=base.hole_mask,
        depth_hole=base.depth_hole.copy(),
        noise=base.noise,
    )
    bad.depth_hole[0, 0, 5, 7] = -0.13
    with pytest.raises(ConditionContractError):
        validate_condition(bad)


# ---------------------------------------------------------------------------
# 4. hole_mask values outside {0, 1} -> ConditionContractError
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad_value", [0.5, -0.1, 1.0001, 2.0])
def test_hole_mask_invalid_values_raise(bad_value):
    base = _valid_batch()
    bad = ConditionBatch(
        rgb_hole=base.rgb_hole,
        hole_mask=base.hole_mask.copy(),
        depth_hole=base.depth_hole,
        noise=base.noise,
    )
    bad.hole_mask[0, 0, 0, 0] = bad_value
    with pytest.raises(ConditionContractError):
        validate_condition(bad)


# ---------------------------------------------------------------------------
# 5. depth_hole contains NaN/Inf -> ConditionContractError
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf"), -float("inf")])
def test_depth_hole_non_finite_raises(bad_value):
    base = _valid_batch()
    bad = ConditionBatch(
        rgb_hole=base.rgb_hole,
        hole_mask=base.hole_mask,
        depth_hole=base.depth_hole.copy(),
        noise=base.noise,
    )
    # Put the NaN/Inf in a known (non-hole) pixel to be sure the
    # validator triggers on non-finite values themselves.
    bad.depth_hole[0, 0, 0, 0] = bad_value
    with pytest.raises(ConditionContractError):
        validate_condition(bad)


# ---------------------------------------------------------------------------
# 6. Shape mismatch (rgb B=2, mask B=1) -> ConditionContractError
# ---------------------------------------------------------------------------


def test_shape_mismatch_raises():
    base = _valid_batch()
    bad_mask = np.zeros((1, 1, H, W), dtype=np.float32)
    bad = ConditionBatch(
        rgb_hole=base.rgb_hole,
        hole_mask=bad_mask,
        depth_hole=base.depth_hole,
        noise=base.noise,
    )
    with pytest.raises(ConditionContractError):
        validate_condition(bad)


def test_noise_shape_mismatch_raises():
    base = _valid_batch()
    bad_noise = np.zeros((B, 4, H // 8 + 1, W // 8), dtype=np.float32)
    bad = ConditionBatch(
        rgb_hole=base.rgb_hole,
        hole_mask=base.hole_mask,
        depth_hole=base.depth_hole,
        noise=bad_noise,
    )
    with pytest.raises(ConditionContractError):
        validate_condition(bad)


# ---------------------------------------------------------------------------
# 7. Empty mask -> inpaint returns the input
# ---------------------------------------------------------------------------


def test_inpaint_empty_mask_returns_input():
    batch = _valid_batch(seed=7, hole_mask=np.zeros((B, 1, H, W), dtype=np.float32))
    candidate = np.full_like(batch.rgb_hole, 0.5)
    out = inpaint(batch, candidate)
    assert out.shape == batch.rgb_hole.shape
    assert out.dtype == np.float32
    np.testing.assert_array_equal(out, batch.rgb_hole)


# ---------------------------------------------------------------------------
# 8. Fully-holed mask -> inpaint returns the clamped candidate
# ---------------------------------------------------------------------------


def test_inpaint_full_hole_returns_clamped_candidate():
    full_mask = np.ones((B, 1, H, W), dtype=np.float32)
    batch = _valid_batch(seed=11, hole_mask=full_mask)
    # Candidate has values outside [0, 1] to test clamping.
    candidate = np.full((B, 3, H, W), 1.7, dtype=np.float32)
    candidate[0, 0, 0, 0] = -0.3
    out = inpaint(batch, candidate)
    assert out.shape == batch.rgb_hole.shape
    assert out.dtype == np.float32
    assert out.min() >= 0.0
    assert out.max() <= 1.0
    # Fully-holed case: result equals the clamped candidate.
    np.testing.assert_allclose(out, np.clip(candidate, 0.0, 1.0))


# ---------------------------------------------------------------------------
# 9. Known region preserved (composite of known rgb_hole + candidate)
# ---------------------------------------------------------------------------


def test_inpaint_known_region_preserved_and_hole_filled():
    batch = _valid_batch(seed=3)  # first sample has a small hole
    candidate = np.full_like(batch.rgb_hole, 0.25)
    out = inpaint(batch, candidate)
    mask = batch.hole_mask  # [B, 1, H, W]
    mask_broadcast = np.broadcast_to(mask, batch.rgb_hole.shape)
    # In the hole, output equals the clamped candidate.
    np.testing.assert_array_equal(out[mask_broadcast == 1], 0.25)
    # Outside the hole, output equals the original rgb_hole.
    np.testing.assert_array_equal(out[mask_broadcast == 0], batch.rgb_hole[mask_broadcast == 0])


def test_inpaint_rejects_non_finite_candidate():
    batch = _valid_batch(seed=1)
    candidate = np.full_like(batch.rgb_hole, 0.5)
    candidate[0, 0, 0, 0] = float("nan")
    with pytest.raises(ConditionContractError):
        inpaint(batch, candidate)


# ---------------------------------------------------------------------------
# 10. SampleManifest dict round-trip
# ---------------------------------------------------------------------------


def test_sample_manifest_round_trip():
    manifest = SampleManifest(
        case_id="case-0001",
        source_id="coco_train2017_42",
        split=Split.TRAIN,
        coord_frame=CoordFrame.SOURCE,
        rgb_path="images/case-0001_rgb.png",
        depth_path="depth/case-0001_depth.npy",
        mask_path="masks/case-0001_mask.png",
        target_path="targets/case-0001_target.png",
        direction=Direction.L2R,
        dmax_px=12,
        depth_normalization=DepthNormalization(p99=4.5, sign_convention="L_neg_R_pos"),
    )
    d = manifest.to_dict()
    assert d["split"] == "train"
    assert d["coord_frame"] == "source"
    assert d["direction"] == "L2R"
    restored = SampleManifest.from_dict(d)
    assert restored == manifest
    # Also exercise the dataclass equality check.
    assert asdict(restored)["case_id"] == "case-0001"


def test_sample_manifest_from_dict_defaults():
    d = {
        "case_id": "case-x",
        "source_id": "src",
        "split": "holdout_masks",
        "coord_frame": "source",
        "rgb_path": "r.png",
        "depth_path": "d.npy",
        "mask_path": "m.png",
        "target_path": "t.png",
        "direction": "R2L",
        "dmax_px": 8,
    }
    m = SampleManifest.from_dict(d)
    assert m.split == Split.HOLDOUT_MASKS
    assert m.direction == Direction.R2L
    assert isinstance(m.depth_normalization, DepthNormalization)


# ---------------------------------------------------------------------------
# 11. to_torch raises ConditionContractError when torch is unavailable
# ---------------------------------------------------------------------------


def test_to_torch_without_torch_raises(monkeypatch):
    # Build a valid batch first so the failure is purely the torch import.
    batch = _valid_batch()
    validate_condition(batch)

    # Stub the import system so that ``import torch`` raises ImportError
    # inside the contracts module. We replace the cached ``torch`` (if
    # any) with None and patch importlib.import_module to refuse torch.
    monkeypatch.setitem(sys.modules, "torch", None)  # type: ignore[arg-type]

    def _fake_import(name, *args, **kwargs):
        if name == "torch" or name.startswith("torch."):
            raise ImportError(f"simulated missing torch for {name}")
        return importlib.import_module(name, *args, **kwargs)

    monkeypatch.setattr(contracts.importlib, "import_module", _fake_import)

    with pytest.raises(ConditionContractError) as exc:
        to_torch(batch)
    assert "torch" in str(exc.value).lower()


def test_to_torch_with_torch_returns_floats():
    torch = pytest.importorskip("torch")
    batch = _valid_batch()
    out = to_torch(batch)
    assert set(out.keys()) == {"rgb_hole", "hole_mask", "depth_hole", "noise"}
    for k, v in out.items():
        assert isinstance(v, torch.Tensor)
        assert v.dtype == torch.float32
        assert tuple(v.shape) == tuple(getattr(batch, k).shape)


# ---------------------------------------------------------------------------
# 12. from_arrays auto-validates
# ---------------------------------------------------------------------------


def test_from_arrays_auto_validates():
    rng = np.random.default_rng(0)
    rgb = rng.random((1, 3, 8, 8), dtype=np.float32)
    mask = np.zeros((1, 1, 8, 8), dtype=np.float32)
    depth = np.zeros((1, 1, 8, 8), dtype=np.float32)
    noise = np.zeros((1, 4, 1, 1), dtype=np.float32)

    # Valid path: factory succeeds.
    ConditionBatch.from_arrays(rgb, mask, depth, noise)

    # Invalid path: hole-mask shape mismatch -> factory raises.
    with pytest.raises(ConditionContractError):
        ConditionBatch.from_arrays(
            rgb,
            np.zeros((1, 1, 9, 8), dtype=np.float32),
            depth,
            noise,
        )


# ---------------------------------------------------------------------------
# Public import surface (acceptance criterion #2)
# ---------------------------------------------------------------------------


def test_public_imports_available_without_torch():
    expected = {
        "ConditionBatch",
        "ConditionContractError",
        "CoordFrame",
        "DepthNormalization",
        "Direction",
        "SampleManifest",
        "Split",
        "inpaint",
        "to_torch",
        "validate_condition",
    }
    for name in expected:
        assert hasattr(contracts, name), f"contracts.{name} is missing"


def test_contracts_module_does_not_import_torch_at_top_level():
    """Guard rail: the contracts module must be importable without torch.

    The test environment may or may not have torch installed; what we
    require is that the module is *capable* of being imported when
    torch is missing. We simulate that by removing torch from
    sys.modules and re-importing contracts.
    """
    # Snapshot modules we may need to restore.
    saved = {k: v for k, v in sys.modules.items() if k == "torch" or k.startswith("torch.")}
    for k in list(saved):
        sys.modules.pop(k, None)

    # Make ``import torch`` raise ImportError.
    class _BlockTorch(types.ModuleType):
        def __getattr__(self, name):  # pragma: no cover - only on access
            raise ImportError("torch blocked for this test")

    sys.modules["torch"] = _BlockTorch("torch")  # type: ignore[assignment]
    try:
        # Force a fresh import.
        reloaded = importlib.reload(contracts)
        # The module must still expose its public API.
        assert hasattr(reloaded, "ConditionBatch")
        assert hasattr(reloaded, "to_torch")
    finally:
        # Restore the original sys.modules state.
        for k in list(sys.modules):
            if k == "torch" or k.startswith("torch."):
                sys.modules.pop(k, None)
        for k, v in saved.items():
            sys.modules[k] = v
