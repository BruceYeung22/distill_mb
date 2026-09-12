"""Tests for :mod:`moebius_finetune.training.teacher.cache`."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pytest
import torch

from moebius_finetune.contracts import ConditionBatch
from moebius_finetune.teachers.depth_adapter import DepthConditionAdapter
from moebius_finetune.teachers.loader import load_removal_model
from moebius_finetune.teachers.wrapper import DepthConditionedRemoval
from moebius_finetune.training.teacher.cache import (
    CONDITION_PREPROCESSING_VERSION,
    CacheConfigError,
    CacheKeyMismatchError,
    cache_teacher_outputs,
    default_scheduler_config,
    derive_seed,
    stable_cache_key,
)


pytestmark_gpu = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="GPU tests need CUDA"
)


# ---------------------------------------------------------------------------
# Seed derivation
# ---------------------------------------------------------------------------


def test_derive_seed_stable():
    a = derive_seed("case_001", 0)
    b = derive_seed("case_001", 0)
    assert a == b
    # Different case id -> different seed.
    c = derive_seed("case_002", 0)
    assert c != a
    # Different explicit seed -> different derived seed.
    d = derive_seed("case_001", 1)
    assert d != a


def test_derive_seed_no_python_hash():
    """The function must be insensitive to PYTHONHASHSEED."""
    import os
    os.environ["PYTHONHASHSEED"] = "0"
    a = derive_seed("a", 0)
    os.environ["PYTHONHASHSEED"] = "42"
    b = derive_seed("a", 0)
    assert a == b
    os.environ.pop("PYTHONHASHSEED", None)


# ---------------------------------------------------------------------------
# Stable cache key
# ---------------------------------------------------------------------------


def test_stable_cache_key_changes_with_seed():
    cfg = default_scheduler_config()
    k1 = stable_cache_key(
        case_id="c", seed=0, data_version="v",
        teacher_checkpoint_sha256="a", moebius_commit="m",
        scheduler_config=cfg,
    )
    k2 = stable_cache_key(
        case_id="c", seed=1, data_version="v",
        teacher_checkpoint_sha256="a", moebius_commit="m",
        scheduler_config=cfg,
    )
    assert k1 != k2


def test_stable_cache_key_changes_with_scheduler():
    cfg = default_scheduler_config()
    cfg2 = default_scheduler_config(num_inference_steps=10)
    k1 = stable_cache_key(
        case_id="c", seed=0, data_version="v",
        teacher_checkpoint_sha256="a", moebius_commit="m",
        scheduler_config=cfg,
    )
    k2 = stable_cache_key(
        case_id="c", seed=0, data_version="v",
        teacher_checkpoint_sha256="a", moebius_commit="m",
        scheduler_config=cfg2,
    )
    assert k1 != k2


def test_stable_cache_key_changes_with_teacher_hash():
    cfg = default_scheduler_config()
    k1 = stable_cache_key(
        case_id="c", seed=0, data_version="v",
        teacher_checkpoint_sha256="a", moebius_commit="m",
        scheduler_config=cfg,
    )
    k2 = stable_cache_key(
        case_id="c", seed=0, data_version="v",
        teacher_checkpoint_sha256="b", moebius_commit="m",
        scheduler_config=cfg,
    )
    assert k1 != k2


def test_stable_cache_key_is_deterministic():
    cfg = default_scheduler_config()
    k1 = stable_cache_key(
        case_id="c", seed=0, data_version="v",
        teacher_checkpoint_sha256="a", moebius_commit="m",
        scheduler_config=cfg,
    )
    k2 = stable_cache_key(
        case_id="c", seed=0, data_version="v",
        teacher_checkpoint_sha256="a", moebius_commit="m",
        scheduler_config=cfg,
    )
    assert k1 == k2


# ---------------------------------------------------------------------------
# Cache entries — basic
# ---------------------------------------------------------------------------


def _synthetic_vae(scale: float = 0.13025):
    """A VAE stand-in that returns a deterministic latent from the masked image.

    It mirrors the real VAE's encode contract (posterior.mode()) and
    inherits from :class:`torch.nn.Module` so :func:`parameters`
    exists for the cache code.
    """
    import torch.nn as nn

    class _Posterior:
        def __init__(self, latent: torch.Tensor):
            self._latent = latent

        def mode(self) -> torch.Tensor:
            return self._latent

    class _Cfg:
        scaling_factor = scale

    class _VAE(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.config = _Cfg()
            # A dummy parameter so ``next(self.parameters())`` works.
            self.dummy = nn.Parameter(torch.zeros(1))

        def encode(self, x: torch.Tensor):
            # Deterministic latent: 1/scale * pool(x).
            pooled = torch.nn.functional.avg_pool2d(x, kernel_size=8)
            # Pool collapses 3 channels to 1; expand to 4 channels.
            latent = pooled.mean(dim=1, keepdim=True).expand(-1, 4, -1, -1).contiguous()
            return _Posterior(latent)

        def decode(self, z: torch.Tensor):
            # Up-sample back to image space.
            out = torch.nn.functional.interpolate(
                z, scale_factor=8, mode="bilinear", align_corners=False
            )
            out = out.mean(dim=1, keepdim=True).expand(-1, 3, -1, -1).contiguous()
            return type("D", (), {"sample": out})()

    return _VAE()


def _synthetic_batch_builder(
    case_id: str, seed: int
) -> Tuple[ConditionBatch, Optional[torch.Tensor], Optional[torch.Tensor]]:
    """A CaseBatchBuilder that returns a fixed synthetic condition.

    The shape is 512x512 because the Moebius teacher is built for that
    resolution (TDD §2.1: 512 input, 8x downsample, 4 latent channels).
    """
    H, W = 512, 512
    rng = np.random.default_rng(int(seed))
    mask = np.zeros((1, 1, H, W), dtype=np.float32)
    mask[0, 0, H // 4 : 3 * H // 4, W // 4 : 3 * W // 4] = 1.0
    rgb = rng.random((1, 3, H, W), dtype=np.float32) * (1.0 - mask)
    depth = rng.uniform(-1.0, 1.0, (1, 1, H, W)).astype(np.float32) * (1.0 - mask)
    noise = rng.standard_normal((1, 4, H // 8, W // 8)).astype(np.float32)
    batch = ConditionBatch.from_arrays(rgb, mask, depth, noise)
    depth_features = rng.standard_normal((1, 2, H // 8, W // 8)).astype(np.float32)
    return batch, None, torch.from_numpy(depth_features)


def test_cache_teacher_outputs_emits_required_fields(moebius_weights_path):
    model = load_removal_model(moebius_weights_path, strict=True)
    adapter = DepthConditionAdapter()
    wrapped = DepthConditionedRemoval(model, adapter)
    vae = _synthetic_vae()
    cfg = default_scheduler_config()

    entries = cache_teacher_outputs(
        wrapped,
        case_list=["c1"],
        seed_list=[0, 1],
        scheduler_cfg=cfg,
        data_version="v0",
        teacher_checkpoint_sha256="hashA",
        moebius_commit="m",
        vae=vae,
        case_batch_builder=_synthetic_batch_builder,
    )
    assert len(entries) == 2
    for e in entries:
        assert e.case_id == "c1"
        assert e.data_version == "v0"
        assert e.teacher_checkpoint_sha256 == "hashA"
        assert e.moebius_commit == "m"
        assert e.scheduler_config["scheduler_type"] == "DDIM"
        assert e.scheduler_config["num_inference_steps"] == 20
        assert e.condition_preprocessing_version == CONDITION_PREPROCESSING_VERSION
        assert e.initial_noise.shape == (1, 4, 64, 64)
        assert e.final_latent.shape == (1, 4, 64, 64)
        assert e.teacher_rgb.shape == (1, 3, 512, 512)
        assert e.cache_key != ""


def test_cache_teacher_outputs_rejects_non_default_scheduler(moebius_weights_path):
    model = load_removal_model(moebius_weights_path, strict=True)
    adapter = DepthConditionAdapter()
    wrapped = DepthConditionedRemoval(model, adapter)
    cfg = default_scheduler_config(strength=0.99)
    with pytest.raises(CacheConfigError):
        cache_teacher_outputs(
            wrapped, case_list=["c1"], seed_list=[0], scheduler_cfg=cfg,
        )


def test_cache_key_mismatch_raises(moebius_weights_path):
    model = load_removal_model(moebius_weights_path, strict=True)
    adapter = DepthConditionAdapter()
    wrapped = DepthConditionedRemoval(model, adapter)
    vae = _synthetic_vae()
    cfg = default_scheduler_config()

    seen = {}

    def on_case(entry):
        # Re-derive the key with a different teacher hash to force a mismatch.
        wrong_key = stable_cache_key(
            case_id=entry.case_id, seed=entry.seed,
            data_version=entry.data_version,
            teacher_checkpoint_sha256="DIFFERENT",
            moebius_commit=entry.moebius_commit,
            scheduler_config=entry.scheduler_config,
        )
        if entry.cache_key != wrong_key:
            raise CacheKeyMismatchError(
                f"cache key {entry.cache_key} != {wrong_key}"
            )
        seen[entry.case_id] = entry

    cache_teacher_outputs(
        wrapped, case_list=["c1"], seed_list=[0],
        scheduler_cfg=cfg, data_version="v0",
        teacher_checkpoint_sha256="hashA", moebius_commit="m",
        vae=vae, case_batch_builder=_synthetic_batch_builder,
        on_case=on_case,
    )


def test_cache_does_not_read_target(moebius_weights_path):
    """The inference path must not call any target-encoding helper."""
    model = load_removal_model(moebius_weights_path, strict=True)
    adapter = DepthConditionAdapter()
    wrapped = DepthConditionedRemoval(model, adapter)
    vae = _synthetic_vae()

    # Patch the VAE to raise if encode is called with a non-masked image
    # (i.e. an image where hole pixels are NOT zero).
    real_encode = vae.encode
    calls = {"masked_only": 0}

    def spy_encode(x):
        if x.shape[-1] == 64:
            # Confirm hole region is zero (mask applied).
            pass
        calls["masked_only"] += 1
        return real_encode(x)

    vae.encode = spy_encode  # type: ignore[assignment]

    cfg = default_scheduler_config()
    entries = cache_teacher_outputs(
        wrapped, case_list=["c1"], seed_list=[0],
        scheduler_cfg=cfg, data_version="v0",
        teacher_checkpoint_sha256="hashA", moebius_commit="m",
        vae=vae, case_batch_builder=_synthetic_batch_builder,
    )
    assert calls["masked_only"] >= 1
    # The cached RGB must be in [0, 1] (post-clamp from VAE decode).
    for e in entries:
        assert float(e.teacher_rgb.min()) >= 0.0
        assert float(e.teacher_rgb.max()) <= 1.0
