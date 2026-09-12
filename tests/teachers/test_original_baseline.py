"""Tests for :mod:`moebius_finetune.teachers.original_baseline`."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from moebius_finetune.contracts import ConditionBatch, validate_condition
from moebius_finetune.teachers.loader import load_removal_model
from moebius_finetune.teachers.original_baseline import OriginalRemovalBaseline
from moebius_finetune.teachers.wrapper import DepthConditionedRemoval
from moebius_finetune.teachers.depth_adapter import DepthConditionAdapter


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="GPU tests need CUDA"
)


def _make_batch(H: int, W: int, *, seed: int = 0) -> ConditionBatch:
    rng = np.random.default_rng(seed)
    rgb = rng.random((1, 3, H, W), dtype=np.float32)
    mask = np.zeros((1, 1, H, W), dtype=np.float32)
    mask[0, 0, H // 4 : 3 * H // 4, W // 4 : 3 * W // 4] = 1.0
    rgb = rgb * (1.0 - mask)
    depth = rng.uniform(-1.0, 1.0, (1, 1, H, W)).astype(np.float32) * (1.0 - mask)
    noise = rng.standard_normal((1, 4, H // 8, W // 8)).astype(np.float32)
    return ConditionBatch.from_arrays(rgb, mask, depth, noise)


def test_baseline_predict_candidate(moebius_weights_path):
    model = load_removal_model(moebius_weights_path, strict=True)
    baseline = OriginalRemovalBaseline(model)
    H = W = 512
    batch = _make_batch(H=H, W=W, seed=7)
    masked_latent = torch.randn(1, 4, H // 8, W // 8, dtype=torch.float32)
    timesteps = torch.tensor([50], dtype=torch.int64)
    with torch.no_grad():
        out = baseline.predict_candidate(
            batch, batch.noise, timesteps=timesteps, masked_latent=masked_latent
        )
    assert out.shape == (1, 4, H // 8, W // 8)


def test_baseline_requires_masked_latent(moebius_weights_path):
    model = load_removal_model(moebius_weights_path, strict=True)
    baseline = OriginalRemovalBaseline(model)
    batch = _make_batch(H=512, W=512, seed=7)
    with pytest.raises(ValueError):
        baseline.predict_candidate(batch, batch.noise)
