"""20-step smoke + resume verification tests for the teacher fine-tune loop."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import numpy as np
import pytest
import torch

from moebius_finetune.teachers.depth_adapter import DepthConditionAdapter
from moebius_finetune.teachers.loader import get_weight_metadata, load_removal_model
from moebius_finetune.teachers.wrapper import DepthConditionedRemoval
from moebius_finetune.training.teacher.finetune import (
    FinetuneConfigError,
    finetune_depth_branch,
    verify_resume_consistency,
)
from moebius_finetune.training.teacher.recipe import LocalSmokeRecipe

from .conftest import moebius_model, moebius_weights_path


pytestmark_gpu = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="GPU tests need CUDA"
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _synthetic_batch_provider(
    H: int = 512, W: int = 512, *, seed: int = 0
):
    """Deterministic synthetic batch provider.

    Yields batches with the same shape contract the trainer expects
    (see :data:`BatchProvider`). The hole and depth are deterministic
    so the smoke test can assert specific properties of the loss.

    The provider also emits a pre-computed ``masked_latent`` (in
    latent space at H/8, W/8) so the smoke does not have to load the
    full Moebius VAE — that would double the smoke's memory cost.
    """
    rng = np.random.default_rng(seed)
    mask = np.zeros((1, 1, H, W), dtype=np.float32)
    mask[0, 0, H // 4 : 3 * H // 4, W // 4 : 3 * W // 4] = 1.0
    rgb = rng.random((1, 3, H, W), dtype=np.float32) * (1.0 - mask)
    depth = rng.uniform(-1.0, 1.0, (1, 1, H, W)).astype(np.float32) * (1.0 - mask)
    noise = rng.standard_normal((1, 4, H // 8, W // 8)).astype(np.float32)
    # Pre-computed masked_latent (1, 4, H/8, W/8). The 4 latent channels
    # are independent normal samples; this is enough to feed the
    # 9-channel UNet input without invoking a real VAE.
    masked_latent = rng.standard_normal((1, 4, H // 8, W // 8)).astype(np.float32)
    state = {"i": 0}

    def provider() -> Dict[str, Any]:
        i = state["i"]
        state["i"] += 1
        # Mix a tiny per-step delta so each step sees a different batch.
        out = {
            "clean_rgb": (rgb + 0.001 * i).astype(np.float32),
            "hole_mask": mask.copy(),
            "depth_hole": depth.copy(),
            "noise": (noise + 0.001 * i).astype(np.float32),
            "masked_latent": (masked_latent + 0.001 * i).astype(np.float32),
        }
        return out

    return provider


# ---------------------------------------------------------------------------
# 20-step smoke
# ---------------------------------------------------------------------------


@pytestmark_gpu
def test_local_smoke_20_steps(moebius_model, moebius_weights_path, tmp_path, caplog=None):
    model = moebius_model
    adapter = DepthConditionAdapter()
    wrapped = DepthConditionedRemoval(model, adapter)
    cfg = LocalSmokeRecipe(steps=20, lr=1e-4)
    provider = _synthetic_batch_provider(H=512, W=512, seed=0)

    meta = get_weight_metadata(moebius_weights_path)
    artifacts = finetune_depth_branch(
        wrapped,
        cfg,
        batch_provider=provider,
        output_dir=tmp_path / "run",
        weight_metadata=meta,
        data_version="smoke_v0",
    )
    # Persist the peak memory number for the handoff report.
    (tmp_path / "smoke_metrics.txt").write_text(
        f"peak_memory_bytes={artifacts.peak_memory_bytes}\n"
        f"global_step={artifacts.global_step}\n"
        f"best_loss={artifacts.best_loss}\n"
        f"peak_memory_gb={artifacts.peak_memory_bytes / (1024 ** 3):.3f}\n",
        encoding="utf-8",
    )
    assert artifacts.global_step == 20
    assert artifacts.best_loss == artifacts.best_loss  # not NaN
    assert artifacts.best_loss != float("inf")
    # The peak memory is recorded (0 on CPU).
    if torch.cuda.is_available():
        # 6 GB cap per spec.
        assert artifacts.peak_memory_bytes < 6 * 1024 ** 3, (
            f"peak memory {artifacts.peak_memory_bytes} exceeds 6 GB"
        )


# ---------------------------------------------------------------------------
# Stage 1 freezes the backbone
# ---------------------------------------------------------------------------


@pytestmark_gpu
def test_stage1_only_trains_depth_adapter(moebius_model, tmp_path):
    model = moebius_model
    adapter = DepthConditionAdapter()
    wrapped = DepthConditionedRemoval(model, adapter)
    cfg = LocalSmokeRecipe(steps=5, lr=1e-4)
    provider = _synthetic_batch_provider(H=512, W=512, seed=1)

    finetune_depth_branch(
        wrapped,
        cfg,
        batch_provider=provider,
        output_dir=tmp_path / "run",
        weight_metadata=None,
        data_version="smoke_v0",
    )
    # The backbone parameters must remain frozen.
    for name, p in wrapped.model.named_parameters():
        assert not p.requires_grad, f"backbone param {name} should be frozen"
    # The depth adapter parameters are trainable.
    for name, p in wrapped.depth_adapter.named_parameters():
        assert p.requires_grad, f"depth param {name} should be trainable"


# ---------------------------------------------------------------------------
# Resume consistency
# ---------------------------------------------------------------------------


@pytestmark_gpu
def test_resume_consistency(moebius_model, tmp_path):
    model = moebius_model
    adapter = DepthConditionAdapter()
    wrapped = DepthConditionedRemoval(model, adapter)
    cfg = LocalSmokeRecipe(steps=4, lr=1e-4)
    provider = _synthetic_batch_provider(H=512, W=512, seed=2)

    artifacts = finetune_depth_branch(
        wrapped,
        cfg,
        batch_provider=provider,
        output_dir=tmp_path / "run",
        weight_metadata={"sha256": "abc"},
        data_version="smoke_v0",
    )
    # The last checkpoint is the final step.
    ckpt = artifacts.checkpoint_paths[-1]
    verify_resume_consistency(ckpt)


@pytestmark_gpu
def test_resume_preserves_global_step(moebius_model, tmp_path):
    """TDD §5.3: ``verify_resume_consistency`` keeps step / random state.

    The function reads the checkpoint twice and compares; we extend
    the contract to also confirm the global step and data_version
    read out are the ones we wrote.
    """
    model = moebius_model
    adapter = DepthConditionAdapter()
    wrapped = DepthConditionedRemoval(model, adapter)
    cfg = LocalSmokeRecipe(steps=3, lr=1e-4)
    provider = _synthetic_batch_provider(H=512, W=512, seed=3)

    artifacts = finetune_depth_branch(
        wrapped,
        cfg,
        batch_provider=provider,
        output_dir=tmp_path / "run",
        weight_metadata=None,
        data_version="smoke_v0",
    )
    ckpt = artifacts.checkpoint_paths[-1]
    verify_resume_consistency(ckpt)
    payload = torch.load(str(ckpt), map_location="cpu", weights_only=False)
    assert int(payload["global_step"]) == 3
    assert str(payload["data_version"]) == "smoke_v0"


# ---------------------------------------------------------------------------
# Finetune rejects non-wrapper model
# ---------------------------------------------------------------------------


def test_finetune_rejects_bad_model(moebius_model, tmp_path):
    class NotAModule:
        pass

    with pytest.raises(FinetuneConfigError):
        finetune_depth_branch(
            NotAModule(),
            LocalSmokeRecipe(steps=1),
            batch_provider=lambda: {},
            output_dir=tmp_path / "x",
        )
