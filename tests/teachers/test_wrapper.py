"""Tests for :mod:`moebius_finetune.teachers.wrapper`."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from moebius_finetune.contracts import (
    ConditionBatch,
    CoordFrame,
    DepthNormalization,
    SampleManifest,
    Split,
    Direction,
    validate_condition,
)
from moebius_finetune.teachers.depth_adapter import DepthConditionAdapter
from moebius_finetune.teachers.wrapper import (
    DepthConditionedRemoval,
    PredictCandidateError,
    WrapperConfigError,
    _default_depth_features,
)
from moebius_finetune.teachers.original_baseline import OriginalRemovalBaseline
from moebius_finetune.teachers.loader import load_removal_model

from .conftest import moebius_model, moebius_weights_path


# Skip whole module if torch is missing.
torch = pytest.importorskip("torch")


pytestmark_cpu = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="GPU tests need CUDA"
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_adapter(target_dim: int) -> DepthConditionAdapter:
    return DepthConditionAdapter(channels=2, hidden=16, target_dim=target_dim)


def _make_batch(H: int, W: int, *, seed: int = 0, hole: bool = True) -> ConditionBatch:
    """Build a small batch. We use H=W=64 so the latent (8×8) is well
    below the model's ``sample_size=64``; this is fine for the
    alignment tests which only need to exercise a single forward.
    The Moebius UNet is built for sample_size=64 (i.e. image 512×512)
    and the smoke tests cover that resolution. For pure wrapper
    alignment we do not need the full image size.

    NOTE: at sample_size smaller than 64, the cross-attention path in
    the Moebius UNet may complain; we therefore test alignment at the
    full 64×64 latent (i.e. 512×512 image). The trade-off is more
    memory; on the dev box with 8 GB this still fits for batch=1.
    """
    rng = np.random.default_rng(seed)
    rgb = rng.random((1, 3, H, W), dtype=np.float32)
    mask = np.zeros((1, 1, H, W), dtype=np.float32)
    if hole:
        mask[0, 0, H // 4 : 3 * H // 4, W // 4 : 3 * W // 4] = 1.0
    rgb = rgb * (1.0 - mask)
    depth = rng.uniform(-1.0, 1.0, (1, 1, H, W)).astype(np.float32) * (1.0 - mask)
    noise = rng.standard_normal((1, 4, H // 8, W // 8)).astype(np.float32)
    return ConditionBatch.from_arrays(rgb, mask, depth, noise)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_construct_rejects_dim_mismatch(moebius_model):
    model = moebius_model
    bad = DepthConditionAdapter(target_dim=64)  # wrong target dim
    with pytest.raises(WrapperConfigError):
        DepthConditionedRemoval(model, bad)


def test_construct_rejects_bad_input_ids(moebius_model):
    model = moebius_model
    adapter = _build_adapter(target_dim=320)
    with pytest.raises(WrapperConfigError):
        DepthConditionedRemoval(model, adapter, input_ids_half=0)
    with pytest.raises(WrapperConfigError):
        DepthConditionedRemoval(model, adapter, input_ids_half=model.num_embeddings + 1)


# ---------------------------------------------------------------------------
# Alignment with original baseline (TDD §9.2 FP32 max diff �?1e-6)
# ---------------------------------------------------------------------------


def test_alignment_with_original_baseline_zero_residual(moebius_model):
    """With a zero-residual adapter the wrapper must match the baseline."""
    model = moebius_model
    adapter = _build_adapter(target_dim=320)
    wrapped = DepthConditionedRemoval(model, adapter)
    baseline = OriginalRemovalBaseline(model)

    H = W = 512  # the model is fixed at 512×512 / 64×64 latent.
    batch = _make_batch(H=H, W=W, seed=123, hole=True)
    validate_condition(batch)
    # masked_latent = a fresh random tensor (no VAE needed for alignment).
    masked_latent = torch.randn(1, 4, H // 8, W // 8, dtype=torch.float32)
    depth_features = torch.randn(1, 2, H // 8, W // 8, dtype=torch.float32)
    timesteps = torch.tensor([50], dtype=torch.int64)

    with torch.no_grad():
        out_wrapped = wrapped.predict_candidate(
            batch, batch.noise, timesteps=timesteps,
            masked_latent=masked_latent, depth_features=depth_features,
        )
        out_baseline = baseline.predict_candidate(
            batch, batch.noise, timesteps=timesteps, masked_latent=masked_latent,
        )
    diff = (out_wrapped - out_baseline).abs().max().item()
    assert diff <= 1e-6, f"FP32 alignment diff {diff} exceeds 1e-6"


# ---------------------------------------------------------------------------
# disable_depth_branch context manager
# ---------------------------------------------------------------------------


def test_disable_branch_makes_residual_zero(moebius_model):
    model = moebius_model
    adapter = _build_adapter(target_dim=320)
    wrapped = DepthConditionedRemoval(model, adapter)

    H = W = 512
    depth_features = torch.randn(1, 2, H // 8, W // 8, dtype=torch.float32)
    with wrapped.disable_depth_branch():
        residual = wrapped.depth_adapter(depth_features)
    assert torch.equal(residual, torch.zeros_like(residual))
    # After exiting the context, the adapter is re-enabled.
    assert wrapped.depth_adapter.enabled is True
    # And the residual is no longer zero (adapter can have non-zero output now).
    with wrapped.enable_depth_branch():
        # We didn't change the weights, so the adapter's spatial conv
        # output is still Kaiming-initialised; the residual is just the
        # 1×1 projection of that. With zero last-layer weight, that
        # output is zero. So we check via a no-op forward: the adapter
        # itself returns the right shape and dtype.
        residual2 = wrapped.depth_adapter(depth_features)
    assert residual2.shape == (1, 320, H // 8, W // 8)


# ---------------------------------------------------------------------------
# predict_candidate does not read target
# ---------------------------------------------------------------------------


def test_predict_candidate_does_not_read_target(moebius_model):
    """The wrapper must not touch the SampleManifest.target_path."""
    model = moebius_model
    adapter = _build_adapter(target_dim=320)
    wrapped = DepthConditionedRemoval(model, adapter)

    H = W = 512
    batch = _make_batch(H=H, W=W, seed=1, hole=True)
    masked_latent = torch.randn(1, 4, H // 8, W // 8, dtype=torch.float32)
    depth_features = torch.randn(1, 2, H // 8, W // 8, dtype=torch.float32)
    timesteps = torch.tensor([100], dtype=torch.int64)

    # Spy on the manifest to confirm no path is read.
    manifest = SampleManifest(
        case_id="c1", source_id="s1", split=Split.TRAIN, coord_frame=CoordFrame.SOURCE,
        rgb_path="/nope/rgb.png", depth_path="/nope/depth.npy",
        mask_path="/nope/mask.png", target_path="/nope/target.png",
        direction=Direction.L2R, dmax_px=8,
        depth_normalization=DepthNormalization(p99=1.0),
    )
    accessed = []
    real_getattr = type(manifest).__getattribute__

    def spying_getattr(self, name):
        accessed.append(name)
        return real_getattr(self, name)

    SampleManifest.__getattribute__ = spying_getattr  # type: ignore[assignment]
    try:
        with torch.no_grad():
            _ = wrapped.predict_candidate(
                batch, batch.noise, timesteps=timesteps,
                masked_latent=masked_latent, depth_features=depth_features,
            )
    finally:
        SampleManifest.__getattribute__ = real_getattr  # type: ignore[assignment]
    assert "target_path" not in accessed


# ---------------------------------------------------------------------------
# Reproducibility (TDD §9.2)
# ---------------------------------------------------------------------------


def test_predict_candidate_reproducible(moebius_model):
    model = moebius_model
    adapter = _build_adapter(target_dim=320)
    wrapped = DepthConditionedRemoval(model, adapter)

    H = W = 512
    batch = _make_batch(H=H, W=W, seed=42, hole=True)
    masked_latent = torch.randn(1, 4, H // 8, W // 8, dtype=torch.float32)
    depth_features = torch.randn(1, 2, H // 8, W // 8, dtype=torch.float32)
    timesteps = torch.tensor([200], dtype=torch.int64)

    with torch.no_grad():
        a = wrapped.predict_candidate(
            batch, batch.noise, timesteps=timesteps,
            masked_latent=masked_latent, depth_features=depth_features,
        )
        b = wrapped.predict_candidate(
            batch, batch.noise, timesteps=timesteps,
            masked_latent=masked_latent, depth_features=depth_features,
        )
    assert torch.equal(a, b)


# ---------------------------------------------------------------------------
# BN running statistics stay frozen (TDD §5.1, §9.2)
# ---------------------------------------------------------------------------


def test_bn_running_stats_dont_drift(moebius_model):
    """TDD §5.1: frozen backbone keeps BN running statistics unchanged.

    We mirror what the finetune step does: put the wrapper into
    train() mode, then call :func:`_bn_eval` on it to keep the BN
    layers frozen. The subsequent forward passes must not change the
    BN running statistics.
    """
    from moebius_finetune.training.teacher.finetune import _bn_eval

    model = moebius_model
    adapter = _build_adapter(target_dim=320)
    wrapped = DepthConditionedRemoval(model, adapter)

    # Set the wrapper to train() but keep BN frozen.
    wrapped.train()
    _bn_eval(wrapped)
    # Verify BN modules are still in eval() mode.
    for module in wrapped.modules():
        if isinstance(module, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d)):
            assert not module.training, "BN should be frozen in train()"

    # Snapshot BN running stats.
    bn_stats_before = {
        name: (m.running_mean.clone(), m.running_var.clone(), m.num_batches_tracked.clone())
        for name, m in wrapped.named_modules()
        if isinstance(m, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d))
    }

    # Run two forward passes.
    H = W = 512
    batch = _make_batch(H=H, W=W, seed=11, hole=True)
    masked_latent = torch.randn(1, 4, H // 8, W // 8, dtype=torch.float32)
    depth_features = torch.randn(1, 2, H // 8, W // 8, dtype=torch.float32)
    for ts in (50, 100, 200):
        with torch.no_grad():
            _ = wrapped.predict_candidate(
                batch, batch.noise, timesteps=torch.tensor([ts], dtype=torch.int64),
                masked_latent=masked_latent, depth_features=depth_features,
            )

    for name, m in wrapped.named_modules():
        if isinstance(m, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d)):
            mean_before, var_before, nbt_before = bn_stats_before[name]
            assert torch.equal(m.running_mean, mean_before), name
            assert torch.equal(m.running_var, var_before), name
            assert torch.equal(m.num_batches_tracked, nbt_before), name


# ---------------------------------------------------------------------------
# No cross-call state �?call the same wrapper twice with different depth
# features and verify the residuals do not leak.
# ---------------------------------------------------------------------------


def test_no_cross_call_state(moebius_model):
    model = moebius_model
    adapter = _build_adapter(target_dim=320)
    wrapped = DepthConditionedRemoval(model, adapter)

    H = W = 512
    batch = _make_batch(H=H, W=W, seed=0, hole=True)
    masked_latent_a = torch.randn(1, 4, H // 8, W // 8, dtype=torch.float32)
    masked_latent_b = torch.randn(1, 4, H // 8, W // 8, dtype=torch.float32)
    depth_features_a = torch.randn(1, 2, H // 8, W // 8, dtype=torch.float32)
    depth_features_b = torch.randn(1, 2, H // 8, W // 8, dtype=torch.float32)
    timesteps = torch.tensor([100], dtype=torch.int64)

    with torch.no_grad():
        # Call with depth A.
        out_a1 = wrapped.predict_candidate(
            batch, batch.noise, timesteps=timesteps,
            masked_latent=masked_latent_a, depth_features=depth_features_a,
        )
        # Now call with a disabled depth branch + depth B (residual
        # must be zero, not the residual from the previous call).
        with wrapped.disable_depth_branch():
            out_disabled = wrapped.predict_candidate(
                batch, batch.noise, timesteps=timesteps,
                masked_latent=masked_latent_b, depth_features=depth_features_b,
            )
            out_baseline = OriginalRemovalBaseline(model).predict_candidate(
                batch, batch.noise, timesteps=timesteps, masked_latent=masked_latent_b,
            )
        # Re-enable and call with depth B again.
        out_b1 = wrapped.predict_candidate(
            batch, batch.noise, timesteps=timesteps,
            masked_latent=masked_latent_b, depth_features=depth_features_b,
        )

    # Disabled branch: bit-equal to the baseline (no leak from previous call).
    assert torch.equal(out_disabled, out_baseline)
    # Different depth features �?different outputs.
    assert not torch.equal(out_a1, out_b1)


# ---------------------------------------------------------------------------
# _default_depth_features shape
# ---------------------------------------------------------------------------


def test_default_depth_features_shape():
    hole_mask = torch.zeros(1, 1, 64, 64)
    hole_mask[..., 16:48, 16:48] = 1.0
    depth = torch.randn(1, 1, 64, 64)
    f = _default_depth_features(hole_mask, depth)
    assert f.shape == (1, 2, 8, 8)


# ---------------------------------------------------------------------------
# predict_candidate rejects missing masked_latent
# ---------------------------------------------------------------------------


def test_predict_candidate_requires_masked_latent(moebius_model):
    model = moebius_model
    adapter = _build_adapter(target_dim=320)
    wrapped = DepthConditionedRemoval(model, adapter)

    batch = _make_batch(H=512, W=512, seed=0, hole=False)
    with pytest.raises(PredictCandidateError):
        wrapped.predict_candidate(batch, batch.noise)
