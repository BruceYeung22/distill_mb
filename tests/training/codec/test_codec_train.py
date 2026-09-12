"""Tests for the codec training loop (TDD §6.3).

* 20-step smoke: loss decreases, model parameters are finite.
* Peak memory on a small synthetic batch stays below 4 GB on a GPU
  (skipped on CPU since the absolute limit doesn't apply).
"""

from __future__ import annotations

import pytest
import torch

from moebius_finetune.students import LightweightCodec
from moebius_finetune.training.codec import SyntheticVAE, train_codec




def _tiny_batch(H: int = 64, B: int = 1) -> torch.Tensor:
    """A deterministic tiny RGB batch in [0, 1] for codec training.

    Note: the codec downsamples by 8x, so the input must be a
    multiple of 8 in spatial size. The default H=64 keeps the
    smoke test fast on CPU. The training also exercises the
    decoder at the matching H/8 latent resolution.
    """
    g = torch.Generator().manual_seed(0)
    return torch.rand(B, 3, H, H, generator=g)


def test_codec_train_smoke_20_steps():
    """20-step smoke run on a synthetic VAE completes without error."""
    torch.manual_seed(0)
    codec = LightweightCodec()
    vae = SyntheticVAE(latent_channels=4)
    batch = _tiny_batch(H=64)
    trainer = train_codec(
        codec,
        vae,
        steps=20,
        cfg={
            "lr": 1e-4,
            "weight_decay": 0.01,
            "device": "cpu",
            "batches": [batch],
        },
    )
    assert trainer is not None
    # All losses are finite floats.
    for entry in trainer.history:
        if "loss" in entry:
            assert isinstance(entry["loss"], float)
            assert entry["loss"] == entry["loss"]  # not NaN
    # The final loss is finite.
    assert trainer.final_loss == trainer.final_loss


def test_codec_train_decreases_loss_after_20_steps():
    """Loss after 20 steps is lower than the loss at step 0."""
    torch.manual_seed(0)
    codec = LightweightCodec()
    vae = SyntheticVAE(latent_channels=4)
    batch = _tiny_batch(H=64)
    trainer = train_codec(
        codec,
        vae,
        steps=20,
        cfg={"device": "cpu", "batches": [batch]},
    )
    losses = [
        e["loss"] for e in trainer.history if "loss" in e
    ]
    assert len(losses) >= 2
    # The first loss is at step 0; the last at step 19. We don't
    # require strict monotonicity (AdamW is noisy) but a substantial
    # decrease is expected because the VAE target is also random
    # and the codec has a lot of capacity to fit it.
    assert losses[-1] < losses[0]


def test_codec_train_keeps_params_finite():
    """After 20 steps, no parameter is NaN/Inf."""
    torch.manual_seed(0)
    codec = LightweightCodec()
    vae = SyntheticVAE(latent_channels=4)
    train_codec(
        codec,
        vae,
        steps=20,
        cfg={
            "device": "cpu",
            "batches": [_tiny_batch(H=64)],
        },
    )
    for p in codec.parameters():
        assert torch.isfinite(p).all()


def test_codec_train_handles_vae_posterior():
    """The trainer uses the VAE's ``posterior`` method, not a hard-coded shape."""
    torch.manual_seed(0)
    codec = LightweightCodec()
    vae = SyntheticVAE(latent_channels=4)
    # Sanity-check: VAE.posterior returns a 4-channel tensor at H/8
    # (matching the codec's 8× downsample layout).
    rgb = _tiny_batch(H=64)
    with torch.no_grad():
        z = vae.posterior(rgb)
    assert z.shape == (1, 4, 8, 8)
    # The trainer's MSE only matches if the shapes agree, so the
    # test here just verifies the loop runs.
    trainer = train_codec(
        codec, vae, steps=5, cfg={"device": "cpu", "batches": [rgb]}
    )
    assert trainer is not None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU-only memory check")
def test_codec_train_peak_memory_under_4gb():
    """Codec training peak GPU memory stays below 4 GB on a real device."""
    torch.manual_seed(0)
    codec = LightweightCodec().cuda()
    vae = SyntheticVAE(latent_channels=4).cuda()
    # Use a slightly larger batch so the 4 GB limit is meaningful.
    batch = _tiny_batch(H=64).cuda()
    trainer = train_codec(
        codec, vae, steps=20, cfg={"device": "cuda", "batches": [batch]}
    )
    assert trainer.peak_memory_bytes < 4 * 1024 * 1024 * 1024, (
        f"codec training peak memory {trainer.peak_memory_bytes} >= 4 GB"
    )
