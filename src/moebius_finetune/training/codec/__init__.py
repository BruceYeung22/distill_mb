"""Codec training (TDD §6.3).

The codec is supervised against the Moebius VAE posterior mode: a
clean RGB image is encoded by both the VAE (frozen) and the small
codec, and the loss is ``MSE(z_codec, z_vae) + L1(decoded, rgb)``.

For environments where the VAE is unavailable, ``train_codec`` accepts
any object that exposes a ``posterior(rgb)`` returning a 4-channel
scaled latent plus a ``decode(latent)`` returning RGB. The agent's
20-step smoke test uses a synthetic VAE that draws the latent from
the same distribution shape and decodes via a small randomly
initialised conv stack — the goal is to prove the training loop
runs, not to demonstrate fidelity.
"""

from __future__ import annotations

from moebius_finetune.training.codec.train import (
    CodecTrainer,
    SyntheticVAE,
    train_codec,
)

__all__ = ["CodecTrainer", "SyntheticVAE", "train_codec"]
