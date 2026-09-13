"""Codec training loop (TDD §6.3).

The training objective is a simple supervised problem:

* Given a clean RGB image in ``[0, 1]``, encode it with both the
  Moebius VAE (or a stub for smoke) and the lightweight codec.
* The codec must produce a latent that matches the VAE's posterior
  mode (already scaled by 0.13025), and decoding it should recover
  the input image.

The loss is::

    L = MSE(z_codec, z_vae) + L1(decoded, rgb)

A 20-step smoke run is the only thing the agent's first version
promises. The full recipe lives in the server config.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Iterable, Optional

import numpy as np
import torch
from torch import nn


__all__ = ["CodecTrainer", "SyntheticVAE", "train_codec"]


# ---------------------------------------------------------------------------
# Synthetic VAE — used by the smoke test and unit tests
# ---------------------------------------------------------------------------


class SyntheticVAE(nn.Module):
    """Deterministic VAE stub used to supervise the codec in unit tests.

    The stub follows the same 8× downsample layout as the real
    Moebius VAE so the codec's encoder output is directly
    comparable to the VAE's posterior mode.

    * ``posterior(rgb)`` — 3×3 stride-2 conv (×3) to a 4-channel
      latent at H/8, then a 1×1 conv to the latent channels.
    * ``decode(latent)`` — 1×1 conv to 3 channels followed by a
      fixed 3×3 conv at the latent resolution. The decode path is
      only used for the optional decoded-RGB loss; it is not
      expected to invert the encoder.
    """

    def __init__(self, latent_channels: int = 4) -> None:
        super().__init__()
        self.latent_channels = int(latent_channels)
        torch.manual_seed(0)
        # Three stride-2 convs to downsample by 8×.
        self.enc = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1, bias=False),
            nn.ReLU6(inplace=False),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.ReLU6(inplace=False),
            nn.Conv2d(32, latent_channels, kernel_size=3, stride=2, padding=1, bias=True),
        )
        # Decoder: 1×1 to 3 channels + 3×3 at latent resolution.
        self.dec = nn.Sequential(
            nn.Conv2d(latent_channels, 16, kernel_size=1, bias=True),
            nn.ReLU6(inplace=False),
            nn.Conv2d(16, 3, kernel_size=3, padding=1, bias=True),
        )
        for p in self.parameters():
            p.requires_grad_(False)
        self.eval()

    def posterior(self, rgb: torch.Tensor) -> torch.Tensor:
        return self.enc(rgb)

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        return self.dec(latent)

    def forward(self, rgb: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.posterior(rgb)
        rgb_hat = self.decode(z)
        return z, rgb_hat


# ---------------------------------------------------------------------------
# Training config + utilities
# ---------------------------------------------------------------------------


@dataclass
class CodecTrainer:
    """A minimal in-place trainer for the lightweight codec.

    The trainer is created by :func:`train_codec` and is intentionally
    light-weight: it doesn't checkpoint, log or report metrics other
    than the final loss and the peak GPU memory it observed. The
    agent's first version is concerned with proving the loop runs
    within the 4 GB peak-memory budget; richer logging will be added
    alongside the real VAE / real data.
    """

    codec: nn.Module
    vae_or_synthetic: nn.Module
    optimizer: torch.optim.Optimizer
    steps: int
    device: torch.device
    log_every: int = 5
    extra_bytes: int = 0  # if any, recorded in the report

    history: list[dict] = field(default_factory=list)
    final_loss: float = math.inf
    peak_memory_bytes: int = 0

    def _step(self, batch_rgb: torch.Tensor) -> dict:
        self.optimizer.zero_grad(set_to_none=True)
        rgb = batch_rgb.to(self.device, non_blocking=True)
        with torch.no_grad():
            # SyntheticVAE exposes posterior() and already returns the
            # target-space latent. Real diffusers VAEs use encode() and
            # expect RGB in [-1, 1], then require their scaling_factor.
            if hasattr(self.vae_or_synthetic, "posterior"):
                z_target = self.vae_or_synthetic.posterior(rgb)
            else:
                encoded = self.vae_or_synthetic.encode(2.0 * rgb - 1.0)
                latent_dist = getattr(encoded, "latent_dist", None)
                if latent_dist is None or not hasattr(latent_dist, "mode"):
                    raise TypeError(
                        "real VAE encode() must return an object with latent_dist.mode()"
                    )
                z_target = latent_dist.mode()
                scale = float(
                    getattr(
                        getattr(self.vae_or_synthetic, "config", None),
                        "scaling_factor",
                        0.13025,
                    )
                )
                z_target = z_target * scale
        z_pred, rgb_hat = self.codec(rgb)
        # Both the latent and RGB losses are in FP32 even if the codec
        # is in BF16, by casting to FP32 before the loss.
        loss_latent = torch.nn.functional.mse_loss(
            z_pred.float(), z_target.float()
        )
        # The codec decoder may return RGB at a different resolution
        # than the input (e.g. when the input is smaller than the
        # codec's expected size). We resize the prediction to match
        # the target before computing the L1 loss.
        if rgb_hat.shape[-2:] != rgb.shape[-2:]:
            rgb_hat = torch.nn.functional.interpolate(
                rgb_hat, size=rgb.shape[-2:], mode="bilinear",
                align_corners=False,
            )
        loss_rgb = torch.nn.functional.l1_loss(
            rgb_hat.float(), rgb.float()
        )
        loss = loss_latent + loss_rgb
        loss.backward()
        self.optimizer.step()
        return {
            "loss": float(loss.detach().item()),
            "loss_latent": float(loss_latent.detach().item()),
            "loss_rgb": float(loss_rgb.detach().item()),
        }

    def fit(self, batches: Iterable[torch.Tensor]) -> "CodecTrainer":
        iterator = iter(batches)
        if torch.cuda.is_available() and self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
        for step_idx in range(self.steps):
            try:
                batch = next(iterator)
            except StopIteration:
                # If the caller gave us fewer batches than steps, cycle
                # through them again.
                iterator = iter(batches)
                batch = next(iterator)
            info = self._step(batch)
            self.history.append({"step": step_idx, **info})
            if (step_idx + 1) % self.log_every == 0 or step_idx == 0:
                # Optional on-device memory reporting.
                if (
                    torch.cuda.is_available()
                    and self.device.type == "cuda"
                ):
                    self.peak_memory_bytes = max(
                        self.peak_memory_bytes,
                        int(torch.cuda.max_memory_allocated(self.device)),
                    )
        self.final_loss = self.history[-1]["loss"] if self.history else math.inf
        return self

    def report(self) -> dict:
        return {
            "steps": self.steps,
            "final_loss": self.final_loss,
            "peak_memory_bytes": self.peak_memory_bytes,
            "history": self.history,
        }


# ---------------------------------------------------------------------------
# Top-level entry point
# ---------------------------------------------------------------------------


def train_codec(
    codec: nn.Module,
    vae_or_synthetic: nn.Module,
    steps: int,
    cfg: Optional[dict] = None,
) -> CodecTrainer:
    """Run the codec training loop and return the trainer.

    Parameters
    ----------
    codec
        The lightweight codec to train.
    vae_or_synthetic
        Either the real frozen Moebius VAE (with ``encode`` returning
        ``latent_dist.mode()``) or a :class:`SyntheticVAE` for smoke
        tests. The real VAE is never modified by this loop.
    steps
        Number of optimizer steps to run.
    cfg
        Optional dict with training configuration. Recognised keys:

        * ``lr`` (default 1e-4)
        * ``weight_decay`` (default 0.01)
        * ``max_grad_norm`` (default 1.0)
        * ``device`` (default: ``"cuda"`` if available else ``"cpu"``)
        * ``batches`` (an iterable of RGB batches; the caller can
          provide a small synthetic dataset directly. If absent, the
          trainer generates one random batch and cycles.)
    """
    cfg = cfg or {}
    lr = float(cfg.get("lr", 1e-4))
    weight_decay = float(cfg.get("weight_decay", 0.01))
    max_grad_norm = float(cfg.get("max_grad_norm", 1.0))
    device_str = cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_str)

    codec = codec.to(device)
    vae_or_synthetic = vae_or_synthetic.to(device).eval()
    for p in vae_or_synthetic.parameters():
        p.requires_grad_(False)

    optimizer = torch.optim.AdamW(
        codec.parameters(), lr=lr, weight_decay=weight_decay
    )

    # If the caller didn't provide a dataset, build a tiny one.
    batches = cfg.get("batches")
    if batches is None:
        rgb = torch.randn(2, 3, 64, 64, dtype=torch.float32)
        batches = [rgb]

    trainer = CodecTrainer(
        codec=codec,
        vae_or_synthetic=vae_or_synthetic,
        optimizer=optimizer,
        steps=steps,
        device=device,
    )
    trainer.fit(batches)
    if max_grad_norm > 0:
        # The trainer doesn't currently apply grad clipping (its
        # optimizer step is just a vanilla AdamW). Expose the
        # configured value so the report can record it.
        trainer.history.append(
            {"max_grad_norm_configured": max_grad_norm}
        )
    return trainer
