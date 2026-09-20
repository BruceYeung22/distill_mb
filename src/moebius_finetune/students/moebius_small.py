"""MoebiusSmallStudent — Moebius-structured student for GRT warp-hole inpainting.

Spec: ``.omo/plans/moebius-small-distill.md`` (todos 1-3). The student is
distilled from the 226M Moebius UNet (TAESDXL latent space, CFG 1.0 target)
via the multi-granularity recipe in
:mod:`moebius_finetune.training.student.multigranular`.

Design decisions (locked):

* ``channels=(128, 256, 512)``, ``blocks=(5, 5, 5)`` per side — ≈10.2 M
  params / ≈6.4 GMac at ``[1, 9, 64, 64]``.
* 64²/32² stages are attention-free gated depthwise convs
  (:class:`~moebius_finetune.students.gated.GatedDW7Block`); the 16² stage
  carries two softmax multi-query attention blocks (``enc2`` last, ``dec2``
  first), reusing :class:`~moebius_finetune.students.mobile_moebius.MobileMQA`.
* Time conditioning is a **continuous** sinusoidal embedding injected once
  per stage (``t0``/``t1``/``t2``) — there is no ``nn.Embedding`` lookup
  table anywhere in this model.
* Input is the 9-channel contract ``[x_t(4) | latent_mask(1) |
  masked_latent(4)]`` — no depth channels. ``latent_mask`` is built with
  ``mode="nearest"`` upstream.
* ``forward`` returns :class:`StudentOutput`, whose ``block_outputs`` are
  the three feature taps (64²×128, 32²×256, 16²×512) consumed by the
  feature-KD loss.

The down/up sampling and attention blocks are imported from
:mod:`moebius_finetune.students.mobile_moebius` rather than re-derived, so
the two students keep the same (already tested) NPU-friendly primitives.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from diffusers.models.embeddings import TimestepEmbedding, get_timestep_embedding

from moebius_finetune.students.gated import GatedDW7Block
from moebius_finetune.students.mobile_moebius import (
    ConvBNAct,
    Downsample,
    MobileMQA,
    TimeInject,
    Upsample,
)

__all__ = ["MoebiusSmallStudent", "SinusoidalTimeEmbedding", "StudentOutput"]


class SinusoidalTimeEmbedding(nn.Module):
    """Continuous sinusoidal t-embedding followed by a SiLU MLP.

    ``t`` is the raw diffusion timestep (0 .. 1000, float or int, ``[B]``).
    ``flip_sin_to_cos=True`` / ``downscale_freq_shift=0`` match the
    diffusers convention used by the Moebius teacher's time embedder.
    """

    def __init__(self, freq_dim: int = 128, time_dim: int = 256) -> None:
        super().__init__()
        if freq_dim <= 0 or time_dim <= 0:
            raise ValueError(
                f"freq_dim and time_dim must be positive, got {(freq_dim, time_dim)}"
            )
        self.freq_dim = int(freq_dim)
        self.time_dim = int(time_dim)
        self.mlp = TimestepEmbedding(self.freq_dim, self.time_dim, act_fn="silu")

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        if t.dim() != 1:
            raise ValueError(f"t must be 1-D [B], got shape {tuple(t.shape)}")
        t = t.float()
        emb = get_timestep_embedding(
            t,
            self.freq_dim,
            flip_sin_to_cos=True,
            downscale_freq_shift=0,
        )
        return self.mlp(emb.to(dtype=self.mlp.linear_1.weight.dtype))


@dataclass
class StudentOutput:
    """Student forward output.

    ``sample`` is the predicted epsilon ``[B, 4, 64, 64]``;
    ``block_outputs`` are the three feature taps, ordered
    ``[64²×c0, 32²×c1, 16²×c2]``.
    """

    sample: torch.Tensor
    block_outputs: list[torch.Tensor]


class MoebiusSmallStudent(nn.Module):
    """Moebius-structured latent student (~10.2 M params).

    Input ``x``: ``[B, in_channels, 64, 64]`` (default ``in_channels=9``).
    Input ``t``: ``[B]`` raw timesteps (int or float) on the same device.
    Output: :class:`StudentOutput`.
    """

    def __init__(
        self,
        in_channels: int = 9,
        out_channels: int = 4,
        channels: tuple[int, int, int] = (128, 256, 512),
        blocks: tuple[int, int, int] = (5, 5, 5),
        time_dim: int = 256,
    ) -> None:
        super().__init__()
        if in_channels <= 0 or out_channels <= 0:
            raise ValueError(
                f"in_channels and out_channels must be positive, got "
                f"{(in_channels, out_channels)}"
            )
        if len(channels) != 3 or any(c <= 0 for c in channels):
            raise ValueError(f"channels must be 3 positive ints, got {channels}")
        if len(blocks) != 3 or any(b < 1 for b in blocks):
            raise ValueError(f"blocks must be 3 ints >= 1, got {blocks}")

        c0, c1, c2 = channels
        b0, b1, b2 = blocks

        # ----------------------- timestep -----------------------------
        self.time_embed = SinusoidalTimeEmbedding(freq_dim=c0, time_dim=time_dim)
        self.t0 = TimeInject(time_dim, c0)
        self.t1 = TimeInject(time_dim, c1)
        self.t2 = TimeInject(time_dim, c2)

        # ----------------------- conv_in ------------------------------
        self.conv_in = nn.Sequential(
            ConvBNAct(in_channels, in_channels, k=3, groups=in_channels, act=True),
            ConvBNAct(in_channels, c0, k=1, act=False),
        )

        # ----------------------- encoder ------------------------------
        self.enc0 = nn.ModuleList([GatedDW7Block(c0) for _ in range(b0)])
        self.down0 = Downsample(c0, c1)
        self.enc1 = nn.ModuleList([GatedDW7Block(c1) for _ in range(b1)])
        self.down1 = Downsample(c1, c2)
        # last slot of enc2 / first slot of dec2 carry the softmax MQA
        self.enc2 = nn.ModuleList(
            [GatedDW7Block(c2) for _ in range(b2 - 1)] + [MobileMQA(c2, 4, 32)]
        )

        # ----------------------- decoder ------------------------------
        self.dec2 = nn.ModuleList(
            [MobileMQA(c2, 4, 32)] + [GatedDW7Block(c2) for _ in range(b2 - 1)]
        )
        self.up1 = Upsample(c2, c1)
        self.dec1 = nn.ModuleList([GatedDW7Block(c1) for _ in range(b1)])
        self.up0 = Upsample(c1, c0)
        self.dec0 = nn.ModuleList([GatedDW7Block(c0) for _ in range(b0)])

        # ----------------------- head ---------------------------------
        self.head = nn.Sequential(
            ConvBNAct(c0, c0, k=3, groups=c0, act=True),
            nn.Conv2d(c0, out_channels, kernel_size=1),
        )

        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.channels = tuple(int(c) for c in channels)
        self.blocks = tuple(int(b) for b in blocks)

    @staticmethod
    def _stage(stage: nn.ModuleList, x: torch.Tensor) -> torch.Tensor:
        for module in stage:
            x = module(x)
        return x

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> StudentOutput:
        if x.shape[1] != self.in_channels:
            raise ValueError(
                f"expected {self.in_channels} input channels, got {x.shape[1]}"
            )
        if t.dim() != 1 or t.shape[0] != x.shape[0]:
            raise ValueError(
                f"t must be [B] matching x's batch ({x.shape[0]}), got {tuple(t.shape)}"
            )
        temb = self.time_embed(t)

        # encoder 64²
        x = self.conv_in(x)
        x = self.t0(x, temb)
        x = self._stage(self.enc0, x)
        skip0 = tap0 = x

        # encoder 32²
        x = self.down0(x)
        x = self.t1(x, temb)
        x = self._stage(self.enc1, x)
        skip1 = tap1 = x

        # encoder 16²
        x = self.down1(x)
        x = self.t2(x, temb)
        x = self._stage(self.enc2, x)
        tap2 = x

        # decoder 16²
        x = self._stage(self.dec2, x)

        # decoder 32²
        x = self.up1(x)
        x = x + skip1
        x = self.t1(x, temb)
        x = self._stage(self.dec1, x)

        # decoder 64²
        x = self.up0(x)
        x = x + skip0
        x = self.t0(x, temb)
        x = self._stage(self.dec0, x)

        return StudentOutput(sample=self.head(x), block_outputs=[tap0, tap1, tap2])
