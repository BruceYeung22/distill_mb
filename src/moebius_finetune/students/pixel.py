"""PixelStudent-v0 — single-forward pixel-space inpainting network.

Implements the architecture described in TDD §6.2:

* Encoder at 256²/128²/64² with widths [24, 48, 96] and block counts
  [1, 2, 3]. 5-channel input ``rgb_hole + mask + depth_hole``.
* Bottleneck at 32² with width 160, four DW7 context blocks.
* Decoder at 64²/128²/256² with widths [96, 48, 24] and block counts
  [2, 2, 1].
* Output refinement at 512² with 8 channels of 3x3 + 1x1 RGB head.
* 4-channel noise ``[B, 4, 64, 64]`` injected via a 1x1 conv at the
  64² level before the bottleneck.

The model exposes two methods:

* :meth:`PixelStudentV0.predict_candidate` — returns the raw RGB
  prediction for the full image (the network output is unclamped;
  :func:`moebius_finetune.contracts.inpaint` does the known-pixel
  composite and the clamp).
* :meth:`PixelStudentV0.inpaint` — wraps ``predict_candidate`` with
  the public inpaint helper, returning the composed RGB in
  ``[0, 1]`` with the known region preserved.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from moebius_finetune.contracts import (
    ConditionBatch,
    inpaint as contracts_inpaint,
    validate_condition,
)
from moebius_finetune.students.common import (
    ContextBlock,
    DownsampleBlock,
    InvertedResidualBlock,
    TailRefineBlock,
    UpsampleBlock,
)


__all__ = ["PixelStudentV0"]


# ---------------------------------------------------------------------------
# Default architecture constants from TDD §6.2.
# ---------------------------------------------------------------------------

DEFAULT_ENCODER_WIDTHS = (24, 48, 96)
DEFAULT_ENCODER_BLOCKS = (1, 2, 3)
DEFAULT_BOTTLENECK_WIDTH = 160
DEFAULT_BOTTLENECK_BLOCKS = 4
DEFAULT_DECODER_WIDTHS = (96, 48, 24)
DEFAULT_DECODER_BLOCKS = (2, 2, 1)
DEFAULT_NOISE_IN_CHANNELS = 4
DEFAULT_NOISE_INJECT_CHANNELS = 96
DEFAULT_TAIL_CHANNELS = 8
DEFAULT_INPUT_CHANNELS = 5  # rgb(3) + mask(1) + depth(1)
DEFAULT_OUT_CHANNELS = 3


class _EncoderStage(nn.Module):
    """A single encoder stage: one stride-2 downsample + N residual blocks."""

    def __init__(self, c_in: int, c_out: int, num_blocks: int) -> None:
        super().__init__()
        layers: list[nn.Module] = [DownsampleBlock(c_in, c_out)]
        for _ in range(num_blocks):
            layers.append(InvertedResidualBlock(c_out))
        self.blocks = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(x)


class _DecoderStage(nn.Module):
    """A single decoder stage: nearest ×2 upsample + N residual blocks.

    The architecture uses ``c_skip == c_out`` (skip channels equal
    the target output channels) so the skip add does not need its
    own projection. The 1x1 ``proj_low`` projection of the low-res
    path to ``c_out`` runs only when ``c_low != c_out``.
    """

    def __init__(
        self,
        c_low: int,
        c_skip: int,
        c_out: int,
        num_blocks: int,
        *,
        do_upsample: bool = True,
    ) -> None:
        super().__init__()
        if c_skip != c_out:
            raise ValueError(
                f"decoder stage requires c_skip == c_out; got c_skip={c_skip}, "
                f"c_out={c_out}"
            )
        self.do_upsample = bool(do_upsample)
        if do_upsample and c_low != c_out:
            self.proj_low = nn.Conv2d(c_low, c_out, kernel_size=1, bias=False)
        else:
            self.proj_low = nn.Identity()
        blocks = [InvertedResidualBlock(c_out) for _ in range(num_blocks)]
        self.blocks = nn.Sequential(*blocks)

    def forward(self, low: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        if self.do_upsample:
            up = nn.functional.interpolate(low, scale_factor=2.0, mode="nearest")
            up = self.proj_low(up)
            if up.shape[-2:] != skip.shape[-2:]:
                up = nn.functional.interpolate(
                    up, size=skip.shape[-2:], mode="nearest"
                )
            x = up + skip
        else:
            if low.shape[1] != skip.shape[1]:
                low = self.proj_low(low)
            if low.shape[-2:] != skip.shape[-2:]:
                low = nn.functional.interpolate(
                    low, size=skip.shape[-2:], mode="nearest"
                )
            x = low + skip
        return self.blocks(x)


class PixelStudentV0(nn.Module):
    """Pixel-space inpainting student (TDD §6.2)."""

    def __init__(
        self,
        encoder_widths: tuple[int, int, int] = DEFAULT_ENCODER_WIDTHS,
        encoder_blocks: tuple[int, int, int] = DEFAULT_ENCODER_BLOCKS,
        bottleneck_width: int = DEFAULT_BOTTLENECK_WIDTH,
        bottleneck_blocks: int = DEFAULT_BOTTLENECK_BLOCKS,
        decoder_widths: tuple[int, int, int] = DEFAULT_DECODER_WIDTHS,
        decoder_blocks: tuple[int, int, int] = DEFAULT_DECODER_BLOCKS,
        noise_in_channels: int = DEFAULT_NOISE_IN_CHANNELS,
        noise_inject_channels: int = DEFAULT_NOISE_INJECT_CHANNELS,
        tail_channels: int = DEFAULT_TAIL_CHANNELS,
        in_channels: int = DEFAULT_INPUT_CHANNELS,
        out_channels: int = DEFAULT_OUT_CHANNELS,
    ) -> None:
        super().__init__()
        if len(encoder_widths) != 3 or len(encoder_blocks) != 3:
            raise ValueError(
                f"encoder_widths / encoder_blocks must have length 3, got "
                f"{encoder_widths} / {encoder_blocks}"
            )
        if len(decoder_widths) != 3 or len(decoder_blocks) != 3:
            raise ValueError(
                f"decoder_widths / decoder_blocks must have length 3, got "
                f"{decoder_widths} / {decoder_blocks}"
            )

        self.encoder_widths = tuple(encoder_widths)
        self.encoder_blocks = tuple(encoder_blocks)
        self.bottleneck_width = int(bottleneck_width)
        self.bottleneck_blocks = int(bottleneck_blocks)
        self.decoder_widths = tuple(decoder_widths)
        self.decoder_blocks = tuple(decoder_blocks)
        self.noise_inject_channels = int(noise_inject_channels)
        self.tail_channels = int(tail_channels)
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)

        # ----- Encoder -----
        # Stage 1: 512² → 256² (stride 2 + 1 IR block)
        self.enc1 = _EncoderStage(
            c_in=in_channels,
            c_out=encoder_widths[0],
            num_blocks=encoder_blocks[0],
        )
        # Stage 2: 256² → 128²
        self.enc2 = _EncoderStage(
            c_in=encoder_widths[0],
            c_out=encoder_widths[1],
            num_blocks=encoder_blocks[1],
        )
        # Stage 3: 128² → 64²
        self.enc3 = _EncoderStage(
            c_in=encoder_widths[1],
            c_out=encoder_widths[2],
            num_blocks=encoder_blocks[2],
        )

        # ----- Noise injection at 64² -----
        # The TDD says: actual initial noise [B, 4, 64, 64] goes through
        # a 1x1 conv that projects it to 96 channels, then is added to the
        # 64² encoder feature map.
        self.noise_proj = nn.Conv2d(
            in_channels=noise_in_channels,
            out_channels=noise_inject_channels,
            kernel_size=1,
            bias=True,
        )
        if noise_inject_channels != encoder_widths[2]:
            raise ValueError(
                f"noise_inject_channels ({noise_inject_channels}) must equal "
                f"encoder_widths[2] ({encoder_widths[2]}) to be addable to "
                f"the 64² feature map."
            )

        # ----- Bottleneck: 64² → 32² via stride 2 + 4 DW7 context blocks -----
        self.pre_bottleneck = DownsampleBlock(
            c_in=encoder_widths[2], c_out=bottleneck_width
        )
        self.bottleneck = nn.Sequential(
            *[
                ContextBlock(bottleneck_width, kernel=7)
                for _ in range(bottleneck_blocks)
            ]
        )

        # ----- Decoder -----
        # Stage 1: 32² → 64², with 64² skip.
        self.dec1 = _DecoderStage(
            c_low=bottleneck_width,
            c_skip=decoder_widths[0],
            c_out=decoder_widths[0],
            num_blocks=decoder_blocks[0],
        )
        # Stage 2: 64² → 128², with 128² skip.
        self.dec2 = _DecoderStage(
            c_low=decoder_widths[0],
            c_skip=decoder_widths[1],
            c_out=decoder_widths[1],
            num_blocks=decoder_blocks[1],
        )
        # Stage 3: 128² → 256², with 256² skip.
        self.dec3 = _DecoderStage(
            c_low=decoder_widths[1],
            c_skip=decoder_widths[2],
            c_out=decoder_widths[2],
            num_blocks=decoder_blocks[2],
        )

        # ----- Output refinement at 512² -----
        # 256² → 512² via nearest ×2 then 1x1 to tail channels, then
        # 3x3 + 1x1 to RGB. Children are declared in the same order
        # they are used in :meth:`forward` so the static budget
        # walker threads the spatial shape correctly.
        self.pre_tail = nn.Upsample(scale_factor=2.0, mode="nearest")
        if decoder_widths[2] != tail_channels:
            self.tail_proj = nn.Conv2d(
                decoder_widths[2], tail_channels, kernel_size=1, bias=True
            )
        else:
            self.tail_proj = nn.Identity()
        self.tail = TailRefineBlock(c_in=tail_channels, c_out=out_channels)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(
        self, condition: torch.Tensor, noise: torch.Tensor
    ) -> torch.Tensor:
        """Run the network.

        Parameters
        ----------
        condition
            Float tensor ``[B, 5, H, W]`` of concatenated
            ``rgb_hole + mask + depth_hole``.
        noise
            Float tensor ``[B, 4, H/8, W/8]`` of the actual initial
            noise. The first forward assumes ``H/8 == 64``, i.e.
            ``H == 512``.

        Returns
        -------
        torch.Tensor
            RGB output ``[B, 3, H, W]`` in ``[0, 1]`` (the network does
            not clamp; the inpaint helper is responsible for the
            final known-pixel composition and clamp).
        """
        if condition.shape[1] != self.in_channels:
            raise ValueError(
                f"expected condition with {self.in_channels} channels, got "
                f"{condition.shape[1]}"
            )
        expected_noise_hw = (condition.shape[2] // 8, condition.shape[3] // 8)
        if tuple(noise.shape[-2:]) != expected_noise_hw:
            raise ValueError(
                f"noise spatial shape {tuple(noise.shape[-2:])} does not "
                f"match condition's H/8 x W/8 = {expected_noise_hw}"
            )

        # ----- Encoder -----
        s1 = self.enc1(condition)              # 512→256
        s2 = self.enc2(s1)                      # 256→128
        s3 = self.enc3(s2)                      # 128→64
        # ----- Noise injection at 64² -----
        noise_feat = self.noise_proj(noise)
        s3 = s3 + noise_feat
        # ----- Bottleneck -----
        b = self.pre_bottleneck(s3)             # 64→32
        b = self.bottleneck(b)                  # 4× DW7 context
        # ----- Decoder -----
        d1 = self.dec1(b, s3)                   # 32→64 with skip s3
        d2 = self.dec2(d1, s2)                  # 64→128 with skip s2
        d3 = self.dec3(d2, s1)                  # 128→256 with skip s1
        # ----- Tail at 512² -----
        t = self.pre_tail(d3)                   # 256→512
        t = self.tail_proj(t)                   # 24→8
        rgb = self.tail(t)                      # 3x3 + 1x1 → RGB
        # Sigmoid is not applied inside the model; the inpaint helper
        # clamps to [0, 1] anyway. We keep the output linear so that
        # gradients flow naturally through the head.
        return rgb

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def predict_candidate(
        self, condition: ConditionBatch, noise: np.ndarray
    ) -> np.ndarray:
        """Run the network and return a numpy ``[B, 3, H, W]`` candidate."""
        validate_condition(condition)
        cond_t, noise_t = self._condition_to_torch(condition, noise)
        with torch.no_grad():
            out = self.forward(cond_t, noise_t)
        return out.detach().cpu().numpy().astype(np.float32, copy=False)

    def inpaint(
        self, condition: ConditionBatch, noise: np.ndarray
    ) -> np.ndarray:
        """Predict the candidate and compose it with the known pixels."""
        candidate = self.predict_candidate(condition, noise)
        return contracts_inpaint(condition, candidate)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _condition_to_torch(
        self, condition: ConditionBatch, noise: np.ndarray
    ) -> tuple[torch.Tensor, torch.Tensor]:
        rgb = condition.rgb_hole
        mask = condition.hole_mask
        depth = condition.depth_hole
        cond = np.concatenate([rgb, mask, depth], axis=1)  # [B, 5, H, W]
        cond_t = torch.from_numpy(np.ascontiguousarray(cond)).to(
            dtype=torch.float32, device=next(self.parameters()).device
        )
        noise_t = torch.from_numpy(np.ascontiguousarray(noise)).to(
            dtype=torch.float32, device=next(self.parameters()).device
        )
        return cond_t, noise_t
