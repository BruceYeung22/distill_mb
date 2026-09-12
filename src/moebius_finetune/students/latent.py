"""LatentStudent-v0 — one-step latent-space inpainting network.

The architecture follows TDD §6.3 and is split into two independent
submodules:

* :class:`LightweightCodec` — a small autoencoder aligned to the
  Moebius 4-channel latent space. Encoder at 256²/128²/64² with widths
  ``[16, 32, 64]`` and block counts ``[1, 2, 2]``; decoder at
  64²/128²/256² with widths ``[64, 32, 16]`` and block counts
  ``[2, 2, 1]``. The output is a 3x3 RGB head.
* :class:`OneStepLatentNet` — the actual denoising student. It takes
  the 10-channel input ``noise(4) + masked_latent(4) + coverage(1) +
  depth_mean(1)``, processes it at 64²/32²/16² with widths
  ``[64, 128, 192]`` and block counts ``[2, 2, 4]`` (the 16² stage
  uses DW7 context blocks), and decodes back at 32²/64² with 2 blocks
  each. It predicts the **clean latent** in a single forward — no
  noise / v-prediction.

:class:`LatentStudentV0` wraps the two: ``predict_candidate`` runs the
latent net once, then decodes the result through the codec. Both
training (``training/codec/train.py``) and distillation
(``training/student/distill.py``) operate on the inner pieces.
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


__all__ = ["LatentStudentV0", "LightweightCodec", "OneStepLatentNet"]


# ---------------------------------------------------------------------------
# Codec defaults (TDD §6.3)
# ---------------------------------------------------------------------------

CODEC_ENC_WIDTHS = (16, 32, 64)
CODEC_ENC_BLOCKS = (1, 2, 2)
CODEC_DEC_WIDTHS = (64, 32, 16)
CODEC_DEC_BLOCKS = (2, 2, 1)
CODEC_LATENT_CHANNELS = 4
CODEC_RGB_CHANNELS = 3

# The Moebius VAE uses a scaling factor of 0.13025. Our codec targets
# the *scaled* latent space directly: encoder outputs are already
# multiplied by 0.13025, decoder inputs must be divided by 0.13025
# before decoding. We expose the constant so that training code can
# compose the right pipeline without hard-coding the value.
VAE_SCALING_FACTOR = 0.13025

# Stage widths / blocks for the one-step latent net (TDD §6.3).
STUDENT_IN_CHANNELS = 10  # noise(4) + masked_latent(4) + coverage(1) + depth_mean(1)
STUDENT_WIDTHS = (64, 128, 192)
STUDENT_BLOCKS = (2, 2, 4)


# ---------------------------------------------------------------------------
# Codec
# ---------------------------------------------------------------------------


class _CodecEncoderStage(nn.Module):
    def __init__(self, c_in: int, c_out: int, num_blocks: int) -> None:
        super().__init__()
        layers: list[nn.Module] = [DownsampleBlock(c_in, c_out)]
        for _ in range(num_blocks):
            layers.append(InvertedResidualBlock(c_out))
        self.blocks = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(x)


class _CodecDecoderStage(nn.Module):
    """A decoder stage: optional upsample + skip-add + N IR blocks.

    The TDD §6.3 layout places the IR blocks at each target
    resolution *after* the upsample, so this stage is::

        y = blocks(upsample(low) + skip)

    but ``upsample`` is only applied when ``do_upsample=True``.
    That lets the architecture stack two IR-only stages between
    two upsamples (e.g. dec1 at 64², dec2 at 128², dec3 at 256²).

    Note: ``c_skip`` does *not* have to equal ``c_out``. When they
    differ, a 1x1 projection of the low-res path to ``c_out`` is
    applied (before the skip add) so the add always sees matching
    channels. The skip tensor is assumed to already be at the
    target resolution and channel count.
    """

    def __init__(
        self,
        c_low: int,
        c_skip: int,
        c_out: int,
        num_blocks: int,
        *,
        do_upsample: bool,
    ) -> None:
        super().__init__()
        if c_skip != c_out:
            raise ValueError(
                f"decoder stage requires c_skip == c_out for the skip add; "
                f"got c_skip={c_skip}, c_out={c_out}"
            )
        self.do_upsample = bool(do_upsample)
        # 1x1 channel projection of the low-res path to c_out. Used
        # both when we upsample (post-upsample projection) and when
        # we don't (the low-res path may have a different channel
        # count than c_out).
        if c_low != c_out:
            self.proj_low = nn.Conv2d(c_low, c_out, kernel_size=1, bias=False)
        else:
            self.proj_low = nn.Identity()
        self.blocks = nn.Sequential(
            *[
                InvertedResidualBlock(c_out)
                for _ in range(num_blocks)
            ]
        )

    def forward(self, low: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        if self.do_upsample:
            up = nn.functional.interpolate(
                low, scale_factor=2.0, mode="nearest"
            )
            up = self.proj_low(up)
            if up.shape[-2:] != skip.shape[-2:]:
                up = nn.functional.interpolate(
                    up, size=skip.shape[-2:], mode="nearest"
                )
            x = up + skip
        else:
            # No upsample: low may have different channel count or
            # spatial size. We project it to match the skip.
            low = self.proj_low(low)
            if low.shape[-2:] != skip.shape[-2:]:
                low = nn.functional.interpolate(
                    low, size=skip.shape[-2:], mode="nearest"
                )
            x = low + skip
        return self.blocks(x)


class LightweightCodec(nn.Module):
    """4-channel latent codec (TDD §6.3).

    The codec mirrors the Moebius VAE layout: encoder downsamples the
    512² input by 8× to a 64² 4-channel latent, and the decoder
    reverses that.  The latent channels *include* the VAE scaling
    factor — the encoder produces a latent ``z`` that is directly
    comparable to the Moebius VAE posterior mode, so
    ``train_codec`` can supervise it with ``MSE(z, z_vae_posterior)``.
    """

    def __init__(
        self,
        encoder_widths: tuple[int, int, int] = CODEC_ENC_WIDTHS,
        encoder_blocks: tuple[int, int, int] = CODEC_ENC_BLOCKS,
        decoder_widths: tuple[int, int, int] = CODEC_DEC_WIDTHS,
        decoder_blocks: tuple[int, int, int] = CODEC_DEC_BLOCKS,
        latent_channels: int = CODEC_LATENT_CHANNELS,
        rgb_channels: int = CODEC_RGB_CHANNELS,
    ) -> None:
        super().__init__()
        if len(encoder_widths) != 3 or len(encoder_blocks) != 3:
            raise ValueError("encoder_widths / encoder_blocks must be length 3")
        if len(decoder_widths) != 3 or len(decoder_blocks) != 3:
            raise ValueError("decoder_widths / decoder_blocks must be length 3")
        self.encoder_widths = tuple(encoder_widths)
        self.encoder_blocks = tuple(encoder_blocks)
        self.decoder_widths = tuple(decoder_widths)
        self.decoder_blocks = tuple(decoder_blocks)
        self.latent_channels = int(latent_channels)
        self.rgb_channels = int(rgb_channels)

        # Encoder: 512² → 256² → 128² → 64².
        self.enc1 = _CodecEncoderStage(3, encoder_widths[0], encoder_blocks[0])
        self.enc2 = _CodecEncoderStage(
            encoder_widths[0], encoder_widths[1], encoder_blocks[1]
        )
        self.enc3 = _CodecEncoderStage(
            encoder_widths[1], encoder_widths[2], encoder_blocks[2]
        )
        # 1x1 projection to the 4-channel latent space.
        self.to_latent = nn.Conv2d(
            encoder_widths[2], latent_channels, kernel_size=1, bias=True
        )

        # Decoder: 64² → 128² → 256². The spec places the upsample to
        # 512² *after* a 1x1 projection to 8 channels (TDD §6.3:
        # "最后投影到 8 通道、上采样至 512²"). So the decoder has two
        # _CodecDecoderStage with do_upsample=True (dec1: 64→128,
        # dec2: 128→256) and a final stage at 256² with no upsample.
        self.from_latent = nn.Conv2d(
            latent_channels, decoder_widths[0], kernel_size=1, bias=True
        )
        self.dec1 = _CodecDecoderStage(
            c_low=decoder_widths[0],
            c_skip=decoder_widths[0],
            c_out=decoder_widths[0],
            num_blocks=decoder_blocks[0],
            do_upsample=True,
        )
        self.dec2 = _CodecDecoderStage(
            c_low=decoder_widths[0],
            c_skip=decoder_widths[1],
            c_out=decoder_widths[1],
            num_blocks=decoder_blocks[1],
            do_upsample=True,
        )
        self.dec3 = _CodecDecoderStage(
            c_low=decoder_widths[1],
            c_skip=decoder_widths[2],
            c_out=decoder_widths[2],
            num_blocks=decoder_blocks[2],
            do_upsample=False,
        )
        # Projection to 8 channels + upsample to 512² + 3x3 RGB head.
        self.to_8ch = nn.Conv2d(
            decoder_widths[2], 8, kernel_size=1, bias=True
        )
        self.pre_tail = nn.Upsample(scale_factor=2.0, mode="nearest")
        self.tail = TailRefineBlock(c_in=8, c_out=rgb_channels)

    def encode(self, rgb: torch.Tensor) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Encode an RGB image (in [0, 1]) into the 4-channel latent.

        The output is in the *Moebius-scaled* latent space, i.e. the
        encoder's output is meant to be compared directly to the
        Moebius VAE posterior mode (which already includes the
        0.13025 scaling factor).
        """
        s1 = self.enc1(rgb)
        s2 = self.enc2(s1)
        s3 = self.enc3(s2)
        return self.to_latent(s3), (s1, s2, s3)

    def decode_latent(
        self,
        latent: torch.Tensor,
        skips: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        """Decode a latent back to RGB. Skips are passed in for the decoder."""
        s1, s2, s3 = skips
        d0 = self.from_latent(latent)
        d1 = self.dec1(d0, s3)        # 64² + 64² skip
        d2 = self.dec2(d1, s2)        # 128² + 32² skip
        d3 = self.dec3(d2, s1)        # 256² + 16² skip (no upsample)
        d3 = self.to_8ch(d3)
        t = self.pre_tail(d3)
        return self.tail(t)

    def forward(self, rgb: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Full encode/decode round-trip on RGB → latent → RGB.

        Returns the latent and the reconstructed RGB. The two share
        the same skip connections, so the decode step does **not**
        re-encode.
        """
        latent, skips = self.encode(rgb)
        recon = self.decode_latent(latent, skips)
        return latent, recon


# ---------------------------------------------------------------------------
# One-step latent net
# ---------------------------------------------------------------------------


class _LatentNetStage(nn.Module):
    def __init__(
        self,
        c_in: int,
        c_out: int,
        num_blocks: int,
        *,
        context_kernel: int | None = None,
        downsample: bool = True,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        if downsample:
            layers.append(DownsampleBlock(c_in, c_out))
        elif c_in != c_out:
            # No downsample but the channel count changes; use a 1x1
            # conv to project (and a foldable BN+ReLU6 for parity).
            layers.append(
                nn.Sequential(
                    nn.Conv2d(c_in, c_out, kernel_size=1, bias=False),
                    nn.BatchNorm2d(c_out),
                    nn.ReLU6(inplace=False),
                )
            )
        if context_kernel is not None:
            layers.extend(
                ContextBlock(c_out, kernel=context_kernel)
                for _ in range(num_blocks)
            )
        else:
            layers.extend(
                InvertedResidualBlock(c_out) for _ in range(num_blocks)
            )
        self.blocks = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(x)


class _LatentNetDecoderStage(nn.Module):
    """Decoder stage for the one-step latent net: optional upsample + IR stack.

    Same semantics as :class:`_CodecDecoderStage` but kept as a
    separate type so the static budget walker can name it.
    """

    def __init__(
        self,
        c_low: int,
        c_skip: int,
        c_out: int,
        num_blocks: int,
        *,
        do_upsample: bool,
    ) -> None:
        super().__init__()
        if c_skip != c_out:
            raise ValueError(
                f"latent net decoder stage requires c_skip == c_out; "
                f"got c_skip={c_skip}, c_out={c_out}"
            )
        self.do_upsample = bool(do_upsample)
        if do_upsample and c_low != c_out:
            self.proj_low = nn.Conv2d(c_low, c_out, kernel_size=1, bias=False)
        else:
            self.proj_low = nn.Identity()
        self.blocks = nn.Sequential(
            *[
                InvertedResidualBlock(c_out)
                for _ in range(num_blocks)
            ]
        )

    def forward(self, low: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        if self.do_upsample:
            up = nn.functional.interpolate(
                low, scale_factor=2.0, mode="nearest"
            )
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


class OneStepLatentNet(nn.Module):
    """The one-step latent denoising network (TDD §6.3).

    Takes a 10-channel input ``[B, 10, 64, 64]`` and predicts the
    clean latent ``[B, 4, 64, 64]`` directly. The output is *not*
    a noise / v-prediction; the first version implements a single
    clean-latent forward.
    """

    def __init__(
        self,
        in_channels: int = STUDENT_IN_CHANNELS,
        widths: tuple[int, int, int] = STUDENT_WIDTHS,
        blocks: tuple[int, int, int] = STUDENT_BLOCKS,
        latent_channels: int = CODEC_LATENT_CHANNELS,
    ) -> None:
        super().__init__()
        if len(widths) != 3 or len(blocks) != 3:
            raise ValueError("widths / blocks must be length 3")
        self.in_channels = int(in_channels)
        self.widths = tuple(widths)
        self.blocks = tuple(blocks)
        self.latent_channels = int(latent_channels)

        # Encoder: 64² → 32² → 16². Two stride-2 downsamples, three
        # blocks worth of resolution.
        self.enc1 = _LatentNetStage(in_channels, widths[0], blocks[0])
        self.enc2 = _LatentNetStage(widths[0], widths[1], blocks[1])
        # The 16² stage is the bottleneck; no further downsample, but
        # the channel count may change. DW7 context blocks handle the
        # largest receptive field at this resolution.
        self.enc3 = _LatentNetStage(
            widths[1], widths[2], blocks[2], context_kernel=7, downsample=False
        )

        # Decoder: 16² → 32² → 64².
        # - dec1 up-samples 16→32 and uses the encoder's 32² feature
        #   (s1) as the skip — same convention as the PixelStudent.
        # - dec2 up-samples 32→64. There is no encoder skip at 64²
        #   (the input is 10 channels at 64², which is the *condition*
        #   not a feature map), so we project a zero-skip and rely on
        #   the upsample path to carry the signal.
        self.dec1 = _LatentNetDecoderStage(
            c_low=widths[2],
            c_skip=widths[0],
            c_out=widths[0],
            num_blocks=2,
            do_upsample=True,
        )
        # dec2 has no real skip; we feed it a tensor of the target
        # shape filled with zeros. The contract is that the static
        # budget walker can still see the upsample and the IR stack.
        self.dec2_upsample = nn.Upsample(scale_factor=2.0, mode="nearest")
        self.dec2_proj = nn.Conv2d(widths[0], widths[0], kernel_size=1, bias=True)
        self.dec2 = nn.Sequential(
            *[InvertedResidualBlock(widths[0]) for _ in range(2)]
        )
        # Final 1x1 projection to the 4-channel latent.
        self.to_latent = nn.Conv2d(widths[0], latent_channels, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] != self.in_channels:
            raise ValueError(
                f"expected {self.in_channels}-channel input, got {x.shape[1]}"
            )
        # Encoder: 64 → 32 → 16.
        s1 = self.enc1(x)
        s2 = self.enc2(s1)
        s3 = self.enc3(s2)
        # dec1: 16 → 32 with s1 (32²) skip.
        d1 = self.dec1(s3, s1)
        # dec2: 32 → 64 with no skip. We upsample + project + IR.
        d2 = self.dec2_upsample(d1)
        d2 = self.dec2_proj(d2)
        d2 = self.dec2(d2)
        return self.to_latent(d2)


# ---------------------------------------------------------------------------
# Combined student
# ---------------------------------------------------------------------------


class LatentStudentV0(nn.Module):
    """One-step latent student with a frozen-on-inference codec.

    At training time the codec is fine-tuned to match the Moebius
    VAE's posterior mode (see :mod:`training.codec.train`). The
    student net itself is trained against paired (noise, teacher
    final latent) examples. At inference, the codec is *frozen*
    and the network outputs the clean latent which is then decoded
    by the same codec into RGB.

    The contract: ``predict_candidate`` returns the **RGB** output
    (decoded), in the same ``[0, 1]`` range as the pixel student.
    """

    def __init__(
        self,
        codec: LightweightCodec | None = None,
        net: OneStepLatentNet | None = None,
    ) -> None:
        super().__init__()
        self.codec = codec if codec is not None else LightweightCodec()
        self.net = net if net is not None else OneStepLatentNet()

    # ------------------------------------------------------------------
    # Condition preparation
    # ------------------------------------------------------------------
    def build_student_input(
        self, condition: ConditionBatch
    ) -> np.ndarray:
        """Compute the 10-channel student input from a :class:`ConditionBatch`.

        The 10 channels are::

            [noise(4) | masked_latent(4) | coverage(1) | depth_mean(1)]

        The masked latent comes from encoding the hole-filled RGB
        image (hole → 0.5, which corresponds to the VAE's normalised
        zero fill — TDD §6.3). Coverage is the area-pooled known
        mask; depth_mean is the area-pooled known depth divided by
        coverage. Both are computed at the 8×-downsampled resolution.
        """
        validate_condition(condition)
        rgb = condition.rgb_hole
        mask = condition.hole_mask  # [B,1,H,W]
        depth = condition.depth_hole  # [B,1,H,W]
        noise = condition.noise  # [B,4,H/8,W/8]
        b, _, h, w = rgb.shape

        # Hole fill at 0.5 in [0, 1] for RGB and 0 for depth.
        rgb_filled = rgb * (1.0 - mask) + 0.5 * mask
        depth_filled = depth * (1.0 - mask)

        # Encode the (filled) image through the codec, but only at
        # inference we treat the codec as fixed. We compute the latent
        # by running a single forward (no grad, codec frozen).
        device = next(self.codec.parameters()).device
        rgb_t = torch.from_numpy(np.ascontiguousarray(rgb_filled)).to(
            dtype=torch.float32, device=device
        )
        with torch.no_grad():
            latent, _ = self.codec.encode(rgb_t)
            masked_latent = latent.cpu().numpy().astype(np.float32, copy=False)
        # masked_latent shape: [B, 4, H/8, W/8]
        if masked_latent.shape != noise.shape:
            raise ValueError(
                f"codec latent shape {masked_latent.shape} != noise shape "
                f"{noise.shape}; the codec must downsample by 8."
            )

        # coverage (1 - M) downsampled by area pooling.
        kernel = 8
        inv = 1.0 - mask
        bsz, _, hh, ww = inv.shape
        coverage = (
            inv.reshape(bsz, 1, hh // kernel, kernel, ww // kernel, kernel)
            .mean(axis=(3, 5))
        )  # [B,1,H/8,W/8]
        # depth_mean: average depth over the same window, divided by coverage
        depth_sum = (
            depth_filled.reshape(bsz, 1, hh // kernel, kernel, ww // kernel, kernel)
            .mean(axis=(3, 5))
            * (kernel * kernel)
        )
        eps = 1e-6
        depth_mean = depth_sum / np.maximum(coverage * (kernel * kernel), eps)
        depth_mean = (coverage > 0).astype(np.float32) * depth_mean

        # Concatenate to the 10-channel student input.
        student_input = np.concatenate(
            [
                noise,
                masked_latent,
                coverage.astype(np.float32),
                depth_mean.astype(np.float32),
            ],
            axis=1,
        )
        return student_input

    # ------------------------------------------------------------------
    # Forward / inpaint
    # ------------------------------------------------------------------
    def forward(
        self, condition: ConditionBatch, noise: np.ndarray
    ) -> np.ndarray:
        """Run the full student and return the RGB prediction (numpy).

        The ``noise`` argument overrides ``condition.noise`` so the
        caller can pin the initial noise without rebuilding the batch.
        """
        if noise is not None and noise is not condition.noise:
            condition = ConditionBatch(
                rgb_hole=condition.rgb_hole,
                hole_mask=condition.hole_mask,
                depth_hole=condition.depth_hole,
                noise=np.ascontiguousarray(noise).astype(np.float32, copy=False),
            )
        student_input = self.build_student_input(condition)
        device = next(self.parameters()).device
        s_t = torch.from_numpy(np.ascontiguousarray(student_input)).to(
            dtype=torch.float32, device=device
        )
        with torch.no_grad():
            latent = self.net(s_t)
            # Decode via the codec (no grad).
            rgb_filled = self._filled_rgb(condition)
            rgb_t = torch.from_numpy(np.ascontiguousarray(rgb_filled)).to(
                dtype=torch.float32, device=device
            )
            _, skips = self.codec.encode(rgb_t)
            recon = self.codec.decode_latent(latent, skips)
        return recon.cpu().numpy().astype(np.float32, copy=False)

    def predict_candidate(
        self, condition: ConditionBatch, noise: np.ndarray
    ) -> np.ndarray:
        """Run the student and return the candidate RGB in ``[B, 3, H, W]``."""
        return self.forward(condition, noise)

    def inpaint(
        self, condition: ConditionBatch, noise: np.ndarray
    ) -> np.ndarray:
        candidate = self.predict_candidate(condition, noise)
        return contracts_inpaint(condition, candidate)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _filled_rgb(condition: ConditionBatch) -> np.ndarray:
        rgb = condition.rgb_hole
        mask = condition.hole_mask
        return (rgb * (1.0 - mask) + 0.5 * mask).astype(np.float32, copy=False)
