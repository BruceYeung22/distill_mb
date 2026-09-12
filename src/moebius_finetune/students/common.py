"""Common building blocks for the two student models.

The blocks mirror the TDD §6.1 specification:

* :class:`InvertedResidualBlock` — 1x1 expansion to 2C → DW3x3 → 1x1
  projection back to C with a residual add. The first two convolutions
  are followed by foldable BatchNorm + ReLU6; the projection has no
  non-linearity.
* :class:`ContextBlock` — same skeleton but the depthwise kernel is
  7x7 (used in the bottleneck where the spatial resolution is the
  lowest).
* :class:`DownsampleBlock` — 3x3 stride-2 convolution + foldable BN +
  ReLU6 (no expansion).
* :class:`UpsampleBlock` — 1x1 channel projection at the low resolution
  to the target channel count, nearest-neighbour ×2, add the skip
  connection, then a single :class:`InvertedResidualBlock`.
* :class:`TailRefineBlock` — 3x3 + 1x1 head that produces the final
  RGB output at full resolution.

All convolutions that are followed by a BatchNorm use a *foldable*
BatchNorm — that is, a regular ``nn.BatchNorm2d`` whose statistics can
later be folded into the preceding convolution via :func:`fuse_bn`.
The foldability is what the budget tool relies on when it counts a
single conv × ReLU6 op instead of separate conv/BN/relu entries.

Everything is kept in pure ``torch`` so the modules can be moved to
either CPU or GPU and exported to ONNX unchanged.
"""

from __future__ import annotations

import torch
from torch import nn


__all__ = [
    "ContextBlock",
    "DownsampleBlock",
    "InvertedResidualBlock",
    "TailRefineBlock",
    "UpsampleBlock",
    "fuse_bn",
]


# ---------------------------------------------------------------------------
# Foldable BatchNorm helpers
# ---------------------------------------------------------------------------


def fuse_bn(conv: nn.Conv2d, bn: nn.BatchNorm2d) -> nn.Conv2d:
    """Fold a BatchNorm2d's statistics into the preceding Conv2d.

    Returns a fresh ``Conv2d`` with the equivalent weights and bias
    applied to ``conv``'s output. The original modules are not
    modified. This is a textbook implementation of the standard
    conv/BN fusion used by post-training quantisation toolchains.
    """
    if not isinstance(conv, nn.Conv2d):
        raise TypeError(f"fuse_bn expects Conv2d, got {type(conv).__name__}")
    if not isinstance(bn, nn.BatchNorm2d):
        raise TypeError(f"fuse_bn expects BatchNorm2d, got {type(bn).__name__}")
    if conv.training or bn.training:
        raise RuntimeError(
            "fuse_bn requires both modules to be in eval() mode so that the "
            "running statistics are the ones folded into the convolution."
        )
    if conv.weight is None:
        raise RuntimeError("fuse_bn: convolution has no weight tensor")

    fused = nn.Conv2d(
        in_channels=conv.in_channels,
        out_channels=conv.out_channels,
        kernel_size=conv.kernel_size,
        stride=conv.stride,
        padding=conv.padding,
        dilation=conv.dilation,
        groups=conv.groups,
        bias=True,
        padding_mode=conv.padding_mode,
        device=conv.weight.device,
        dtype=conv.weight.dtype,
    )

    with torch.no_grad():
        # scale = bn.weight / sqrt(bn.running_var + bn.eps)
        eps = bn.eps
        gamma = bn.weight if bn.affine else torch.ones_like(bn.running_mean)
        beta = bn.bias if bn.affine else torch.zeros_like(bn.running_mean)
        mean = bn.running_mean
        var = bn.running_var
        scale = gamma / torch.sqrt(var + eps)
        # Broadcast scale over (out_channels, 1, 1) to multiply the conv weight.
        fused_weight = conv.weight * scale.reshape(-1, 1, 1, 1)
        # The BN: y = (x - mean) * scale + beta
        #       = x * scale + (beta - mean * scale)
        # So the fused bias is:
        #   fused_bias = (conv.bias if any) * scale + (beta - mean * scale)
        conv_bias = conv.bias if conv.bias is not None else torch.zeros_like(beta)
        fused_bias = conv_bias * scale + (beta - mean * scale)
        fused.weight.copy_(fused_weight)
        fused.bias.copy_(fused_bias)

    fused.eval()
    return fused


# ---------------------------------------------------------------------------
# Inverted residual block
# ---------------------------------------------------------------------------


class _ConvBNActivation(nn.Module):
    """3x3 (or other kernel) convolution followed by foldable BN + ReLU6."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        groups: int = 1,
    ) -> None:
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU6(inplace=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))

    def fuse(self) -> "_ConvBNActivation":
        """Return a copy of self where conv+BN are fused into a single Conv2d."""
        fused = _ConvBNActivation(
            in_channels=self.conv.in_channels,
            out_channels=self.conv.out_channels,
            kernel_size=self.conv.kernel_size[0],
            stride=self.conv.stride[0],
            groups=self.conv.groups,
        )
        # Switch both to eval so that running stats are the ones folded.
        self.eval()
        fused.conv = fuse_bn(self.conv, self.bn)
        fused.bn = nn.Identity()  # placeholder, never executed
        return fused


class InvertedResidualBlock(nn.Module):
    """MobileNetV2-style inverted residual block (TDD §6.1).

    The block performs::

        y = x + project(act(dw(act(expand(x)))))

    with ``expand = Conv1x1(c, expansion*c) + BN + ReLU6``,
    ``dw = DWConv3x3(expansion*c, expansion*c) + BN + ReLU6``,
    ``project = Conv1x1(expansion*c, c)`` (no non-linearity). The
    residual add is only applied when ``in_channels == out_channels``
    and ``stride == 1``.
    """

    def __init__(self, channels: int, expansion: int = 2) -> None:
        super().__init__()
        if expansion < 1:
            raise ValueError(f"expansion must be >= 1, got {expansion}")
        expanded = channels * expansion
        self.expand = _ConvBNActivation(
            in_channels=channels, out_channels=expanded, kernel_size=1
        )
        self.dw = _ConvBNActivation(
            in_channels=expanded,
            out_channels=expanded,
            kernel_size=3,
            groups=expanded,
        )
        self.project = nn.Conv2d(
            in_channels=expanded,
            out_channels=channels,
            kernel_size=1,
            bias=False,
        )
        self._has_residual = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.expand(x)
        out = self.dw(out)
        out = self.project(out)
        if self._has_residual:
            out = out + x
        return out

    def fuse(self) -> "InvertedResidualBlock":
        """Return a copy of self with the BN layers folded into the convs."""
        fused = InvertedResidualBlock(
            channels=self.expand.conv.in_channels,
            expansion=self.expand.conv.out_channels // self.expand.conv.in_channels,
        )
        fused.expand = self.expand.fuse()
        fused.dw = self.dw.fuse()
        # The project conv has no BN, so just copy its weights.
        with torch.no_grad():
            fused.project.weight.copy_(self.project.weight)
        return fused


class ContextBlock(InvertedResidualBlock):
    """Inverted-residual block with a 7x7 depthwise kernel.

    Used at the lowest spatial resolution where the larger receptive
    field is cheap and helps aggregate the global context.
    """

    def __init__(self, channels: int, expansion: int = 2, kernel: int = 7) -> None:
        nn.Module.__init__(self)
        if expansion < 1:
            raise ValueError(f"expansion must be >= 1, got {expansion}")
        expanded = channels * expansion
        self.expand = _ConvBNActivation(
            in_channels=channels, out_channels=expanded, kernel_size=1
        )
        self.dw = _ConvBNActivation(
            in_channels=expanded,
            out_channels=expanded,
            kernel_size=kernel,
            groups=expanded,
        )
        self.project = nn.Conv2d(
            in_channels=expanded,
            out_channels=channels,
            kernel_size=1,
            bias=False,
        )
        self._has_residual = True


# ---------------------------------------------------------------------------
# Down / Up sampling
# ---------------------------------------------------------------------------


class DownsampleBlock(nn.Module):
    """3x3 stride-2 convolution + foldable BN + ReLU6 (TDD §6.1)."""

    def __init__(self, c_in: int, c_out: int) -> None:
        super().__init__()
        self.convbn = _ConvBNActivation(
            in_channels=c_in, out_channels=c_out, kernel_size=3, stride=2
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.convbn(x)

    def fuse(self) -> "DownsampleBlock":
        fused = DownsampleBlock(
            c_in=self.convbn.conv.in_channels,
            c_out=self.convbn.conv.out_channels,
        )
        fused.convbn = self.convbn.fuse()
        return fused


class UpsampleBlock(nn.Module):
    """1x1 channel projection → nearest ×2 → add skip → 1 inverted residual.

    The block takes a low-resolution tensor of ``c_low`` channels,
    projects it to ``c_out`` channels (so it can be added to the
    skip tensor), upsamples by 2, adds the skip tensor (which must
    already be at the high resolution and have ``c_skip ==
    c_out`` channels), and finally applies a single inverted
    residual block.
    """

    def __init__(self, c_low: int, c_skip: int, c_out: int) -> None:
        super().__init__()
        if c_skip != c_out:
            raise ValueError(
                f"UpsampleBlock expects c_skip == c_out so the skip can be "
                f"added without an extra projection; got c_skip={c_skip}, "
                f"c_out={c_out}."
            )
        # 1x1 channel projection of the low-res path to ``c_out``.
        if c_low != c_out:
            self.proj_low = nn.Conv2d(c_low, c_out, kernel_size=1, bias=False)
        else:
            self.proj_low = nn.Identity()
        self.ir = InvertedResidualBlock(c_out)

    def forward(self, low: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        # low: [B, c_low, H, W]; skip: [B, c_out, 2H, 2W]
        up = nn.functional.interpolate(low, scale_factor=2.0, mode="nearest")
        up = self.proj_low(up)
        if up.shape[-2:] != skip.shape[-2:]:
            up = nn.functional.interpolate(up, size=skip.shape[-2:], mode="nearest")
        return self.ir(up + skip)


class TailRefineBlock(nn.Module):
    """3x3 + 1x1 head at the full 512x512 resolution (TDD §6.1).

    Takes a small-channel feature map and produces the final 3-channel
    RGB output. Both convolutions are kept in the IR-friendly foldable
    form so the budget tool can fuse them with downstream ops.
    """

    def __init__(self, c_in: int = 8, c_out: int = 3) -> None:
        super().__init__()
        self.refine = _ConvBNActivation(
            in_channels=c_in, out_channels=c_in, kernel_size=3
        )
        self.proj = nn.Conv2d(c_in, c_out, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.refine(x)
        return self.proj(x)
