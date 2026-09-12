"""Tests for the common student building blocks.

Covers the four building blocks specified in TDD §6.1
(InvertedResidualBlock, ContextBlock, DownsampleBlock, UpsampleBlock,
TailRefineBlock) and the BatchNorm folding utility (``fuse_bn``).

All tests are pure torch and run on CPU.
"""

from __future__ import annotations

import math

import pytest
import torch
from torch import nn

from moebius_finetune.students.common import (
    ContextBlock,
    DownsampleBlock,
    InvertedResidualBlock,
    TailRefineBlock,
    UpsampleBlock,
    fuse_bn,
)


# All tests below are CPU-only torch tests; they do not need a GPU.


# ---------------------------------------------------------------------------
# InvertedResidualBlock
# ---------------------------------------------------------------------------


def test_inverted_residual_block_channels_preserved():
    """The block keeps the channel count unchanged for ``stride=1``."""
    torch.manual_seed(0)
    block = InvertedResidualBlock(24, expansion=2)
    block.eval()
    x = torch.randn(2, 24, 16, 16)
    with torch.no_grad():
        y = block(x)
    assert y.shape == (2, 24, 16, 16)


def test_inverted_residual_block_residual_gradient_flows():
    """Gradient reaches the input through the residual add."""
    block = InvertedResidualBlock(8)
    x = torch.randn(1, 8, 4, 4, requires_grad=True)
    y = block(x)
    y.sum().backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    assert x.grad.abs().sum() > 0.0  # not all zero


def test_inverted_residual_block_expansion_changes_inner_channels():
    """expansion=2 makes the inner DW operate on 2*C channels."""
    block = InvertedResidualBlock(16, expansion=2)
    # expand conv: in=16, out=32
    assert block.expand.conv.in_channels == 16
    assert block.expand.conv.out_channels == 32
    # dw conv: in=32, out=32, groups=32 (depthwise)
    assert block.dw.conv.in_channels == 32
    assert block.dw.conv.out_channels == 32
    assert block.dw.conv.groups == 32
    # project conv: in=32, out=16
    assert block.project.in_channels == 32
    assert block.project.out_channels == 16


def test_inverted_residual_block_dw_kernel_3x3():
    """Default IR block uses a 3x3 depthwise kernel."""
    block = InvertedResidualBlock(8)
    assert block.dw.conv.kernel_size == (3, 3)


# ---------------------------------------------------------------------------
# ContextBlock
# ---------------------------------------------------------------------------


def test_context_block_uses_7x7_dw_kernel():
    """ContextBlock defaults to a 7x7 depthwise kernel."""
    block = ContextBlock(16)
    assert block.dw.conv.kernel_size == (7, 7)
    assert block.dw.conv.groups == 32  # 16 * expansion(2) = 32


def test_context_block_keeps_channel_count():
    block = ContextBlock(8, kernel=7)
    block.eval()
    x = torch.randn(1, 8, 8, 8)
    with torch.no_grad():
        y = block(x)
    assert y.shape == (1, 8, 8, 8)


# ---------------------------------------------------------------------------
# DownsampleBlock
# ---------------------------------------------------------------------------


def test_downsample_block_strides_by_2():
    """DownsampleBlock halves the spatial resolution."""
    block = DownsampleBlock(8, 16)
    block.eval()
    x = torch.randn(1, 8, 32, 32)
    with torch.no_grad():
        y = block(x)
    assert y.shape == (1, 16, 16, 16)


def test_downsample_block_channel_change():
    """The block projects from ``c_in`` to ``c_out``."""
    block = DownsampleBlock(8, 16)
    assert block.convbn.conv.in_channels == 8
    assert block.convbn.conv.out_channels == 16
    assert block.convbn.conv.stride == (2, 2)


# ---------------------------------------------------------------------------
# UpsampleBlock
# ---------------------------------------------------------------------------


def test_upsample_block_doubles_spatial():
    """The block's nearest upsample doubles the spatial resolution."""
    block = UpsampleBlock(c_low=16, c_skip=16, c_out=16)
    block.eval()
    low = torch.randn(1, 16, 16, 16)
    skip = torch.randn(1, 16, 32, 32)
    with torch.no_grad():
        y = block(low, skip)
    assert y.shape == (1, 16, 32, 32)


def test_upsample_block_channel_projection():
    """When ``c_low != c_out``, a 1x1 conv is used to project the low path."""
    block = UpsampleBlock(c_low=32, c_skip=16, c_out=16)
    assert isinstance(block.proj_low, nn.Conv2d)
    assert block.proj_low.in_channels == 32
    assert block.proj_low.out_channels == 16
    block.eval()
    low = torch.randn(1, 32, 8, 8)
    skip = torch.randn(1, 16, 16, 16)
    with torch.no_grad():
        y = block(low, skip)
    assert y.shape == (1, 16, 16, 16)


def test_upsample_block_skip_mismatch_raises():
    """``c_skip != c_out`` is rejected (we never silently mismatch)."""
    with pytest.raises(ValueError):
        UpsampleBlock(c_low=8, c_skip=8, c_out=16)


# ---------------------------------------------------------------------------
# TailRefineBlock
# ---------------------------------------------------------------------------


def test_tail_refine_block_default_8ch_input_rgb_output():
    """The default tail is 8 channels in, 3 channels out, at the same H, W."""
    block = TailRefineBlock(c_in=8, c_out=3)
    block.eval()
    x = torch.randn(1, 8, 32, 32)
    with torch.no_grad():
        y = block(x)
    assert y.shape == (1, 3, 32, 32)


def test_tail_refine_block_3x3_then_1x1():
    """The first conv is 3x3 (the refine), the second is 1x1 (the RGB)."""
    block = TailRefineBlock(c_in=8, c_out=3)
    assert block.refine.conv.kernel_size == (3, 3)
    assert block.proj.kernel_size == (1, 1)
    assert block.proj.out_channels == 3


# ---------------------------------------------------------------------------
# BN folding
# ---------------------------------------------------------------------------


def _make_conv_bn(c_in: int, c_out: int, k: int = 3) -> tuple[nn.Conv2d, nn.BatchNorm2d]:
    conv = nn.Conv2d(c_in, c_out, kernel_size=k, bias=True)
    bn = nn.BatchNorm2d(c_out)
    return conv, bn


def test_fuse_bn_returns_equivalent_convolution():
    """After folding, the original conv+BN and the fused conv agree in FP32."""
    torch.manual_seed(42)
    conv, bn = _make_conv_bn(8, 16, k=3)
    conv.eval()
    bn.eval()
    # Add a non-trivial BN to exercise the affine path.
    with torch.no_grad():
        bn.weight.copy_(torch.randn(16) * 0.5 + 1.0)
        bn.bias.copy_(torch.randn(16) * 0.1)
        bn.running_mean.copy_(torch.randn(16) * 0.05)
        bn.running_var.copy_(torch.rand(16) * 0.5 + 0.1)
    fused = fuse_bn(conv, bn)
    x = torch.randn(2, 8, 16, 16)
    with torch.no_grad():
        y_ref = bn(conv(x))
        y_fused = fused(x)
    assert torch.allclose(y_ref, y_fused, atol=1e-5)


def test_fuse_bn_rejects_training_mode():
    """Folding requires both modules in eval mode."""
    conv, bn = _make_conv_bn(4, 4)
    conv.train()
    bn.eval()
    with pytest.raises(RuntimeError):
        fuse_bn(conv, bn)


def test_fuse_bn_rejects_non_conv_input():
    bn = nn.BatchNorm2d(4).eval()
    with pytest.raises(TypeError):
        fuse_bn("not a conv", bn)  # type: ignore[arg-type]


def test_fuse_bn_preserves_no_bias_conv():
    """A conv with bias=False is fused correctly (we treat bias as 0)."""
    conv = nn.Conv2d(4, 8, kernel_size=1, bias=False).eval()
    bn = nn.BatchNorm2d(8).eval()
    fused = fuse_bn(conv, bn)
    assert fused.bias is not None  # fused always has a bias
    x = torch.randn(1, 4, 4, 4)
    with torch.no_grad():
        y_ref = bn(conv(x))
        y_fused = fused(x)
    assert torch.allclose(y_ref, y_fused, atol=1e-5)


# ---------------------------------------------------------------------------
# Block-level fusion helpers
# ---------------------------------------------------------------------------


def test_inverted_residual_block_fuse_produces_no_bn():
    """``InvertedResidualBlock.fuse()`` returns a block without BN modules."""
    block = InvertedResidualBlock(8)
    block.eval()
    fused = block.fuse()
    has_bn = any(
        isinstance(m, nn.BatchNorm2d) for m in fused.modules()
    )
    assert not has_bn


def test_inverted_residual_block_fuse_matches_original():
    """The fused block produces the same output as the original (FP32 atol=1e-5)."""
    torch.manual_seed(7)
    block = InvertedResidualBlock(8)
    block.eval()
    fused = block.fuse()
    x = torch.randn(2, 8, 16, 16)
    with torch.no_grad():
        y_ref = block(x)
        y_fused = fused(x)
    assert torch.allclose(y_ref, y_fused, atol=1e-5)
