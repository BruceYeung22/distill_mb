"""Tests for :mod:`moebius_finetune.teachers.depth_adapter` (TDD2 §1)."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from moebius_finetune.teachers.depth_adapter import (
    DepthAdapterConfigError,
    DepthConditionAdapter,
)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_default_construct():
    a = DepthConditionAdapter()
    assert a.channels == 2
    assert a.hidden == 64
    assert a.target_dim == 320


def test_construct_rejects_invalid_dims():
    with pytest.raises(DepthAdapterConfigError):
        DepthConditionAdapter(channels=0, hidden=64, target_dim=320)
    with pytest.raises(DepthAdapterConfigError):
        DepthConditionAdapter(channels=2, hidden=0, target_dim=320)
    with pytest.raises(DepthAdapterConfigError):
        DepthConditionAdapter(channels=2, hidden=64, target_dim=0)


def test_v2_block_structure():
    """TDD2 §1: pw 1×1 → ReLU → dw 3×3 (groups=hidden) + BN → ReLU → pw 1×1."""
    a = DepthConditionAdapter()
    assert isinstance(a.pw_in, nn.Conv2d)
    assert a.pw_in.kernel_size == (1, 1)
    assert a.pw_in.in_channels == 2
    assert a.pw_in.out_channels == 64
    assert a.pw_in.bias is not None

    assert isinstance(a.dw_conv, nn.Conv2d)
    assert a.dw_conv.kernel_size == (3, 3)
    assert a.dw_conv.groups == 64
    assert a.dw_conv.in_channels == 64
    assert a.dw_conv.out_channels == 64
    assert a.dw_conv.bias is None

    assert isinstance(a.bn, nn.BatchNorm2d)
    assert a.bn.num_features == 64
    assert a.bn.affine is True
    assert a.bn.track_running_stats is True

    assert isinstance(a.channel_proj, nn.Conv2d)
    assert a.channel_proj.kernel_size == (1, 1)
    assert a.channel_proj.in_channels == 64
    assert a.channel_proj.out_channels == 320


# ---------------------------------------------------------------------------
# Zero initialization
# ---------------------------------------------------------------------------


def test_final_layer_zero_initialized():
    """TDD2 §1: the final 1×1 weight/bias are zero-initialized."""
    a = DepthConditionAdapter()
    assert torch.equal(a.channel_proj.weight, torch.zeros_like(a.channel_proj.weight))
    assert torch.equal(a.channel_proj.bias, torch.zeros_like(a.channel_proj.bias))
    # The expansion layer is NOT zero-initialized (otherwise no gradient
    # would flow into the projection).
    assert a.pw_in.weight.abs().sum() > 0.0


# ---------------------------------------------------------------------------
# Forward — zero output
# ---------------------------------------------------------------------------


def test_zero_input_zero_output():
    a = DepthConditionAdapter()
    a.eval()
    with torch.no_grad():
        x = torch.zeros(1, 2, 64, 64)
        y = a(x)
    # FP32 atol = 1e-7 (TDD §9.2)
    assert torch.allclose(y, torch.zeros_like(y), atol=1e-7)


def test_zero_input_zero_output_custom_shape():
    a = DepthConditionAdapter(channels=2, hidden=8, target_dim=320)
    a.eval()
    with torch.no_grad():
        x = torch.zeros(2, 2, 32, 24)
        y = a(x)
    assert y.shape == (2, 320, 32, 24)
    assert torch.allclose(y, torch.zeros_like(y), atol=1e-7)


def test_output_shape_matches_input_spatial():
    a = DepthConditionAdapter()
    a.eval()
    with torch.no_grad():
        x = torch.randn(3, 2, 64, 64)
        y = a(x)
    assert y.shape == (3, 320, 64, 64)


# ---------------------------------------------------------------------------
# Forward — disabled branch returns zeros
# ---------------------------------------------------------------------------


def test_disable_returns_zero():
    a = DepthConditionAdapter()
    a.disable()
    with torch.no_grad():
        x = torch.randn(1, 2, 64, 64)
        y = a(x)
    assert torch.allclose(y, torch.zeros_like(y), atol=1e-7)


# ---------------------------------------------------------------------------
# BN conventions
# ---------------------------------------------------------------------------


def test_bn_train_mode_updates_running_stats():
    """TDD2 §5: adapter BN accumulates running stats in train mode only."""
    a = DepthConditionAdapter()
    a.train()
    assert a.bn.num_batches_tracked.item() == 0
    x = torch.randn(4, 2, 64, 64)
    with torch.no_grad():
        a(x)
    assert a.bn.num_batches_tracked.item() == 1
    assert (
        a.bn.running_mean.abs().sum() > 0.0 or a.bn.running_var.abs().sum() > 0.0
    )

    tracked = a.bn.num_batches_tracked.item()
    mean_snapshot = a.bn.running_mean.clone()
    a.eval()
    with torch.no_grad():
        a(torch.randn(4, 2, 64, 64))
    assert a.bn.num_batches_tracked.item() == tracked
    assert torch.equal(a.bn.running_mean, mean_snapshot)


def test_eval_uses_running_stats():
    """In eval mode the BN must be deterministic (running stats)."""
    a = DepthConditionAdapter()
    a.train()
    with torch.no_grad():
        a(torch.randn(8, 2, 64, 64))
    a.eval()
    x1 = torch.randn(1, 2, 64, 64)
    with torch.no_grad():
        y1 = a(x1)
        y2 = a(x1)
    assert torch.equal(y1, y2)


# ---------------------------------------------------------------------------
# Grad flow
# ---------------------------------------------------------------------------


def test_projection_grad_nonzero_after_zero_init():
    """The zero-initialized pw2 still receives a (non-None) gradient."""
    a = DepthConditionAdapter()
    x = torch.randn(1, 2, 64, 64, requires_grad=True)
    y = a(x).sum()
    y.backward()
    assert a.channel_proj.weight.grad is not None
    assert a.channel_proj.weight.grad.shape == a.channel_proj.weight.shape


def test_front_layers_zero_grad_until_projection_moves():
    """TDD2 §1: with pw2 at zero, front-layer grads are exactly 0 on the
    first backward; after one optimizer step they become non-zero."""
    a = DepthConditionAdapter()
    x = torch.randn(1, 2, 64, 64)
    target = torch.randn(1, 320, 64, 64)
    loss = (a(x) - target).pow(2).mean()
    loss.backward()
    # Front layers (pw_in / dw / BN affine) sit behind the zero pw2.
    assert a.pw_in.weight.grad is not None
    assert float(a.pw_in.weight.grad.abs().max()) == 0.0
    assert float(a.bn.weight.grad.abs().max()) == 0.0

    optim = torch.optim.SGD(a.parameters(), lr=1e-3)
    optim.step()
    optim.zero_grad(set_to_none=True)

    x = torch.randn(1, 2, 64, 64)
    target = torch.randn(1, 320, 64, 64)
    loss = (a(x) - target).pow(2).mean()
    loss.backward()
    assert a.pw_in.weight.grad is not None
    assert a.pw_in.weight.grad.abs().max() > 0.0
    assert a.dw_conv.weight.grad.abs().max() > 0.0


# ---------------------------------------------------------------------------
# State dict helpers
# ---------------------------------------------------------------------------


def test_state_dict_round_trip():
    a = DepthConditionAdapter()
    sd = a.state_dict()
    a2 = DepthConditionAdapter()
    a2._zero_init_final_layer()
    a2.pw_in.weight.data.fill_(0.0)
    a2.pw_in.bias.data.fill_(0.0)
    a2.load_state_dict(sd, strict=True)
    # After load the last layer is still zero (from sd) and the rest
    # match the original.
    assert torch.equal(a2.channel_proj.weight, torch.zeros_like(a2.channel_proj.weight))
    assert torch.equal(a2.pw_in.weight, a.pw_in.weight)
    assert torch.equal(a2.dw_conv.weight, a.dw_conv.weight)
    assert torch.equal(a2.bn.running_mean, a.bn.running_mean)
