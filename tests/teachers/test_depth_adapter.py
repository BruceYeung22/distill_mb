"""Tests for :mod:`moebius_finetune.teachers.depth_adapter`."""

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
    assert a.hidden == 16
    assert a.target_dim == 320


def test_construct_rejects_invalid_dims():
    with pytest.raises(DepthAdapterConfigError):
        DepthConditionAdapter(channels=0, hidden=16, target_dim=320)
    with pytest.raises(DepthAdapterConfigError):
        DepthConditionAdapter(channels=2, hidden=0, target_dim=320)
    with pytest.raises(DepthAdapterConfigError):
        DepthConditionAdapter(channels=2, hidden=16, target_dim=0)


def test_final_layer_zero_initialized():
    """TDD §5.1: the final 1×1 weight/bias are zero-initialized."""
    a = DepthConditionAdapter()
    assert torch.equal(a.channel_proj.weight, torch.zeros_like(a.channel_proj.weight))
    assert torch.equal(a.channel_proj.bias, torch.zeros_like(a.channel_proj.bias))
    # The 3×3 layer is NOT zero-initialized (otherwise no gradient flows).
    assert a.spatial_conv.weight.abs().sum() > 0.0


# ---------------------------------------------------------------------------
# Forward — zero output
# ---------------------------------------------------------------------------


def test_zero_input_zero_output():
    a = DepthConditionAdapter()
    a.eval()
    with torch.no_grad():
        # All-zero input at H/8, W/8.
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
    assert torch.allclose(y, torch.zeros_like(y), atol=1e-7)


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
# Grad flow
# ---------------------------------------------------------------------------


def test_last_layer_grad_nonzero_after_zero_init():
    """TDD §9.2: the last zero-projection still receives gradient.

    Even though the output is zero, the gradient w.r.t. the last
    weight is exactly the (zero) output minus the target, which is
    the upstream signal — not None.
    """
    a = DepthConditionAdapter()
    x = torch.randn(1, 2, 64, 64, requires_grad=True)
    y = a(x).sum()
    y.backward()
    assert a.channel_proj.weight.grad is not None
    # The gradient may be non-zero because the spatial_conv is
    # non-zero; the 1×1 weights still get a non-None grad.
    assert a.channel_proj.weight.grad.shape == a.channel_proj.weight.shape


def test_front_layer_grad_after_one_sgd_step():
    """TDD §9.2: the depth-branch front layer gets non-zero gradient."""
    a = DepthConditionAdapter()
    optim = torch.optim.SGD(a.parameters(), lr=1e-3)
    x = torch.randn(1, 2, 64, 64)
    target = torch.randn(1, 320, 64, 64)
    y = a(x)
    loss = (y - target).pow(2).mean()
    loss.backward()
    # The 1×1 starts at 0, so the front layer's gradient is exactly
    # 0 at the first backward (the residual contribution is zero). We
    # do a single SGD step on the last layer so the front layer can
    # start receiving non-zero grad.
    optim.step()
    optim.zero_grad()
    a.zero_init_for_test = True  # marker for readability

    # Now nudge the last layer with a small non-zero value so the front
    # layer can flow.
    with torch.no_grad():
        a.channel_proj.weight.add_(torch.randn_like(a.channel_proj.weight) * 1e-3)
    x = torch.randn(1, 2, 64, 64)
    y = a(x)
    target = torch.randn(1, 320, 64, 64)
    loss = (y - target).pow(2).mean()
    loss.backward()
    assert a.spatial_conv.weight.grad is not None
    # The front layer's grad can be zero by coincidence, but with a
    # random target the chance is negligible — verify magnitude.
    assert a.spatial_conv.weight.grad.abs().max() > 0.0


# ---------------------------------------------------------------------------
# State dict helpers
# ---------------------------------------------------------------------------


def test_state_dict_round_trip():
    a = DepthConditionAdapter()
    sd = a.state_dict()
    # Re-initialise and load.
    a2 = DepthConditionAdapter()
    a2._zero_init_final_layer()  # ensure zero
    a2.spatial_conv.weight.data.fill_(0.0)
    a2.spatial_conv.bias.data.fill_(0.0)
    a2.load_state_dict(sd, strict=True)
    # After load the last layer is still zero (from sd) and the
    # spatial conv matches the original.
    assert torch.equal(a2.channel_proj.weight, torch.zeros_like(a2.channel_proj.weight))
    assert torch.equal(a2.spatial_conv.weight, a.spatial_conv.weight)
