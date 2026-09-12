"""Tests for :mod:`moebius_finetune.training.teacher.losses`."""

from __future__ import annotations

import pytest
import torch

from moebius_finetune.training.teacher.losses import (
    LossConfigError,
    combined_epsilon_loss,
    epsilon_mse,
)


def test_epsilon_mse_zero_when_perfect():
    pred = torch.zeros(2, 4, 8, 8)
    target = torch.zeros(2, 4, 8, 8)
    mask = torch.ones(2, 1, 8, 8)
    assert epsilon_mse(pred, target, mask).item() == 0.0


def test_epsilon_mse_broadcasts_mask():
    pred = torch.zeros(2, 4, 8, 8)
    target = torch.zeros(2, 4, 8, 8)
    mask = torch.ones(2, 1, 8, 8)  # broadcast over channels
    out = epsilon_mse(pred, target, mask)
    assert out.shape == ()


def test_epsilon_mse_empty_mask_returns_zero():
    pred = torch.ones(2, 4, 8, 8)
    target = torch.zeros(2, 4, 8, 8)
    mask = torch.zeros(2, 1, 8, 8)
    out = epsilon_mse(pred, target, mask)
    assert out.item() == 0.0
    # No gradient should be required.
    assert not out.requires_grad


def test_epsilon_mse_per_case_normalisation():
    """A case with half the hole pixels should have the same MSE per-pixel."""
    pred = torch.zeros(2, 4, 4, 4)
    target = torch.zeros(2, 4, 4, 4)
    # Both cases have a 4x4 hole, but the second one only has 8 hole pixels.
    mask = torch.zeros(2, 1, 4, 4)
    mask[0] = 1.0
    mask[1, 0, :2, :2] = 1.0  # 4 pixels
    pred[0] = 1.0  # (pred-target)^2 = 1 per pixel, average = 1
    pred[1, :, :2, :2] = 1.0
    out = epsilon_mse(pred, target, mask)
    # Both cases contribute 1.0 per pixel on their respective masks; the
    # per-case normaliser makes them comparable. The batch mean is 1.
    assert torch.allclose(out, torch.tensor(1.0), atol=1e-5)


def test_epsilon_mse_shape_mismatch_raises():
    pred = torch.zeros(2, 4, 4, 4)
    target = torch.zeros(2, 4, 4, 5)
    mask = torch.ones(2, 1, 4, 4)
    with pytest.raises(LossConfigError):
        epsilon_mse(pred, target, mask)


def test_combined_epsilon_loss_components():
    pred = torch.zeros(1, 4, 4, 4)
    target = torch.zeros(1, 4, 4, 4)
    hole_mask = torch.ones(1, 1, 4, 4) * 0.5
    pred[..., :2, :2] = 1.0  # contributes to hole
    out = combined_epsilon_loss(pred, target, hole_mask, known_weight=0.1)
    # Should be a finite positive scalar.
    assert out.shape == ()
    assert out.item() > 0.0


def test_combined_epsilon_loss_rejects_negative_weight():
    pred = torch.zeros(1, 4, 4, 4)
    target = torch.zeros(1, 4, 4, 4)
    mask = torch.zeros(1, 1, 4, 4)
    with pytest.raises(LossConfigError):
        combined_epsilon_loss(pred, target, mask, known_weight=-0.1)


def test_combined_epsilon_loss_zero_when_match():
    pred = torch.zeros(1, 4, 4, 4)
    target = pred
    mask = torch.ones(1, 1, 4, 4)
    out = combined_epsilon_loss(pred, target, mask)
    assert out.item() == 0.0


def test_losses_run_in_fp32():
    """Even if the inputs arrive as fp16 we reduce in fp32."""
    pred = torch.zeros(2, 4, 4, 4, dtype=torch.float16)
    target = torch.zeros(2, 4, 4, 4, dtype=torch.float16)
    mask = torch.ones(2, 1, 4, 4, dtype=torch.float16)
    out = epsilon_mse(pred, target, mask)
    assert out.dtype == torch.float32
