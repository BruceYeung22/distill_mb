"""Tests for the MobileMoebius 10-step diffusion student (TDD3 M1).

Spec: ``tdd/moebius-student-distill-2026-09-14.md`` §1 and §5 (M1).

All tests are pure torch, CPU-only, and use small spatial sizes where
possible to keep runtime low. The full-resolution (64×64) forward is
exercised once to pin the deployment shape and the parameter budget.
"""

from __future__ import annotations

import pytest
import torch

from moebius_finetune.students import MobileMoebius


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_default_config_matches_tdd3_s1_s2():
    """Defaults: in=11 (9 latent contract + 2 depth features), (32,128,512)."""
    model = MobileMoebius()
    assert model.in_channels == 11
    assert model.time_embed.num_steps == 10


def test_invalid_configs_rejected():
    with pytest.raises(ValueError):
        MobileMoebius(in_channels=0)
    with pytest.raises(ValueError):
        MobileMoebius(out_channels=-1)
    with pytest.raises(ValueError):
        MobileMoebius(channels=(32, 128))  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        MobileMoebius(blocks=(2, 3, 3, 3, 3, 0))


# ---------------------------------------------------------------------------
# Forward contract
# ---------------------------------------------------------------------------


def test_forward_shape_default_config_64x64():
    """Deployment shape: [B, 11, 64, 64] → [B, 4, 64, 64]."""
    torch.manual_seed(0)
    model = MobileMoebius().eval()
    x = torch.randn(2, 11, 64, 64)
    step = torch.tensor([3, 7], dtype=torch.long)
    with torch.no_grad():
        y = model(x, step)
    assert y.shape == (2, 4, 64, 64)
    assert torch.isfinite(y).all()


def test_forward_rejects_wrong_channel_count():
    model = MobileMoebius()
    x = torch.randn(1, 9, 64, 64)  # 9ch → default model wants 11
    with pytest.raises(ValueError, match="11 input channels"):
        model(x, torch.tensor([0]))


def test_param_budget_within_tdd3_range():
    """TDD3 §1 measured 8.662 M at in=9; 11ch adds a handful of params."""
    model = MobileMoebius()
    params = sum(p.numel() for p in model.parameters())
    assert 8.5e6 < params < 8.8e6, f"param count {params} outside TDD3 budget"


def test_legacy_256_config_still_constructible():
    """The pre-TDD3 (32,128,256) / 9ch prototype config must keep working."""
    model = MobileMoebius(in_channels=9, channels=(32, 128, 256)).eval()
    x = torch.randn(1, 9, 64, 64)
    with torch.no_grad():
        y = model(x, torch.tensor([3]))
    assert y.shape == (1, 4, 64, 64)
    assert torch.isfinite(y).all()


# ---------------------------------------------------------------------------
# Timestep conditioning
# ---------------------------------------------------------------------------


def test_step_id_out_of_range_raises():
    model = MobileMoebius()
    x = torch.randn(1, 11, 16, 16)
    with pytest.raises(ValueError, match="out of range"):
        model(x, torch.tensor([10]))  # num_steps=10 → valid 0..9
    with pytest.raises(ValueError, match="out of range"):
        model(x, torch.tensor([-1]))


def test_step_id_changes_output():
    """Different schedule indices must produce different epsilons."""
    torch.manual_seed(0)
    model = MobileMoebius().eval()
    x = torch.randn(1, 11, 16, 16)
    with torch.no_grad():
        y0 = model(x, torch.tensor([0]))
        y5 = model(x, torch.tensor([5]))
    assert not torch.allclose(y0, y5)


def test_batched_mixed_step_ids():
    """Per-sample step indices within one batch are honoured."""
    torch.manual_seed(0)
    model = MobileMoebius().eval()
    x = torch.randn(1, 11, 16, 16).expand(2, -1, -1, -1).contiguous()
    with torch.no_grad():
        mixed = model(x, torch.tensor([0, 5]))
        same = model(x, torch.tensor([0, 0]))
    assert not torch.allclose(mixed[0], mixed[1])
    assert torch.allclose(same[0], same[1])


# ---------------------------------------------------------------------------
# Training-mode sanity
# ---------------------------------------------------------------------------


def test_train_mode_batch_statistics_finite_and_grad_flows():
    """BN runs in train mode with batch > 1 (TDD3 S5); grads reach input."""
    torch.manual_seed(0)
    model = MobileMoebius().train()
    x = torch.randn(4, 11, 16, 16, requires_grad=True)
    y = model(x, torch.tensor([1, 2, 3, 4]))
    loss = y.pow(2).mean()
    loss.backward()
    assert torch.isfinite(loss)
    assert x.grad is not None and torch.isfinite(x.grad).all()
    # at least one BatchNorm in the model received a finite gradient
    # (BatchNorm2d layers live inside ConvBNAct Sequentials, so filter by
    # module type rather than by parameter name).
    from torch import nn as _nn

    bn_params = [
        m.weight for m in model.modules() if isinstance(m, _nn.BatchNorm2d)
    ]
    assert bn_params
    grads = [p.grad for p in bn_params if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)
