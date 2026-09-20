"""Tests for the MoebiusSmallStudent (plan: moebius-small-distill, todos 1-4).

Pure-torch CPU tests. The full 64×64 forward is exercised once to pin the
deployment shape, the three feature taps, and the parameter budget.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from moebius_finetune.students.gated import GatedDW7Block
from moebius_finetune.students.moebius_small import (
    MoebiusSmallStudent,
    SinusoidalTimeEmbedding,
)


# ---------------------------------------------------------------------------
# GatedDW7Block (todo 1)
# ---------------------------------------------------------------------------


def test_gated_block_attribute_names_and_param_formula():
    """Plan todo 1: attributes are pinned (pw2 is an anchor target)."""
    block = GatedDW7Block(128)
    assert [name for name, _ in block.named_children()] == [
        "norm",
        "pw1",
        "dw",
        "pw2",
    ]
    c = 128
    assert sum(p.numel() for p in block.parameters()) == 3 * c * c + 51 * c
    assert sum(p.numel() for p in block.parameters()) == 55_680


def test_gated_block_shape_preserved_and_finite():
    torch.manual_seed(0)
    block = GatedDW7Block(32).eval()
    x = torch.randn(2, 32, 16, 16)
    with torch.no_grad():
        y = block(x)
    assert y.shape == x.shape
    assert torch.isfinite(y).all()


def test_gated_block_rejects_non_positive_channels():
    with pytest.raises(ValueError):
        GatedDW7Block(0)


# ---------------------------------------------------------------------------
# Timestep embedding (todo 2)
# ---------------------------------------------------------------------------


def test_timestep_embedding_has_no_lookup_table():
    """Plan todo 2/4: the discrete nn.Embedding schedule table is gone."""
    emb = SinusoidalTimeEmbedding()
    assert not any(isinstance(m, nn.Embedding) for m in emb.modules())


def test_timestep_embedding_is_deterministic_and_t_sensitive():
    torch.manual_seed(0)
    emb = SinusoidalTimeEmbedding().eval()
    t = torch.tensor([0.0, 500.0, 999.0])
    with torch.no_grad():
        a = emb(t)
        b = emb(t)
    assert a.shape == (3, 256)
    assert torch.allclose(a, b)
    assert not torch.allclose(a[0], a[2])


def test_timestep_embedding_accepts_integer_t():
    emb = SinusoidalTimeEmbedding().eval()
    with torch.no_grad():
        assert torch.allclose(emb(torch.tensor([500])), emb(torch.tensor([500.0])))


def test_timestep_embedding_rejects_non_1d():
    emb = SinusoidalTimeEmbedding()
    with pytest.raises(ValueError):
        emb(torch.zeros(2, 3))


# ---------------------------------------------------------------------------
# MoebiusSmallStudent (todo 3)
# ---------------------------------------------------------------------------


def test_default_config_is_locked():
    model = MoebiusSmallStudent()
    assert model.in_channels == 9
    assert model.channels == (128, 256, 512)
    assert model.blocks == (5, 5, 5)
    # attention only at 16²: last slot of enc2, first slot of dec2
    assert type(model.enc2[-1]).__name__ == "MobileMQA"
    assert type(model.enc2[0]).__name__ == "GatedDW7Block"
    assert type(model.dec2[0]).__name__ == "MobileMQA"
    assert type(model.dec2[-1]).__name__ == "GatedDW7Block"
    # 64²/32² stages are attention-free
    for stage in (model.enc0, model.enc1, model.dec0, model.dec1):
        assert all(type(m).__name__ == "GatedDW7Block" for m in stage)
    assert not any(isinstance(m, nn.Embedding) for m in model.modules())


def test_invalid_configs_rejected():
    with pytest.raises(ValueError):
        MoebiusSmallStudent(in_channels=0)
    with pytest.raises(ValueError):
        MoebiusSmallStudent(out_channels=-1)
    with pytest.raises(ValueError):
        MoebiusSmallStudent(channels=(128, 256))  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        MoebiusSmallStudent(blocks=(5, 5, 0))  # type: ignore[arg-type]


def test_forward_shapes_and_taps():
    torch.manual_seed(0)
    model = MoebiusSmallStudent().eval()
    x = torch.randn(2, 9, 64, 64)
    t = torch.tensor([3.0, 800.0])
    with torch.no_grad():
        out = model(x, t)
    assert out.sample.shape == (2, 4, 64, 64)
    assert torch.isfinite(out.sample).all()
    assert [tuple(b.shape) for b in out.block_outputs] == [
        (2, 128, 64, 64),
        (2, 256, 32, 32),
        (2, 512, 16, 16),
    ]
    assert all(torch.isfinite(b).all() for b in out.block_outputs)


def test_forward_rejects_wrong_channel_count():
    model = MoebiusSmallStudent()
    with pytest.raises(ValueError, match="9 input channels"):
        model(torch.randn(1, 11, 64, 64), torch.tensor([0.0]))


def test_forward_rejects_mismatched_t():
    model = MoebiusSmallStudent()
    with pytest.raises(ValueError):
        model(torch.randn(2, 9, 64, 64), torch.tensor([0.0]))
    with pytest.raises(ValueError):
        model(torch.randn(1, 9, 64, 64), torch.zeros(1, 1))


def test_param_budget_matches_plan():
    """Plan: ≈10.2 M measured; acceptance band [9.0 M, 12.5 M]."""
    model = MoebiusSmallStudent()
    params = sum(p.numel() for p in model.parameters())
    assert 9.0e6 < params < 12.5e6, f"param count {params} outside plan band"
    assert params == 10_158_695


def test_t_changes_output():
    torch.manual_seed(0)
    model = MoebiusSmallStudent().eval()
    x = torch.randn(1, 9, 16, 16)
    with torch.no_grad():
        y0 = model(x, torch.tensor([0.0])).sample
        y9 = model(x, torch.tensor([999.0])).sample
    assert not torch.allclose(y0, y9)


def test_batched_mixed_t_is_per_sample():
    torch.manual_seed(0)
    model = MoebiusSmallStudent().eval()
    x = torch.randn(1, 9, 16, 16).expand(2, -1, -1, -1).contiguous()
    with torch.no_grad():
        mixed = model(x, torch.tensor([0.0, 900.0])).sample
        same = model(x, torch.tensor([0.0, 0.0])).sample
    assert not torch.allclose(mixed[0], mixed[1])
    assert torch.allclose(same[0], same[1])


def test_anchor_parameters_are_leaf_convs():
    """Plan todo 7 anchors: enc2[-2].pw2 and head[-1] must be leaf weights."""
    model = MoebiusSmallStudent()
    feat_anchor = model.enc2[-2].pw2.weight
    out_anchor = model.head[-1].weight
    assert isinstance(feat_anchor, nn.Parameter) and isinstance(out_anchor, nn.Parameter)
    assert feat_anchor.requires_grad and out_anchor.requires_grad

    torch.manual_seed(0)
    x = torch.randn(2, 9, 16, 16)
    out = model(x, torch.tensor([1.0, 2.0]))
    loss = out.sample.pow(2).mean() + out.block_outputs[2].pow(2).mean()
    loss.backward()
    assert feat_anchor.grad is not None and torch.isfinite(feat_anchor.grad).all()
    assert out_anchor.grad is not None and torch.isfinite(out_anchor.grad).all()


def test_train_mode_batch_statistics_finite_and_grad_flows():
    torch.manual_seed(0)
    model = MoebiusSmallStudent().train()
    x = torch.randn(4, 9, 16, 16, requires_grad=True)
    out = model(x, torch.tensor([1.0, 2.0, 3.0, 4.0]))
    loss = out.sample.pow(2).mean()
    loss.backward()
    assert torch.isfinite(loss)
    assert x.grad is not None and torch.isfinite(x.grad).all()
    bn_weights = [m.weight for m in model.modules() if isinstance(m, nn.BatchNorm2d)]
    assert bn_weights
    grads = [p.grad for p in bn_weights if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)
