"""Tests for the LatentStudent-v0 (TDD §6.3).

Covers:

* LightweightCodec encode/decode round-trip on RGB.
* OneStepLatentNet single forward shape.
* LatentStudentV0 full pipeline (build_student_input + net + decode).
* Save/load round-trip.
* Combined MACs (codec + net) within the 8 GMACs global budget.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from moebius_finetune.contracts import (
    ConditionBatch,
    validate_condition,
)
from moebius_finetune.data.synthetic import make_synthetic_batch
from moebius_finetune.deployment.budget import (
    compute_algorithm_macs,
    compute_combined_macs,
)
from moebius_finetune.students import (
    LatentStudentV0,
    LightweightCodec,
    OneStepLatentNet,
)




SMALL_H = 128  # H/8 = 16, so the net input is 16x16


def _batch(H: int = SMALL_H) -> ConditionBatch:
    return make_synthetic_batch(H=H, B=1, hole_spec="wide", seed=0)


# ---------------------------------------------------------------------------
# LightweightCodec
# ---------------------------------------------------------------------------


def test_codec_encode_decode_roundtrip_at_128():
    """Codec encodes RGB to a 4-channel latent and decodes it back to RGB."""
    codec = LightweightCodec()
    codec.eval()
    rgb = torch.randn(1, 3, 128, 128).clamp(0.0, 1.0)
    with torch.no_grad():
        latent, recon = codec(rgb)
    assert latent.shape == (1, 4, 16, 16)
    assert recon.shape == (1, 3, 128, 128)
    assert torch.isfinite(latent).all()
    assert torch.isfinite(recon).all()


def test_codec_encode_decode_reconstruction_error_at_128():
    """A freshly initialised codec reconstructs within a permissive L1 bound.

    We don't require a tight reconstruction (the codec is randomly
    initialised) — only that the L1 error is finite and not absurdly
    large.
    """
    torch.manual_seed(0)
    codec = LightweightCodec()
    codec.eval()
    rgb = torch.rand(1, 3, 128, 128)
    with torch.no_grad():
        _, recon = codec(rgb)
    l1 = (recon - rgb).abs().mean().item()
    assert 0.0 <= l1 < 1.0  # permissive upper bound


def test_codec_4_channel_latent_at_64_for_512_input():
    """At H=512, the latent is 4 channels at 64x64."""
    codec = LightweightCodec()
    codec.eval()
    rgb = torch.rand(1, 3, 512, 512)
    with torch.no_grad():
        latent, recon = codec(rgb)
    assert latent.shape == (1, 4, 64, 64)
    assert recon.shape == (1, 3, 512, 512)


# ---------------------------------------------------------------------------
# OneStepLatentNet
# ---------------------------------------------------------------------------


def test_one_step_latent_net_forward_shape():
    net = OneStepLatentNet()
    net.eval()
    x = torch.randn(1, 10, 64, 64)
    with torch.no_grad():
        y = net(x)
    assert y.shape == (1, 4, 64, 64)
    assert torch.isfinite(y).all()


def test_one_step_latent_net_backward():
    net = OneStepLatentNet()
    net.train()
    x = torch.randn(1, 10, 64, 64)
    y = net(x)
    target = torch.zeros_like(y)
    ((y - target) ** 2).mean().backward()
    for p in net.parameters():
        if p.requires_grad:
            assert p.grad is not None
            assert torch.isfinite(p.grad).all()


def test_one_step_latent_net_rejects_wrong_channels():
    net = OneStepLatentNet()
    net.eval()
    with pytest.raises(ValueError):
        net(torch.randn(1, 9, 64, 64))


# ---------------------------------------------------------------------------
# LatentStudentV0
# ---------------------------------------------------------------------------


def test_latent_student_build_student_input_shape():
    """``build_student_input`` produces a 10-channel tensor at H/8 x W/8."""
    model = LatentStudentV0()
    model.eval()
    batch = _batch(H=128)
    s = model.build_student_input(batch)
    assert s.shape == (1, 10, 16, 16)
    assert s.dtype == np.float32


def test_latent_student_predict_candidate_shape():
    model = LatentStudentV0()
    model.eval()
    batch = _batch(H=128)
    candidate = model.predict_candidate(batch, batch.noise)
    assert candidate.shape == batch.rgb_hole.shape
    assert candidate.dtype == np.float32
    assert np.isfinite(candidate).all()


def test_latent_student_inpaint_clamps_and_preserves_known():
    model = LatentStudentV0()
    model.eval()
    batch = make_synthetic_batch(H=128, B=1, hole_spec="wide", seed=1)
    out = model.inpaint(batch, batch.noise)
    assert out.min() >= 0.0
    assert out.max() <= 1.0
    # Known region (where mask == 0) must match rgb_hole exactly.
    mask = batch.hole_mask
    known = mask == 0
    np.testing.assert_array_equal(
        out[np.broadcast_to(known, out.shape)],
        batch.rgb_hole[np.broadcast_to(known, batch.rgb_hole.shape)],
    )


def test_latent_student_save_load_roundtrip(tmp_path: Path):
    model = LatentStudentV0()
    model.eval()
    batch = _batch(H=128)
    before = model.predict_candidate(batch, batch.noise)
    ckpt = tmp_path / "latent.pt"
    torch.save(model.state_dict(), ckpt)
    model2 = LatentStudentV0()
    model2.load_state_dict(torch.load(ckpt, map_location="cpu"))
    model2.eval()
    after = model2.predict_candidate(batch, batch.noise)
    np.testing.assert_allclose(before, after, atol=1e-6)


def test_latent_student_combined_macs_within_budget():
    """Combined codec + net MACs must fit the 8 GMACs global budget.

    The spec's 2.05 GMACs hand-roll is an *approximation* (TDD §6.3
    notes: "参数数字未包含所有 bias/BN，读写未包含全部实际布局与
    转换开销，最终以程序化核算为准"). Our programmatic count is
    the ground truth; we still confirm it sits within a reasonable
    band around the spec's hand-roll so the test acts as a regression
    check, but the upper bound is generous enough to account for the
    bias/BN/skips the hand-roll missed.
    """
    model = LatentStudentV0()
    combined = compute_combined_macs(
        [
            (model.codec, (1, 3, 512, 512)),
            (model.net, (1, 10, 64, 64)),
        ]
    )
    gmacs = combined["total_macs_g"]
    assert gmacs <= 8.0, f"latent {gmacs:.3f} GMACs > 8 GMACs budget"
    # Spec hand-roll 2.05 GMACs. Allow a ±30% band — the spec is
    # explicitly an approximation and the actual count includes the
    # bias/BN/skips the hand-roll did not.
    assert 1.45 <= gmacs <= 2.70, (
        f"latent combined {gmacs:.3f} GMACs is outside the ±30% band "
        f"of the spec's 2.05 GMACs hand-roll."
    )
