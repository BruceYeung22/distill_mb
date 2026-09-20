"""Tests for the multi-granularity distillation losses (plan todos 5-6).

The four loss terms, the training-only feature projections, and the
adaptive gradient-norm weighting are pure-torch. The E-LatentLPIPS term
additionally needs the ``elatentlpips`` package plus its pretrained
checkpoint, so those tests skip when either is unavailable.
"""

from __future__ import annotations

import math
import os

import pytest
import torch
import torch.nn.functional as F
from diffusers import DDPMScheduler

from moebius_finetune.students.moebius_small import MoebiusSmallStudent, StudentOutput
from moebius_finetune.training.student.multigranular import (
    ELATENTLPIPS_INPUT_SCALE,
    ELATENTLPIPS_LOSS_WEIGHT,
    ELATENTLPIPS_SDXL_FACTOR,
    KD_LOSS_WEIGHT,
    TAP_PAIRS,
    FeatureProjections,
    cal_adaptive_weights,
    cal_elatentlpips_loss,
    cal_kd_loss,
    cal_task_loss,
    total_loss,
)

TEACHER_SHAPES = {2: (1280, 16, 16), 3: (1280, 32, 32), 5: (320, 64, 64)}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _small_student() -> MoebiusSmallStudent:
    torch.manual_seed(0)
    return MoebiusSmallStudent(
        in_channels=9, out_channels=4, channels=(8, 16, 32), blocks=(2, 2, 2), time_dim=32
    )


def _student_out(student: MoebiusSmallStudent, batch: int = 2) -> StudentOutput:
    x = torch.randn(batch, 9, 64, 64)
    return student(x, torch.rand(batch) * 1000.0)


def _fake_teacher(shapes: dict[int, tuple[int, ...]], batch: int = 2) -> StudentOutput:
    """A frozen teacher carrying the *measured* Moebius tap shapes."""
    sample = torch.randn(batch, 4, 64, 64)
    n_taps = max(shapes) + 1
    taps = [
        torch.randn(batch, *shapes[i]) if i in shapes else torch.zeros(batch, 1, 1, 1)
        for i in range(n_taps)
    ]
    return StudentOutput(sample=sample, block_outputs=taps)


# ---------------------------------------------------------------------------
# Feature projections and the KD terms
# ---------------------------------------------------------------------------


def test_tap_pairs_match_measured_teacher_shapes():
    """Plan: enc0↔[5], enc1↔[3], enc2↔[2] (student channels 128/256/512)."""
    assert TAP_PAIRS == ((0, 5), (1, 3), (2, 2))


def test_feature_projections_param_count_and_shapes():
    projs = FeatureProjections()
    expected = 128 * 320 + 256 * 1280 + 512 * 1280
    assert sum(p.numel() for p in projs.parameters()) == expected
    feats = [
        torch.randn(1, 128, 64, 64),
        torch.randn(1, 256, 32, 32),
        torch.randn(1, 512, 16, 16),
    ]
    out = projs(feats)
    assert [tuple(t.shape) for t in out] == [
        (1, 320, 64, 64),
        (1, 1280, 32, 32),
        (1, 1280, 16, 16),
    ]
    with pytest.raises(ValueError):
        projs(feats[:2])
    with pytest.raises(ValueError):
        FeatureProjections((128, 256), (320, 1280, 1280))


def test_kd_losses_finite_differentiable_and_matched_to_measured_taps():
    student = _small_student()
    out = _student_out(student)
    teacher = _fake_teacher(TEACHER_SHAPES)
    projs = FeatureProjections((8, 16, 32), (320, 1280, 1280))

    featkd, outkd = cal_kd_loss(out, teacher, projs)
    assert torch.isfinite(featkd) and torch.isfinite(outkd)
    assert featkd.requires_grad and outkd.requires_grad
    (featkd + outkd).backward()
    assert student.head[-1].weight.grad is not None


def test_task_loss_is_plain_mse_against_noise():
    student = _small_student()
    out = _student_out(student)
    noise = torch.randn_like(out.sample)
    assert torch.allclose(cal_task_loss(out, noise), F.mse_loss(out.sample, noise))


def test_kd_loss_detaches_teacher():
    """Teacher outputs must never carry gradient (they are frozen)."""
    student = _small_student()
    out = _student_out(student)
    teacher = _fake_teacher(TEACHER_SHAPES)
    teacher.sample.requires_grad_(True)
    teacher.block_outputs[2].requires_grad_(True)
    projs = FeatureProjections((8, 16, 32), (320, 1280, 1280))
    featkd, outkd = cal_kd_loss(out, teacher, projs)
    (featkd + outkd).backward()
    assert teacher.sample.grad is None
    assert teacher.block_outputs[2].grad is None


# ---------------------------------------------------------------------------
# Adaptive gradient-norm weighting
# ---------------------------------------------------------------------------


def test_adaptive_weights_detached_finite_and_graph_survives():
    student = _small_student()
    out = _student_out(student)
    teacher = _fake_teacher(TEACHER_SHAPES)
    projs = FeatureProjections((8, 16, 32), (320, 1280, 1280))

    featkd, outkd = cal_kd_loss(out, teacher, projs)
    task = cal_task_loss(out, torch.randn_like(out.sample))
    # stand-in for the perceptual term: still a differentiable function of the student
    lpips_like = (out.sample - torch.randn_like(out.sample)).pow(2).mean()

    feat_w, out_w_outkd, out_w_lpips, diag = cal_adaptive_weights(
        featkd,
        task,
        outkd,
        lpips_like,
        feat_anchor=student.enc2[-2].pw2.weight,
        out_anchor=student.head[-1].weight,
    )
    assert isinstance(feat_w, torch.Tensor) and not feat_w.requires_grad
    assert torch.isfinite(feat_w) and float(feat_w) >= 0.0
    for weight in (out_w_outkd, out_w_lpips):
        assert isinstance(weight, torch.Tensor) and not weight.requires_grad
        assert torch.isfinite(weight) and float(weight) >= 0.0
    assert set(diag) >= {"feat_gnorm_featkd", "feat_gnorm_task", "out_gnorm_task"}
    assert all(math.isfinite(v) for v in diag.values())

    total = total_loss(
        featkd,
        task,
        outkd,
        lpips_like,
        feat_weight_task=feat_w,
        out_weight_outkd=out_w_outkd,
        out_weight_elatentlpips=out_w_lpips,
    )
    assert torch.isfinite(total)
    total.backward()
    assert student.head[-1].weight.grad is not None


def test_adaptive_weights_handle_disabled_terms():
    student = _small_student()
    out = _student_out(student)
    task = cal_task_loss(out, torch.randn_like(out.sample))
    feat_w, out_w_outkd, out_w_lpips, _ = cal_adaptive_weights(
        None,
        task,
        None,
        None,
        feat_anchor=student.enc2[-2].pw2.weight,
        out_anchor=student.head[-1].weight,
    )
    assert float(feat_w) == 1.0
    assert float(out_w_outkd) == 0.0 and float(out_w_lpips) == 0.0
    total = total_loss(
        None,
        task,
        None,
        None,
        feat_weight_task=feat_w,
        out_weight_outkd=out_w_outkd,
        out_weight_elatentlpips=out_w_lpips,
    )
    total.backward()
    assert torch.isfinite(total)


def test_total_loss_matches_recipe_combination():
    """``featkd*KD + feat_w * (task*0.5 + outkd*w*KD + lpips*w*0.5)``."""
    featkd = torch.tensor(2.0, requires_grad=True)
    task = torch.tensor(4.0, requires_grad=True)
    outkd = torch.tensor(8.0, requires_grad=True)
    lpips = torch.tensor(1.0, requires_grad=True)
    total = total_loss(
        featkd,
        task,
        outkd,
        lpips,
        feat_weight_task=torch.tensor(3.0),
        out_weight_outkd=torch.tensor(0.5),
        out_weight_elatentlpips=torch.tensor(0.25),
    )
    expected = (
        featkd * KD_LOSS_WEIGHT
        + 3.0
        * (
            task * 0.5
            + outkd * 0.5 * KD_LOSS_WEIGHT
            + lpips * 0.25 * ELATENTLPIPS_LOSS_WEIGHT
        )
    )
    assert torch.allclose(total, expected)


# ---------------------------------------------------------------------------
# E-LatentLPIPS term (todo 6) — skipped until package + weights are present
# ---------------------------------------------------------------------------


def _ckpt_dir() -> str:
    return os.environ.get(
        "ELATENTLPIPS_CKPT_DIR",
        "/home/dog/datasets/moebius_finetune/elatentlpips_ckpt",
    )


def _elatentlpips_available() -> bool:
    try:
        import elatentlpips  # noqa: F401
    except Exception:
        return False
    return os.path.exists(os.path.join(_ckpt_dir(), "sdxl_latest_vgg16_tuned.pth"))


requires_lpips = pytest.mark.skipif(
    not _elatentlpips_available(),
    reason="elatentlpips package or its pretrained checkpoint is unavailable",
)


def _load_lpips():
    """Load via the project's staging helper (mirror + trunk shim, CWD-independent)."""
    from moebius_finetune.training.student.elatentlpips_setup import load_elatentlpips

    return load_elatentlpips()


def _small_scheduler() -> DDPMScheduler:
    return DDPMScheduler(
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        num_train_timesteps=1000,
        clip_sample=False,
    )


@requires_lpips
def test_elatentlpips_term_runs_and_backprops():
    torch.manual_seed(0)
    student = _small_student()
    out = _student_out(student)
    scheduler = _small_scheduler()
    noise = torch.randn(2, 4, 64, 64)
    t = torch.tensor([500, 700])
    noisy = scheduler.add_noise(torch.randn(2, 4, 64, 64), noise, t)

    lpips = _load_lpips()
    loss = cal_elatentlpips_loss(out, noise, lpips, scheduler, t, noisy)
    assert torch.isfinite(loss)
    loss.backward()
    assert student.head[-1].weight.grad is not None


@requires_lpips
def test_elatentlpips_scale_is_in_the_calibrated_regime():
    """Todo 6 probe: the pinned input scaling must match the trunk's calibration.

    The trunk is ``CalibratedLatentVGG16BN`` — a VGG16-BN whose first
    conv was replaced by a 4-channel one and whose BatchNorm running
    statistics were fitted to the authors' training latents. Feeding the
    *calibrated* input scale therefore yields post-BN activations with
    roughly unit variance; a mis-scaled input does not.

    Measured latent scales on this box (one COCO image, SDXL VAE):

    ==============================  ======
    quantity                        std
    ==============================  ======
    SDXL VAE raw                    8.33
    SDXL pipeline (``* 0.13025``)   1.08
    TAESDXL (``scaling_factor=1``)  1.10
    ==============================  ======

    The library's ``normalize=True`` applies the ``0.13025`` factor
    itself, and its docstring says to pass **raw** encoder outputs — so
    the calibrated net-input scale is the pipeline scale (~1.08). Our
    latents already sit there, hence ``ELATENTLPIPS_INPUT_SCALE``.
    """
    torch.manual_seed(0)
    lpips = _load_lpips()
    slices = list(lpips.net.slice1)
    bn_index = next(
        i for i, module in enumerate(slices) if isinstance(module, torch.nn.BatchNorm2d)
    )
    assert bn_index > 0, "the trunk should carry a conv before its first BatchNorm"

    # unit-variance latents; the net input scale is what we vary
    z = torch.randn(4, 4, 64, 64)

    def post_bn_std(net_input_std: float) -> float:
        with torch.no_grad():
            h = z * net_input_std
            for module in slices[: bn_index + 1]:
                h = module(h)
            return float(h.std())

    # what the pinned policy feeds the net: our latent std (1.10) times the
    # pinned pre-scale. ``normalize=False`` is used precisely *because* the
    # pre-scale already lands us at pipeline scale.
    pinned_net_input = 1.10 * ELATENTLPIPS_INPUT_SCALE
    double_normalized = 1.10 * ELATENTLPIPS_SDXL_FACTOR

    pinned_std = post_bn_std(pinned_net_input)
    double_std = post_bn_std(double_normalized)

    # the pinned regime must be closer to unit variance than the
    # double-normalised alternative that the recipe's literal call produces
    assert abs(math.log(pinned_std)) < abs(math.log(double_std)), (
        f"pinned net-input std {pinned_std:.4f} is farther from the calibrated "
        f"regime than the double-normalised {double_std:.4f}"
    )
    assert 0.2 < pinned_std < 5.0, (
        f"pinned net-input scale produces post-BN std {pinned_std:.4f}, "
        f"which is not the trunk's calibrated regime"
    )
