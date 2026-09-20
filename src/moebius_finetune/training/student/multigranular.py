"""Multi-granularity distillation losses (plan: moebius-small-distill, todo 5).

Faithful port of ``Moebius/train_distillation.py:94-194`` — the four-term
objective and the adaptive gradient-norm weighting — adapted to the
:class:`~moebius_finetune.students.moebius_small.MoebiusSmallStudent` tap
layout.

Terms (all computed against the frozen teacher):

``loss_featkd``
    Per-pair MSE between the student's feature taps (projected to the
    teacher's channel width by :class:`FeatureProjections`) and the
    teacher's ``block_outputs``. Pairs are fixed by **measured** tap
    shapes, not by index symmetry — the teacher records each tap *after*
    that block's own resampling, so there is no 64² encoder tap and no
    16² decoder tap on the teacher side:

    ==============  ====================  =========================
    student tap     teacher tap           teacher shape
    ==============  ====================  =========================
    ``enc0`` (0)    ``block_outputs[5]``  320 ch @ 64²
    ``enc1`` (1)    ``block_outputs[3]``  1280 ch @ 32²
    ``enc2`` (2)    ``block_outputs[2]``  1280 ch @ 16²
    ==============  ====================  =========================

``loss_outkd``
    MSE between student and teacher epsilon predictions.

``loss_task``
    MSE between the student's epsilon and the sampled noise (the
    ground-truth training target).

``loss_elatentlpips``
    E-LatentLPIPS distance between the x0 estimates decoded from the
    student's and the target's epsilon predictions (the
    ``scheduler.step(...).pred_original_sample`` construction of the
    original recipe).

Weighting follows ``cal_adaptive_weights_type8``: the gradient norm of
each auxiliary term at two anchor parameters is compared against the
gradient norm of the task term, producing per-term scaling factors that
are re-estimated every step. The anchors are ``enc2[-2].pw2.weight``
(feature-anchor) and ``head[-1].weight`` (output-anchor) — any leaf
parameter in the shared trunk works; these are the closest analogues of
the original's ``down_blocks[2].attentions[1].proj_out.weight`` and
``conv_out.conv_pw.weight``.

Constant weights match ``config/train_demo.sh``: ``feat_loss_weight=1.0``,
``KD_loss_weight=0.01``, ``task_loss_weight=0.5``,
``elatentlpips_loss_weight=0.5``.
"""

from __future__ import annotations

from typing import Iterable, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from moebius_finetune.students.moebius_small import StudentOutput

__all__ = [
    "ELATENTLPIPS_INPUT_SCALE",
    "ELATENTLPIPS_LOSS_WEIGHT",
    "ELATENTLPIPS_SDXL_FACTOR",
    "FEAT_LOSS_WEIGHT",
    "KD_LOSS_WEIGHT",
    "TAP_PAIRS",
    "TASK_LOSS_WEIGHT",
    "FeatureProjections",
    "cal_adaptive_weights",
    "cal_elatentlpips_loss",
    "cal_kd_loss",
    "cal_task_loss",
    "total_loss",
]


FEAT_LOSS_WEIGHT = 1.0
KD_LOSS_WEIGHT = 0.01
TASK_LOSS_WEIGHT = 0.5
ELATENTLPIPS_LOSS_WEIGHT = 0.5

#: SDXL VAE pipeline scaling factor (`raw_latent * 0.13025` = UNet input).
ELATENTLPIPS_SDXL_FACTOR = 0.13025

#: Pre-scale applied to our latents before the E-LatentLPIPS call
#: (``normalize=False`` is passed for the same reason).
#:
#: The library's ``normalize=True`` multiplies by :data:`ELATENTLPIPS_SDXL_FACTOR`
#: internally and its docstring asks for *raw* encoder outputs. Our TAESDXL
#: latents already live at the SDXL pipeline scale (std ≈ 1.10, versus SDXL
#: raw 8.33 → pipeline 1.08), so the literal recipe call would normalise them
#: twice and drive the calibrated trunk 7.7x below its trained regime.
#: Verified by ``test_elatentlpips_scale_is_in_the_calibrated_regime``.
ELATENTLPIPS_INPUT_SCALE = 1.0

#: ``(student_tap_index, teacher_block_output_index)`` — measured shapes.
TAP_PAIRS: tuple[tuple[int, int], ...] = ((0, 5), (1, 3), (2, 2))


class FeatureProjections(nn.Module):
    """Training-only 1×1 projections from student taps to teacher channels.

    Discarded at deployment (the deployed student never runs the teacher
    or the projections).
    """

    def __init__(
        self,
        student_channels: Sequence[int] = (128, 256, 512),
        teacher_channels: Sequence[int] = (320, 1280, 1280),
    ) -> None:
        super().__init__()
        if len(student_channels) != len(teacher_channels):
            raise ValueError(
                f"student_channels and teacher_channels must have the same length, "
                f"got {len(student_channels)} and {len(teacher_channels)}"
            )
        if not student_channels:
            raise ValueError("at least one tap pair is required")
        self.projs = nn.ModuleList(
            [
                nn.Conv2d(int(s), int(t), 1, bias=False)
                for s, t in zip(student_channels, teacher_channels)
            ]
        )

    def forward(self, features: Sequence[torch.Tensor]) -> list[torch.Tensor]:
        if len(features) != len(self.projs):
            raise ValueError(
                f"expected {len(self.projs)} feature taps, got {len(features)}"
            )
        return [proj(feat) for proj, feat in zip(self.projs, features)]


def cal_kd_loss(
    pred_S: StudentOutput,
    pred_T: StudentOutput,
    projections: FeatureProjections,
    *,
    tap_pairs: Iterable[tuple[int, int]] = TAP_PAIRS,
    feat_loss_weight: float = FEAT_LOSS_WEIGHT,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Feature KD (all pairs) and output KD, matching ``cal_KD_loss``."""
    pairs = tuple(tap_pairs)
    projected = projections(pred_S.block_outputs)
    feat_losses = [
        F.mse_loss(projected[s_idx], pred_T.block_outputs[t_idx].detach(), reduction="mean")
        for s_idx, t_idx in pairs
    ]
    loss_featkd = sum(feat_losses) * feat_loss_weight
    loss_outkd = F.mse_loss(
        pred_S.sample.float(), pred_T.sample.float().detach(), reduction="mean"
    )
    return loss_featkd, loss_outkd


def cal_task_loss(pred_S: StudentOutput, target: torch.Tensor) -> torch.Tensor:
    """Ground-truth noise regression, matching ``cal_task_loss``."""
    return F.mse_loss(pred_S.sample.float(), target.float(), reduction="mean")


def cal_elatentlpips_loss(
    pred_S: StudentOutput,
    target: torch.Tensor,
    elatentlpips_model: nn.Module,
    noise_scheduler: object,
    timesteps: torch.Tensor,
    noisy_latents: torch.Tensor,
) -> torch.Tensor:
    """Perceptual distance between the two x0 estimates.

    Both the student's and the target's epsilon are turned into x0 with
    the *same* ``scheduler.step(...).pred_original_sample`` used by the
    original recipe (``train_distillation.py:150-158``), then compared by
    E-LatentLPIPS with ``normalize=True`` (the library scales raw latents
    by the SDXL VAE factor internally) and ``ensembling=True``.
    """
    x0_pred = torch.stack(
        [
            noise_scheduler.step(n, t, noisy_latent).pred_original_sample
            for (n, t, noisy_latent) in zip(pred_S.sample, timesteps, noisy_latents)
        ]
    )
    x0_target = torch.stack(
        [
            noise_scheduler.step(tgt, t, noisy_latent).pred_original_sample
            for (tgt, t, noisy_latent) in zip(target.float(), timesteps, noisy_latents)
        ]
    )
    return elatentlpips_model(
        x0_pred * ELATENTLPIPS_INPUT_SCALE,
        x0_target * ELATENTLPIPS_INPUT_SCALE,
        normalize=False,
        ensembling=True,
    ).mean()


def _grad_norm(loss: Optional[torch.Tensor], param: torch.Tensor) -> Optional[torch.Tensor]:
    if loss is None:
        return None
    grad = torch.autograd.grad(loss, param, retain_graph=True)[0]
    return torch.norm(grad)


def cal_adaptive_weights(
    featkd_loss: Optional[torch.Tensor],
    task_loss: torch.Tensor,
    outkd_loss: Optional[torch.Tensor],
    elatentlpips_loss: Optional[torch.Tensor],
    *,
    feat_anchor: torch.Tensor,
    out_anchor: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, float]]:
    """Gradient-norm balancing, matching ``cal_adaptive_weights_type8``.

    Returns ``(feat_weight_task, out_weight_outkd, out_weight_elatentlpips,
    diagnostics)`` as detached 0-dim tensors. Disabled terms (``None``)
    yield a zero weight and are excluded from the graph walks.
    """
    feat_grads = {
        "featkd": _grad_norm(featkd_loss, feat_anchor),
        "task": _grad_norm(task_loss, feat_anchor),
        "outkd": _grad_norm(outkd_loss, feat_anchor),
        "elatentlpips": _grad_norm(elatentlpips_loss, feat_anchor),
    }
    out_grads = {
        "task": _grad_norm(task_loss, out_anchor),
        "outkd": _grad_norm(outkd_loss, out_anchor),
        "elatentlpips": _grad_norm(elatentlpips_loss, out_anchor),
    }

    task_feat_norm = feat_grads["task"]
    if task_feat_norm is None:
        raise ValueError("task_loss must always be part of the training objective")
    zero = torch.zeros((), device=task_feat_norm.device)

    if featkd_loss is None:
        feat_weight_task = torch.ones((), device=task_feat_norm.device)
    else:
        feat_weight_task = torch.clamp(
            feat_grads["featkd"] / (task_feat_norm + 1e-4), 0.0, 1e4
        ).detach()

    task_out_norm = out_grads["task"]
    if outkd_loss is None:
        out_weight_outkd = zero
    else:
        out_weight_outkd = torch.clamp(
            task_out_norm / (out_grads["outkd"] + 1e-6), 0.0, 1e6
        ).detach()

    if elatentlpips_loss is None:
        out_weight_elatentlpips = zero
    else:
        out_weight_elatentlpips = torch.clamp(
            task_out_norm / (out_grads["elatentlpips"] + 1e-6), 0.0, 1e6
        ).detach()

    diagnostics = {
        "feat_gnorm_featkd": float(feat_grads["featkd"]) if feat_grads["featkd"] is not None else 0.0,
        "feat_gnorm_task": float(feat_grads["task"]),
        "out_gnorm_task": float(out_grads["task"]),
    }
    if out_grads["outkd"] is not None:
        diagnostics["out_gnorm_outkd"] = float(out_grads["outkd"])
    if out_grads["elatentlpips"] is not None:
        diagnostics["out_gnorm_elatentlpips"] = float(out_grads["elatentlpips"])
    return feat_weight_task, out_weight_outkd, out_weight_elatentlpips, diagnostics


def total_loss(
    loss_featkd: Optional[torch.Tensor],
    loss_task: torch.Tensor,
    loss_outkd: Optional[torch.Tensor],
    loss_elatentlpips: Optional[torch.Tensor],
    *,
    feat_weight_task: torch.Tensor,
    out_weight_outkd: torch.Tensor,
    out_weight_elatentlpips: torch.Tensor,
) -> torch.Tensor:
    """Combine the four terms exactly as ``train_distillation.py:364-368``."""
    loss = torch.zeros((), device=loss_task.device, dtype=loss_task.dtype)
    if loss_featkd is not None:
        loss = loss + loss_featkd * KD_LOSS_WEIGHT
    balanced = loss_task * TASK_LOSS_WEIGHT
    if loss_outkd is not None:
        balanced = balanced + loss_outkd * out_weight_outkd * KD_LOSS_WEIGHT
    if loss_elatentlpips is not None:
        balanced = (
            balanced
            + loss_elatentlpips * out_weight_elatentlpips * ELATENTLPIPS_LOSS_WEIGHT
        )
    return loss + feat_weight_task * balanced
