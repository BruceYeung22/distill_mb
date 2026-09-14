"""Tests for the fixed 10-step student schedule (TDD3 M2).

The hand-written DDIM update is numerically pinned against
``diffusers.DDIMScheduler.step`` so the student integrator is exactly
the teacher's integrator semantics on the subsampled grid.
"""

from __future__ import annotations

import pytest
import torch

from moebius_finetune.students.schedule import StudentSchedule


CFG = {
    "scheduler_type": "DDIM",
    "num_inference_steps": 20,
    "num_train_timesteps": 1000,
    "beta_start": 0.00085,
    "beta_end": 0.012,
    "beta_schedule": "scaled_linear",
    "steps_offset": 1,
    "prediction_type": "epsilon",
}


@pytest.fixture(scope="module")
def schedule() -> StudentSchedule:
    return StudentSchedule.from_scheduler_config(CFG)


def test_table_matches_even_subsample_of_teacher_20_steps(schedule: StudentSchedule):
    from diffusers import DDIMScheduler

    ddim = DDIMScheduler(
        num_train_timesteps=1000, beta_start=0.00085, beta_end=0.012,
        beta_schedule="scaled_linear", clip_sample=False,
        set_alpha_to_one=False, steps_offset=1,
    )
    ddim.set_timesteps(20)
    expected = ddim.timesteps.cpu()[::2]
    assert torch.equal(schedule.timesteps, expected)
    assert schedule.timesteps.tolist() == [951, 851, 751, 651, 551, 451, 351, 251, 151, 51]
    # and it is NOT the same as set_timesteps(10) — the reason this
    # module builds the table by hand.
    ddim.set_timesteps(10)
    assert not torch.equal(schedule.timesteps, ddim.timesteps.cpu())


def test_alphas_bar_descends_in_x0_weight(schedule: StudentSchedule):
    ab = schedule.alphas_bar
    assert torch.all(ab[1:] > ab[:-1])  # ᾱ grows as t decreases
    assert float(ab[0]) < 0.02 and float(ab[-1]) > 0.9
    assert float(schedule.final_alpha_bar) == pytest.approx(
        float(schedule.alphas_bar_full[0]), abs=1e-7
    )


def test_q_sample_deterministic_and_matches_formula(schedule: StudentSchedule):
    torch.manual_seed(0)
    x0 = torch.randn(2, 4, 8, 8)
    eps = torch.randn(2, 4, 8, 8)
    idx = torch.tensor([0, 9])
    x_t = schedule.q_sample(x0, eps, idx)
    ab = schedule.alphas_bar[[0, 9]].view(-1, 1, 1, 1)
    expected = ab.sqrt() * x0 + (1 - ab).sqrt() * eps
    assert torch.allclose(x_t, expected, atol=1e-6)
    assert torch.equal(schedule.q_sample(x0, eps, idx), x_t)


def test_q_sample_rejects_out_of_range(schedule: StudentSchedule):
    x0 = torch.randn(1, 4, 8, 8)
    eps = torch.randn_like(x0)
    with pytest.raises(ValueError, match="out of range"):
        schedule.q_sample(x0, eps, torch.tensor([10]))
    with pytest.raises(ValueError, match="out of range"):
        schedule.q_sample(x0, eps, torch.tensor([-1]))


def test_ddim_step_matches_diffusers_mid_trajectory(schedule: StudentSchedule):
    """step from t=951 towards the next table entry must equal the
    teacher scheduler stepping from 951 to 901 (same ᾱ pair)."""
    from diffusers import DDIMScheduler

    torch.manual_seed(1)
    ddim = DDIMScheduler(
        num_train_timesteps=1000, beta_start=0.00085, beta_end=0.012,
        beta_schedule="scaled_linear", clip_sample=False,
        set_alpha_to_one=False, steps_offset=1,
    )
    ddim.set_timesteps(20)
    ddim.alphas_cumprod = ddim.alphas_cumprod.cpu()

    x = torch.randn(1, 4, 16, 16)
    eps = torch.randn(1, 4, 16, 16)

    ref = ddim.step(eps, torch.tensor([951]), x, return_dict=False)[0]

    # hand-written update with the SAME ᾱ pair (951 → 901):
    ab_t = schedule.alphas_bar_full[951]
    ab_prev = schedule.alphas_bar_full[901]
    pred_x0 = (x - (1 - ab_t).sqrt() * eps) / ab_t.sqrt()
    expected = ab_prev.sqrt() * pred_x0 + (1 - ab_prev).sqrt() * eps
    assert torch.allclose(ref, expected, atol=1e-6)

    # and our ddim_step on entry 0 (t=951 → next entry t=851) uses the
    # subsampled pair — sanity that the table wiring is consistent.
    ours = schedule.ddim_step(x, eps, 0)
    ab851 = schedule.alphas_bar_full[851]
    exp851 = ab851.sqrt() * pred_x0 + (1 - ab851).sqrt() * eps
    assert torch.allclose(ours, exp851, atol=1e-6)


def test_ddim_step_final_uses_final_alpha_bar(schedule: StudentSchedule):
    """After the last table entry (t=51) the update integrates towards
    final_alpha_bar (alphas_cumprod[0], set_alpha_to_one=False)."""
    torch.manual_seed(2)
    x = torch.randn(1, 4, 16, 16)
    eps = torch.randn(1, 4, 16, 16)
    ours = schedule.ddim_step(x, eps, schedule.num_steps - 1)
    ab_t = schedule.alphas_bar[-1]
    ab_prev = schedule.final_alpha_bar
    pred_x0 = (x - (1 - ab_t).sqrt() * eps) / ab_t.sqrt()
    expected = ab_prev.sqrt() * pred_x0 + (1 - ab_prev).sqrt() * eps
    assert torch.allclose(ours, expected, atol=1e-6)


def test_ddim_step_rejects_bad_index(schedule: StudentSchedule):
    x = torch.randn(1, 4, 8, 8)
    eps = torch.randn_like(x)
    with pytest.raises(ValueError):
        schedule.ddim_step(x, eps, -1)
    with pytest.raises(ValueError):
        schedule.ddim_step(x, eps, schedule.num_steps)
