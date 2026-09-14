"""Tests for the RAM trajectory cache and teacher-free distillation (TDD3 §8).

Stub teacher + tiny student on CPU; no VAE, no ZipDepth, no GPU.
"""

from __future__ import annotations

import pytest
import torch

from moebius_finetune.students.mobile_moebius import MobileMoebius
from moebius_finetune.students.schedule import StudentSchedule
from moebius_finetune.training.student.mobile_distill import CaseBank
from moebius_finetune.training.student.trajectory_cache import (
    TrajTrainConfig,
    TrajectoryRamCache,
    recompute_states,
    rollout_states,
    train_mobile_student_traj,
)
from moebius_finetune.training.teacher.cache import default_scheduler_config

SCHEDULE = StudentSchedule.from_scheduler_config(default_scheduler_config())


def _stub_teacher(scale: float = 0.5):
    calls = {"t": []}

    def fn(x9: torch.Tensor, t: torch.Tensor, df: torch.Tensor) -> torch.Tensor:
        calls["t"].extend(int(v) for v in t.tolist())
        return x9[:, :4].detach() * scale

    return fn, calls


def _tiny_student() -> MobileMoebius:
    torch.manual_seed(0)
    return MobileMoebius(
        in_channels=11, out_channels=4,
        channels=(8, 16, 32), blocks=(1, 1, 1, 1, 1, 1),
        num_steps=10, time_dim=32,
    )


def _fake_bank(n: int = 4, size: int = 16) -> CaseBank:
    torch.manual_seed(1)
    entries = []
    for i in range(n):
        entries.append(
            {
                "case_id": f"case_{i}",
                "x0": torch.randn(1, 4, size, size),
                "ml": torch.randn(1, 4, size, size),
                "latent_mask": (torch.rand(1, 1, size, size) > 0.5).float(),
                "df": torch.randn(1, 2, size, size),
            }
        )
    return CaseBank(entries)


def test_full_timesteps_grid():
    t20 = SCHEDULE.full_timesteps()
    assert t20.tolist() == [951, 901, 851, 801, 751, 701, 651, 601, 551,
                            501, 451, 401, 351, 301, 251, 201, 151, 101, 51, 1]
    assert torch.equal(t20[::2], SCHEDULE.timesteps)


def test_rollout_captures_eps_on_teacher_grid():
    torch.manual_seed(2)
    B, size = 2, 16
    batch = {
        "ml": torch.randn(B, 4, size, size),
        "latent_mask": (torch.rand(B, 1, size, size) > 0.5).float(),
        "df": torch.randn(B, 2, size, size),
    }
    noise = torch.randn(B, 4, size, size)
    fn, calls = _stub_teacher(0.5)

    eps_traj = rollout_states(teacher_fn=fn, batch=batch, noise=noise, schedule=SCHEDULE)
    assert eps_traj.shape == (B, SCHEDULE.num_steps, 4, size, size)
    # teacher was queried on the full 20-step grid, each t once per batch
    assert set(calls["t"]) == set(SCHEDULE.full_timesteps().tolist())

    # captured eps[i] must equal the stub evaluated at the state the
    # teacher stood on at t_i — verify by replaying the chain manually.
    ab_full = SCHEDULE.alphas_bar_full
    x = noise.clone()
    for pos, t in enumerate(SCHEDULE.full_timesteps().tolist()):
        ab_t = ab_full[t].view(1, 1, 1, 1)
        ab_prev = (
            ab_full[SCHEDULE.full_timesteps()[pos + 1]].view(1, 1, 1, 1)
            if pos + 1 < 20
            else SCHEDULE.final_alpha_bar.view(1, 1, 1, 1)
        )
        if pos % 2 == 0:
            expected_eps = x[:, :4] * 0.5
            assert torch.allclose(eps_traj[:, pos // 2], expected_eps, atol=1e-5)
        pred_x0 = (x - (1 - ab_t).sqrt() * (x[:, :4] * 0.5)) / ab_t.sqrt()
        x = ab_prev.sqrt() * pred_x0 + (1 - ab_prev).sqrt() * (x[:, :4] * 0.5)


def test_recompute_states_matches_manual_chain():
    torch.manual_seed(3)
    B, size = 2, 16
    eps_traj = torch.randn(B, SCHEDULE.num_steps, 4, size, size)
    noise = torch.randn(B, 4, size, size)
    states = recompute_states(eps_traj=eps_traj, noise=noise, schedule=SCHEDULE)
    assert states.shape == (B, SCHEDULE.num_steps, 4, size, size)

    x = noise.clone()
    for i in range(SCHEDULE.num_steps):
        assert torch.allclose(states[:, i], x, atol=1e-5)
        ab_t = SCHEDULE.alphas_bar[i].view(1, 1, 1, 1)
        ab_prev = (
            SCHEDULE.alphas_bar[i + 1].view(1, 1, 1, 1)
            if i + 1 < SCHEDULE.num_steps
            else SCHEDULE.final_alpha_bar.view(1, 1, 1, 1)
        )
        pred_x0 = (x - (1 - ab_t).sqrt() * eps_traj[:, i]) / ab_t.sqrt()
        x = ab_prev.sqrt() * pred_x0 + (1 - ab_prev).sqrt() * eps_traj[:, i]


def test_ram_cache_roundtrip_and_sampling():
    torch.manual_seed(4)
    cache = TrajectoryRamCache(SCHEDULE, latent_hw=16)
    B, size = 3, 16
    for _ in range(2):  # two extend chunks
        cache.extend(
            eps=torch.randn(B, SCHEDULE.num_steps, 4, size, size),
            noise=torch.randn(B, 4, size, size),
            case_idx=torch.arange(B),
        )
    cache.finalize()
    assert cache.num_traj == 6
    assert cache.ram_gb > 0.0
    with pytest.raises(ValueError):
        cache.extend(torch.randn(1, 3, 4, size, size), torch.randn(1, 4, size, size),
                     torch.zeros(1, dtype=torch.int64))

    sample = cache.sample(4)
    assert sample["eps"].shape == (4, SCHEDULE.num_steps, 4, size, size)
    assert sample["eps"].dtype == torch.bfloat16
    assert sample["eps"].is_pinned() and sample["noise"].is_pinned()
    assert sample["case_idx"].min() >= 0 and sample["case_idx"].max() < 6


def test_traj_train_loop_learns_stub_target(tmp_path):
    """Cache built from a stub teacher (eps = 0.5·x_t on its own chain):
    the student must drive the trajectory eps-MSE down, teacher-free."""
    torch.manual_seed(5)
    B, size = 4, 16
    batch = {
        "ml": torch.randn(B, 4, size, size),
        "latent_mask": (torch.rand(B, 1, size, size) > 0.5).float(),
        "df": torch.randn(B, 2, size, size),
    }
    noise = torch.randn(B, 4, size, size)
    fn, _ = _stub_teacher(0.5)
    eps_traj = rollout_states(teacher_fn=fn, batch=batch, noise=noise, schedule=SCHEDULE)

    cache = TrajectoryRamCache(SCHEDULE)
    for k in range(3):  # 3 identical trajs per case → 12 trajectories
        cache.extend(eps_traj, noise, torch.arange(B))
    cache.finalize()

    cfg = TrajTrainConfig(steps=60, microbatch=6, grad_accum=1, lr=5e-3,
                          warmup=5, save_every=30, log_every=20, seed=0)
    history = train_mobile_student_traj(
        student=_tiny_student(), bank=_fake_bank(), cache=cache,
        schedule=SCHEDULE, config=cfg, device=torch.device("cpu"),
        output_dir=str(tmp_path), log=lambda _m: None,
    )
    assert history[-1].step == cfg.steps
    first = sum(s.loss for s in history[:2]) / 2
    last = sum(s.loss for s in history[-2:]) / 2
    assert last < first * 0.9, f"loss did not drop: {first} → {last}"

    ckpt = tmp_path / f"student_step{cfg.steps:07d}.pt"
    assert ckpt.exists()
    payload = torch.load(ckpt, map_location="cpu", weights_only=False)
    assert payload["config"]["objective"] == "trajectory-eps-MSE"
    student = _tiny_student()
    student.load_state_dict(payload["model_state"])


def test_traj_train_rejects_oversized_microbatch():
    cache = TrajectoryRamCache(SCHEDULE)
    cache.extend(torch.randn(2, SCHEDULE.num_steps, 4, 16, 16),
                 torch.randn(2, 4, 16, 16), torch.zeros(2, dtype=torch.int64))
    cache.finalize()
    with pytest.raises(ValueError, match="exceeds cache size"):
        train_mobile_student_traj(
            student=_tiny_student(), bank=_fake_bank(), cache=cache,
            schedule=SCHEDULE,
            config=TrajTrainConfig(steps=2, microbatch=8),
            device=torch.device("cpu"), log=lambda _m: None,
        )
