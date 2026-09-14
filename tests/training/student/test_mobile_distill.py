"""Tests for the MobileMoebius teacher-forced distillation loop (TDD3 M3).

Uses a tiny student config and a deterministic stub teacher so the
loop mechanics (batching, schedule wiring, checkpointing, sampling)
are verified on CPU without the 226M teacher or any VAE/ZipDepth
dependency.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from moebius_finetune.students.mobile_moebius import MobileMoebius
from moebius_finetune.students.schedule import StudentSchedule
from moebius_finetune.training.student.mobile_distill import (
    CaseBank,
    MobileDistillConfig,
    sample_mobile_student,
    sample_teacher_reference,
    train_mobile_student,
)
from moebius_finetune.training.teacher.cache import default_scheduler_config


SCHED_CFG = default_scheduler_config()


def _tiny_student() -> MobileMoebius:
    torch.manual_seed(0)
    return MobileMoebius(
        in_channels=11,
        out_channels=4,
        channels=(8, 16, 32),
        blocks=(1, 1, 1, 1, 1, 1),
        num_steps=10,
        time_dim=32,
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


def _stub_teacher(target_scale: float = 0.5):
    """Deterministic teacher: eps = scale · x_t. Records call metadata."""
    calls = {"t": [], "shapes": []}

    def fn(x9: torch.Tensor, t: torch.Tensor, df: torch.Tensor) -> torch.Tensor:
        calls["t"].extend(int(v) for v in t.tolist())
        calls["shapes"].append(tuple(x9.shape))
        return x9[:, :4].detach() * target_scale

    return fn, calls


SCHEDULE = StudentSchedule.from_scheduler_config(SCHED_CFG)


def test_train_loop_learns_stub_teacher(tmp_path):
    cfg = MobileDistillConfig(
        steps=80, microbatch=4, grad_accum=1, lr=5e-3, warmup=5,
        save_every=40, log_every=10, seed=0,
    )
    student = _tiny_student()
    teacher_fn, calls = _stub_teacher(0.5)
    bank = _fake_bank()

    history = train_mobile_student(
        student=student,
        teacher_fn=teacher_fn,
        bank=bank,
        schedule=SCHEDULE,
        config=cfg,
        device=torch.device("cpu"),
        output_dir=str(tmp_path),
        log=lambda _msg: None,
    )

    assert history and history[-1].step == cfg.steps
    assert all(torch.isfinite(torch.tensor(s.loss)) for s in history)
    # teacher saw exactly the timesteps mapped from the schedule table
    valid_t = set(SCHEDULE.timesteps.tolist())
    assert set(calls["t"]).issubset(valid_t)
    assert all(s == (4, 9, 16, 16) for s in calls["shapes"])
    # loss decreased meaningfully on this trivially learnable target
    first = sum(s.loss for s in history[:2]) / 2
    last = sum(s.loss for s in history[-2:]) / 2
    assert last < first * 0.6, f"loss did not drop: {first} → {last}"
    # checkpoints exist (save_every=40, final)
    ckpts = sorted(tmp_path.glob("student_step*.pt"))
    assert [p.name for p in ckpts] == ["student_step0000040.pt", "student_step0000080.pt"]
    payload = torch.load(ckpts[-1], map_location="cpu", weights_only=False)
    assert payload["step"] == cfg.steps
    assert payload["schedule"]["timesteps"] == SCHEDULE.timesteps.tolist()
    student.load_state_dict(payload["model_state"])  # loads back cleanly


def test_train_loop_rejects_microbatch_larger_than_bank():
    cfg = MobileDistillConfig(steps=2, microbatch=8)
    with __import__("pytest").raises(ValueError, match="exceeds bank size"):
        train_mobile_student(
            student=_tiny_student(),
            teacher_fn=_stub_teacher()[0],
            bank=_fake_bank(n=4),
            schedule=SCHEDULE,
            config=cfg,
            device=torch.device("cpu"),
            log=lambda _msg: None,
        )


def test_sample_mobile_student_runs_and_matches_bank_shape():
    student = _tiny_student().eval()
    bank = _fake_bank(n=1)
    torch.manual_seed(3)
    noise = torch.randn(1, 4, 16, 16)
    out = sample_mobile_student(
        student=student,
        bank_entry=bank.entries[0],
        schedule=SCHEDULE,
        initial_noise=noise,
        device=torch.device("cpu"),
    )
    assert out.shape == (1, 4, 16, 16)
    assert torch.isfinite(out).all()
    # deterministic: same noise → same latent
    out2 = sample_mobile_student(
        student=student,
        bank_entry=bank.entries[0],
        schedule=SCHEDULE,
        initial_noise=noise,
        device=torch.device("cpu"),
    )
    assert torch.allclose(out, out2, atol=1e-6)


def test_sample_teacher_reference_runs():
    teacher_fn, _ = _stub_teacher(0.5)
    bank = _fake_bank(n=1)
    torch.manual_seed(4)
    noise = torch.randn(1, 4, 16, 16)
    out = sample_teacher_reference(
        teacher_fn=teacher_fn,
        bank_entry=bank.entries[0],
        schedule=SCHEDULE,
        initial_noise=noise,
        device=torch.device("cpu"),
        alphas_bar_full=SCHEDULE.alphas_bar_full,
    )
    assert out.shape == (1, 4, 16, 16)
    assert torch.isfinite(out).all()
