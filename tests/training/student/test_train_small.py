"""Smoke tests for the multi-granularity training loop (plan todo 8).

Everything heavy is injected: a stub TAESDXL, a stub teacher carrying the
measured tap shapes, a stub perceptual metric, and a synthetic case
builder. This verifies the loop mechanics (seeding, the long/float
timestep split, nearest-neighbour mask resizing, the no-grad teacher
call, checkpointing and resume) on CPU in seconds.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import DDPMScheduler

from moebius_finetune.students.moebius_small import MoebiusSmallStudent, StudentOutput
from moebius_finetune.training.student.prompt import HOLE_FILL_VALUE, build_masked_prompt
from moebius_finetune.training.student.train_small import TrainConfig, _to_latents, train

#: teacher taps 2/3/5 must match FeatureProjections' default teacher channels
TAP_SHAPES = {2: (1280, 16, 16), 3: (1280, 32, 32), 5: (320, 64, 64)}
RGB = 512


class _StubTAESD(nn.Module):
    """8x spatial reduction with a 4-channel bottleneck."""

    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.ones(1), requires_grad=False)

    def encoder(self, x01: torch.Tensor) -> torch.Tensor:
        pooled = F.avg_pool2d(x01, 8)
        return torch.cat([pooled, pooled], dim=1)[:, :4]


class _StubLPIPS(nn.Module):
    def forward(self, a, b, **kwargs) -> torch.Tensor:
        return (a - b).pow(2).mean()


class _StubTeacher(nn.Module):
    """Deterministic teacher that records how it was called."""

    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.ones(1), requires_grad=False)
        self.calls: dict[str, list] = {"grad_enabled": [], "t_dtype": [], "batch": []}

    def forward(self, x9, t, input_ids):
        self.calls["grad_enabled"].append(torch.is_grad_enabled())
        self.calls["t_dtype"].append(t.dtype)
        self.calls["batch"].append(int(x9.shape[0]))
        batch = x9.shape[0]
        sample = x9[:, :4] * 0.25
        taps = [
            torch.full((batch, *TAP_SHAPES[i]), 0.5) if i in TAP_SHAPES else torch.zeros(batch, 1, 1, 1)
            for i in range(max(TAP_SHAPES) + 1)
        ]
        return StudentOutput(sample=sample, block_outputs=taps)


def _builder(subset) -> list[dict]:
    rng = np.random.default_rng(0)
    out = []
    for _ in subset:
        target = rng.random((3, RGB, RGB), dtype=np.float32)
        mask = (rng.random((1, RGB, RGB), dtype=np.float32) > 0.6).astype(np.float32)
        out.append(
            {"target": target, "rgb_hole": target * (1 - mask), "hole_mask": mask}
        )
    return out


def _components() -> dict:
    torch.manual_seed(0)
    student = MoebiusSmallStudent(
        in_channels=9, out_channels=4, channels=(8, 16, 32), blocks=(2, 2, 2), time_dim=32
    )
    return {
        "student": student,
        "taesd": _StubTAESD(),
        "teacher": _StubTeacher(),
        "lpips": _StubLPIPS(),
        "scheduler": DDPMScheduler(
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            num_train_timesteps=1000,
            clip_sample=False,
        ),
        "cases": list(range(8)),
        "builder": _builder,
        "device": torch.device("cpu"),
    }


@pytest.fixture
def stub_components(monkeypatch):
    """Inject the stub stack and record the timestep dtype the student sees."""
    parts = _components()
    seen_t: list[torch.dtype] = []
    original = parts["student"].time_embed.forward

    def spy(t):
        seen_t.append(t.dtype)
        return original(t)

    monkeypatch.setattr(parts["student"].time_embed, "forward", spy)
    return parts, seen_t


def _cfg(tmp_path, **overrides) -> TrainConfig:
    base = dict(
        steps=5,
        batch_size=2,
        lr=1e-3,
        warmup=2,
        log_every=5,
        save_every=5,
        seed=0,
        workers=1,
        out_dir=str(tmp_path),
    )
    base.update(overrides)
    return TrainConfig(**base)


def test_loop_runs_losses_finite_and_checkpoint_roundtrips(tmp_path, stub_components):
    components, _seen_t = stub_components
    messages: list[str] = []
    result = train(_cfg(tmp_path), messages.append, **components)

    assert result["steps"] == 5
    assert len(result["history"]) == 5
    assert all(np.isfinite(r["loss"]) for r in result["history"])
    for key in ("loss_featkd", "loss_outkd", "loss_task", "loss_elatentlpips"):
        assert all(np.isfinite(r[key]) for r in result["history"])
    assert any("saved final.pt" in m for m in messages)

    payload = torch.load(tmp_path / "final.pt", map_location="cpu", weights_only=False)
    assert payload["step"] == 5
    assert "projection_state" in payload
    fresh = MoebiusSmallStudent(
        in_channels=9, out_channels=4, channels=(8, 16, 32), blocks=(2, 2, 2), time_dim=32
    )
    fresh.load_state_dict(payload["model_state"])


def test_teacher_runs_under_no_grad_and_timestep_types_are_not_mixed(
    tmp_path, stub_components
):
    components, seen_t = stub_components
    train(_cfg(tmp_path), lambda _m: None, **components)
    teacher = components["teacher"]
    assert teacher.calls["grad_enabled"] and not any(teacher.calls["grad_enabled"])
    assert all(dtype == torch.int64 for dtype in teacher.calls["t_dtype"])
    assert seen_t and all(dtype.is_floating_point for dtype in seen_t)


def test_latent_mask_is_resized_with_nearest(tmp_path, stub_components, monkeypatch):
    calls: list[tuple] = []
    real = F.interpolate

    def spy(input, size=None, **kwargs):
        calls.append((kwargs.get("mode"), tuple(size) if size else None))
        return real(input, size=size, **kwargs)

    monkeypatch.setattr(F, "interpolate", spy)
    components, _seen_t = stub_components
    train(_cfg(tmp_path, steps=2), lambda _m: None, **components)
    mask_calls = [mode for mode, size in calls if size == (64, 64)]
    assert mask_calls == ["nearest"] * 2


def test_disabled_perceptual_term_still_trains(tmp_path, stub_components):
    components, _seen_t = stub_components
    result = train(
        _cfg(tmp_path, steps=2), lambda _m: None, **{**components, "lpips": None}
    )
    assert all(r["loss_elatentlpips"] == 0.0 for r in result["history"])
    assert all(np.isfinite(r["loss"]) for r in result["history"])


def test_resume_continues_from_checkpoint(tmp_path, stub_components):
    components, _seen_t = stub_components
    first = train(_cfg(tmp_path, steps=3), lambda _m: None, **components)
    assert first["history"][-1]["step"] == 3

    resumed = train(
        _cfg(tmp_path, steps=6, resume=str(tmp_path / "final.pt")),
        lambda _m: None,
        **components,
    )
    assert len(resumed["history"]) == 3
    assert resumed["history"][0]["step"] == 4
    assert all(np.isfinite(r["loss"]) for r in resumed["history"])


# ---------------------------------------------------------------------------
# Prompt convention (the hole must be mid-gray, not black)
# ---------------------------------------------------------------------------


def test_build_masked_prompt_puts_midgray_in_the_hole():
    rgb = torch.full((1, 3, 4, 4), 0.25)
    mask = torch.zeros(1, 1, 4, 4)
    mask[..., 1:3, 1:3] = 1.0
    prompt = build_masked_prompt(rgb, mask)
    assert float(prompt[0, 0, 1, 1]) == HOLE_FILL_VALUE == 0.5
    assert float(prompt[0, 0, 0, 0]) == 0.25
    # in the model's [-1, 1] space the hole reads as 0, not -1
    assert abs(float(prompt[0, 0, 1, 1] * 2 - 1)) < 1e-6


def test_to_latents_feeds_the_encoder_a_midgray_hole():
    """The training loop must not hand TAESDXL a black hole."""
    recorded: list[torch.Tensor] = []

    class _Recorder(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.dummy = nn.Parameter(torch.zeros(1), requires_grad=False)

        def encoder(self, x01: torch.Tensor) -> torch.Tensor:
            recorded.append(x01)
            return x01[:, :4]

    built = [
        {
            "target": np.full((3, 8, 8), 0.25, np.float32),
            "rgb_hole": np.zeros((3, 8, 8), np.float32),
            "hole_mask": np.ones((1, 8, 8), np.float32),
        }
    ]
    _to_latents(built, _Recorder(), torch.device("cpu"), torch.float32)
    prompt = recorded[1]
    assert float(prompt.min()) == 0.5 and float(prompt.max()) == 0.5
