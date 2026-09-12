"""Tests for the student distillation trainer (TDD §7.2).

* 20-step smoke on the synthetic cache for both the pixel and
  latent students.
* Cache key mismatch raises a clear error.
* Peak memory on a real GPU stays under 5 GB.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from moebius_finetune.students import (
    LatentStudentV0,
    PixelStudentV0,
)
from moebius_finetune.training.student import (
    SyntheticTeacherCache,
    TeacherCacheEntry,
    distill_student,
    validate_cache_keys,
)




# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _build_cache(num_cases: int = 2, seed: int = 0) -> list:
    return SyntheticTeacherCache(
        resolution=64, num_cases=num_cases, seed=seed
    ).entries()


# ---------------------------------------------------------------------------
# Pixel student smoke
# ---------------------------------------------------------------------------


def test_pixel_student_distill_20_steps(synthetic_teacher_cache):
    """20-step smoke for the pixel student on the synthetic cache."""
    torch.manual_seed(0)
    student = PixelStudentV0()
    trainer = distill_student(
        student,
        synthetic_teacher_cache,
        cfg={
            "steps": 20,
            "lr": 1e-4,
            "weight_decay": 0.01,
            "device": "cpu",
            "expected_scheduler_cfg": synthetic_teacher_cache[0].scheduler_cfg,
        },
    )
    # Final loss is finite.
    assert trainer.final_loss == trainer.final_loss
    # History has 20 entries.
    assert len(trainer.history) == 20
    # All losses are finite floats.
    for entry in trainer.history:
        assert entry["loss"] == entry["loss"]  # not NaN


def test_pixel_student_distill_keeps_params_finite(synthetic_teacher_cache):
    student = PixelStudentV0()
    distill_student(
        student,
        synthetic_teacher_cache,
        cfg={
            "steps": 20,
            "device": "cpu",
            "expected_scheduler_cfg": synthetic_teacher_cache[0].scheduler_cfg,
        },
    )
    for p in student.parameters():
        assert torch.isfinite(p).all()


# ---------------------------------------------------------------------------
# Latent student smoke
# ---------------------------------------------------------------------------


def test_latent_student_distill_20_steps(synthetic_teacher_cache):
    torch.manual_seed(0)
    student = LatentStudentV0()
    trainer = distill_student(
        student,
        synthetic_teacher_cache,
        cfg={
            "steps": 20,
            "lr": 1e-4,
            "weight_decay": 0.01,
            "device": "cpu",
            "expected_scheduler_cfg": synthetic_teacher_cache[0].scheduler_cfg,
        },
    )
    assert trainer.final_loss == trainer.final_loss
    assert len(trainer.history) == 20


# ---------------------------------------------------------------------------
# Cache key mismatch
# ---------------------------------------------------------------------------


def test_cache_key_mismatch_raises():
    """``validate_cache_keys`` rejects a cache whose scheduler config differs."""
    cache = _build_cache(num_cases=1, seed=0)
    bad_expected = {"num_steps": 999, "eta": 0.0}  # num_steps differs
    with pytest.raises(ValueError):
        validate_cache_keys(cache, bad_expected)


def test_cache_key_mismatch_raises_via_distill():
    """The trainer surfaces cache-key mismatches as a ValueError."""
    cache = _build_cache(num_cases=1, seed=0)
    student = PixelStudentV0()
    with pytest.raises(ValueError):
        distill_student(
            student,
            cache,
            cfg={
                "steps": 1,
                "device": "cpu",
                "expected_scheduler_cfg": {"num_steps": 999, "eta": 0.0},
            },
        )


def test_cache_schema_mismatch_raises():
    """An entry missing a required key raises ``KeyError``."""
    bad = {
        "case_id": "x",
        "seed": 0,
        "initial_noise": np.zeros((4, 8, 8), dtype=np.float32),
        # missing: final_latent, teacher_rgb, scheduler_cfg, rgb_hole, hole_mask, depth_hole
    }
    with pytest.raises(KeyError):
        TeacherCacheEntry.from_dict(bad)


# ---------------------------------------------------------------------------
# Memory (GPU only)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU-only memory check")
def test_pixel_student_distill_peak_memory_under_5gb():
    cache = _build_cache(num_cases=2, seed=0)
    student = PixelStudentV0().cuda()
    trainer = distill_student(
        student,
        cache,
        cfg={
            "steps": 20,
            "device": "cuda",
            "expected_scheduler_cfg": cache[0].scheduler_cfg,
        },
    )
    assert trainer.peak_memory_bytes < 5 * 1024 * 1024 * 1024, (
        f"pixel student peak memory {trainer.peak_memory_bytes} >= 5 GB"
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU-only memory check")
def test_latent_student_distill_peak_memory_under_5gb():
    cache = _build_cache(num_cases=2, seed=0)
    student = LatentStudentV0().cuda()
    trainer = distill_student(
        student,
        cache,
        cfg={
            "steps": 20,
            "device": "cuda",
            "expected_scheduler_cfg": cache[0].scheduler_cfg,
        },
    )
    assert trainer.peak_memory_bytes < 5 * 1024 * 1024 * 1024, (
        f"latent student peak memory {trainer.peak_memory_bytes} >= 5 GB"
    )
