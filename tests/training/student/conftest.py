"""Shared fixtures for the student-distillation tests.

The :class:`SyntheticTeacherCache` produces a small, well-typed
cache that satisfies the schema B will eventually use. **This cache
is synthetic; the resulting smoke test is *not* a quality
benchmark.** It only proves the distillation loop runs end to end.
"""

from __future__ import annotations

import numpy as np
import pytest

from moebius_finetune.training.student import SyntheticTeacherCache


@pytest.fixture
def synthetic_teacher_cache() -> list:
    """A small teacher cache (2 cases at 64x64) for the 20-step smoke."""
    return SyntheticTeacherCache(resolution=64, num_cases=2, seed=0).entries()


@pytest.fixture
def tiny_synthetic_teacher_cache() -> list:
    """An even smaller cache (1 case) for the cache-key mismatch test."""
    return SyntheticTeacherCache(resolution=64, num_cases=1, seed=7).entries()
