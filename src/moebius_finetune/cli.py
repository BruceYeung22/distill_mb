"""Console entry points for the moebius-finetune package.

Stage 0 ships a minimal stub: each entry point is wired in a later
stage by Agents A/B/C and currently raises :class:`NotImplementedError`
with a clear message. Keeping the module present ensures
``pyproject.toml`` console scripts resolve at import time.
"""

from __future__ import annotations

from typing import Sequence


def _not_implemented(stage: str) -> "NotImplementedError":
    return NotImplementedError(f"wired in later stage ({stage})")


def prepare_data(argv: Sequence[str] | None = None) -> int:
    """prepare-data entry point (Agent A)."""
    raise _not_implemented("A: data preparation")


def train_teacher(argv: Sequence[str] | None = None) -> int:
    """train-teacher entry point (Agent B)."""
    raise _not_implemented("B: teacher fine-tuning")


def cache_teacher(argv: Sequence[str] | None = None) -> int:
    """cache-teacher entry point (Agent B)."""
    raise _not_implemented("B: teacher cache generation")


def train_codec(argv: Sequence[str] | None = None) -> int:
    """train-codec entry point (Agent C)."""
    raise _not_implemented("C: lightweight codec training")


def train_student(argv: Sequence[str] | None = None) -> int:
    """train-student entry point (Agent C)."""
    raise _not_implemented("C: student training")


def evaluate(argv: Sequence[str] | None = None) -> int:
    """evaluate entry point (Agent A)."""
    raise _not_implemented("A: unified evaluation")


def profile(argv: Sequence[str] | None = None) -> int:
    """profile entry point (Agent C)."""
    raise _not_implemented("C: budget profiling")


def export(argv: Sequence[str] | None = None) -> int:
    """export entry point (Agent C)."""
    raise _not_implemented("C: ONNX / RKNN export")
