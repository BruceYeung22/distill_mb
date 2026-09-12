"""Student training (TDD §7).

Provides the offline one-step distillation loss and the 20-step smoke
trainer for both the pixel and the latent students. The training
loop reads from a teacher cache (produced by Agent B) and produces
a student that predicts the teacher's terminal latent / RGB from
the same initial noise.
"""

from __future__ import annotations

from moebius_finetune.training.student.distill import (
    DistillTrainer,
    SyntheticTeacherCache,
    TeacherCacheEntry,
    distill_student,
    validate_cache_keys,
)
from moebius_finetune.training.student.losses import (
    boundary_gradient_l1,
    distillation_loss,
    hole_l1,
)

__all__ = [
    "DistillTrainer",
    "SyntheticTeacherCache",
    "TeacherCacheEntry",
    "boundary_gradient_l1",
    "distillation_loss",
    "distill_student",
    "hole_l1",
    "validate_cache_keys",
]
