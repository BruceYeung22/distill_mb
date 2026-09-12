"""Teacher fine-tuning and cache subpackage (Agent B).

This subpackage owns the training loop, the loss functions, the recipe
configuration and the teacher cache for the Moebius depth-conditioned
teacher. It must not import any other agent's module
(``data/``, ``evaluation/``, ``students/``, ``training/codec/``,
``training/student/``).
"""

from __future__ import annotations

from .cache import (
    CacheKeyMismatchError,
    TeacherCacheEntry,
    cache_teacher_outputs,
    derive_seed,
    stable_cache_key,
)
from .finetune import (
    TeacherFinetuneArtifacts,
    finetune_depth_branch,
    verify_resume_consistency,
)
from .losses import combined_epsilon_loss, epsilon_mse
from .recipe import (
    DepthBranchOnlyRecipe,
    JointUnfreezeRecipe,
    LocalSmokeRecipe,
)

__all__ = [
    "CacheKeyMismatchError",
    "DepthBranchOnlyRecipe",
    "JointUnfreezeRecipe",
    "LocalSmokeRecipe",
    "TeacherCacheEntry",
    "TeacherFinetuneArtifacts",
    "cache_teacher_outputs",
    "combined_epsilon_loss",
    "derive_seed",
    "epsilon_mse",
    "finetune_depth_branch",
    "stable_cache_key",
    "verify_resume_consistency",
]
