"""Teacher fine-tuning recipes (TDD §5.3).

Each recipe is a frozen dataclass holding the optimizer / scheduler /
stage configuration for a phase of teacher training. The recipes do
**not** import anything from ``configs/`` (TDD §5.3: "三种 recipe 都不
直接 import ``configs/``") — they take plain dicts at construction and
are therefore testable in isolation.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class RecipeConfigError(ValueError):
    """Raised when a recipe dict is malformed."""


# ---------------------------------------------------------------------------
# Recipe
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TeacherTrainingStage:
    """A single training stage (depth-only or open-backbone)."""

    name: str
    max_steps: int
    backbone_lr: float
    depth_branch_lr: float
    optimizer: str = "adamw"
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    grad_accum_steps: int = 1
    amp: bool = True
    amp_dtype: str = "bf16"  # "bf16" or "fp16"
    gradient_checkpointing: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "max_steps": self.max_steps,
            "backbone_lr": self.backbone_lr,
            "depth_branch_lr": self.depth_branch_lr,
            "optimizer": self.optimizer,
            "weight_decay": self.weight_decay,
            "max_grad_norm": self.max_grad_norm,
            "grad_accum_steps": self.grad_accum_steps,
            "amp": self.amp,
            "amp_dtype": self.amp_dtype,
            "gradient_checkpointing": self.gradient_checkpointing,
        }


@dataclass(frozen=True)
class BaseRecipe:
    """Common recipe parameters shared by all three recipes."""

    seed: int = 0
    batch_size: int = 1
    resolution: int = 512
    log_every_steps: int = 1
    save_every_steps: int = 100
    grad_accum_steps: int = 1
    amp: bool = True
    amp_dtype: str = "bf16"
    gradient_checkpointing: bool = True

    def stage(self) -> TeacherTrainingStage:
        raise NotImplementedError("subclass responsibility")

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "seed": self.seed,
            "batch_size": self.batch_size,
            "resolution": self.resolution,
            "log_every_steps": self.log_every_steps,
            "save_every_steps": self.save_every_steps,
            "grad_accum_steps": self.grad_accum_steps,
            "amp": self.amp,
            "amp_dtype": self.amp_dtype,
            "gradient_checkpointing": self.gradient_checkpointing,
            "stage": self.stage().to_dict(),
        }
        return d


@dataclass(frozen=True)
class DepthBranchOnlyRecipe(BaseRecipe):
    """Stage 1: backbone frozen, only the depth branch trains (TDD §5.3)."""

    steps: int = 2000
    lr: float = 1e-4
    backbone_lr: float = 0.0

    def stage(self) -> TeacherTrainingStage:
        return TeacherTrainingStage(
            name="depth_only",
            max_steps=int(self.steps),
            backbone_lr=float(self.backbone_lr),
            depth_branch_lr=float(self.lr),
            optimizer="adamw",
            weight_decay=0.01,
            max_grad_norm=1.0,
            grad_accum_steps=int(self.grad_accum_steps),
            amp=bool(self.amp),
            amp_dtype=str(self.amp_dtype),
            gradient_checkpointing=bool(self.gradient_checkpointing),
        )


@dataclass(frozen=True)
class JointUnfreezeRecipe(BaseRecipe):
    """Stage 2: open the backbone, two learning rates (TDD §5.3)."""

    steps: int = 18000
    branch_lr: float = 1e-4
    backbone_lr: float = 1e-5

    def stage(self) -> TeacherTrainingStage:
        return TeacherTrainingStage(
            name="open_backbone",
            max_steps=int(self.steps),
            backbone_lr=float(self.backbone_lr),
            depth_branch_lr=float(self.branch_lr),
            optimizer="adamw",
            weight_decay=0.01,
            max_grad_norm=1.0,
            grad_accum_steps=int(self.grad_accum_steps),
            amp=bool(self.amp),
            amp_dtype=str(self.amp_dtype),
            gradient_checkpointing=bool(self.gradient_checkpointing),
        )


@dataclass(frozen=True)
class LocalSmokeRecipe(BaseRecipe):
    """20-step smoke for the local dev box (TDD §5.3)."""

    steps: int = 20
    lr: float = 1e-4
    backbone_lr: float = 0.0
    amp: bool = False  # CPU-only by default

    def __post_init__(self) -> None:
        # Local smoke runs on CPU and should not try BF16.
        if self.amp_dtype != "fp32":
            object.__setattr__(self, "amp_dtype", "fp32")
        if self.amp:
            object.__setattr__(self, "amp", False)

    def stage(self) -> TeacherTrainingStage:
        return TeacherTrainingStage(
            name="local_smoke",
            max_steps=int(self.steps),
            backbone_lr=float(self.backbone_lr),
            depth_branch_lr=float(self.lr),
            optimizer="adamw",
            weight_decay=0.01,
            max_grad_norm=1.0,
            grad_accum_steps=int(self.grad_accum_steps),
            amp=bool(self.amp),
            amp_dtype=str(self.amp_dtype),
            gradient_checkpointing=bool(self.gradient_checkpointing),
        )


# ---------------------------------------------------------------------------
# YAML loader
# ---------------------------------------------------------------------------


def recipe_from_yaml_dict(data: Mapping[str, Any]) -> BaseRecipe:
    """Build a recipe from a parsed YAML dict (no IO).

    The recipe factories are intentionally decoupled from
    ``moebius_finetune.contracts`` so the YAML loader in stage 0 can
    round-trip freely. The mapping must contain a ``kind`` field
    selecting the recipe type; for the two-stage recipe, an
    additional ``stage`` field is required.

    Supported kinds: ``depth_only``, ``open_backbone``, ``local_smoke``.
    """
    if not isinstance(data, Mapping):
        raise RecipeConfigError(
            f"recipe must be a mapping, got {type(data).__name__}"
        )
    kind = data.get("kind") or data.get("name")
    if kind is None:
        raise RecipeConfigError("recipe must include a 'kind' field")
    kind_str = str(kind).lower()
    common_kwargs: Dict[str, Any] = dict(
        seed=int(data.get("seed", 0)),
        batch_size=int(data.get("batch_size", 1)),
        resolution=int(data.get("resolution", 512)),
        log_every_steps=int(data.get("log_every_steps", 1)),
        save_every_steps=int(data.get("save_every_steps", 100)),
        grad_accum_steps=int(data.get("grad_accum_steps", 1)),
        amp=bool(data.get("amp", True)),
        amp_dtype=str(data.get("amp_dtype", "bf16")),
        gradient_checkpointing=bool(data.get("gradient_checkpointing", True)),
    )
    if kind_str in ("depth_only", "depth_branch_only", "stage1"):
        return DepthBranchOnlyRecipe(
            **common_kwargs,
            steps=int(data.get("steps", 2000)),
            lr=float(data.get("lr", 1e-4)),
            backbone_lr=float(data.get("backbone_lr", 0.0)),
        )
    if kind_str in ("open_backbone", "joint_unfreeze", "stage2"):
        return JointUnfreezeRecipe(
            **common_kwargs,
            steps=int(data.get("steps", 18000)),
            branch_lr=float(data.get("branch_lr", 1e-4)),
            backbone_lr=float(data.get("backbone_lr", 1e-5)),
        )
    if kind_str in ("local_smoke", "smoke"):
        amp = bool(data.get("amp", False))
        amp_dtype = str(data.get("amp_dtype", "fp32"))
        common_kwargs.pop("amp", None)
        common_kwargs.pop("amp_dtype", None)
        return LocalSmokeRecipe(
            **common_kwargs,
            steps=int(data.get("steps", 20)),
            lr=float(data.get("lr", 1e-4)),
            backbone_lr=float(data.get("backbone_lr", 0.0)),
            amp=amp,
            amp_dtype=amp_dtype,
        )
    raise RecipeConfigError(f"Unknown recipe kind: {kind!r}")


__all__ = [
    "BaseRecipe",
    "DepthBranchOnlyRecipe",
    "JointUnfreezeRecipe",
    "LocalSmokeRecipe",
    "RecipeConfigError",
    "TeacherTrainingStage",
    "recipe_from_yaml_dict",
]
