"""Tests for :mod:`moebius_finetune.training.teacher.recipe`."""

from __future__ import annotations

import pytest

from moebius_finetune.training.teacher.recipe import (
    ConvInUnfreezeRecipe,
    DepthBranchOnlyRecipe,
    JointUnfreezeRecipe,
    LocalSmokeRecipe,
    RecipeConfigError,
    recipe_from_yaml_dict,
)


def test_depth_branch_only_defaults():
    r = DepthBranchOnlyRecipe()
    s = r.stage()
    assert s.max_steps == 2000
    assert s.backbone_lr == 0.0
    assert s.conv_in_lr == 0.0
    assert s.depth_branch_lr == 1e-4
    assert s.optimizer == "adamw"
    assert s.weight_decay == 0.01
    assert s.max_grad_norm == 1.0


def test_joint_unfreeze_defaults():
    r = JointUnfreezeRecipe()
    s = r.stage()
    assert s.max_steps == 18000
    assert s.backbone_lr == 1e-5
    assert s.depth_branch_lr == 1e-4
    # conv_in follows the backbone lr on the full-open path.
    assert s.conv_in_lr == 1e-5


def test_conv_in_unfreeze_defaults():
    """TDD2 D2: stage 2 unfreezes ONLY conv_in; backbone stays at 0."""
    r = ConvInUnfreezeRecipe()
    s = r.stage()
    assert s.name == "unfreeze_conv_in"
    assert s.max_steps == 18000
    assert s.backbone_lr == 0.0
    assert s.conv_in_lr == 1e-5
    assert s.depth_branch_lr == 1e-4


def test_recipe_from_yaml_conv_in_unfreeze():
    cfg = {
        "kind": "unfreeze_conv_in",
        "steps": 555,
        "branch_lr": 5e-4,
        "conv_in_lr": 2e-5,
    }
    r = recipe_from_yaml_dict(cfg)
    assert isinstance(r, ConvInUnfreezeRecipe)
    s = r.stage()
    assert s.max_steps == 555
    assert s.depth_branch_lr == 5e-4
    assert s.conv_in_lr == 2e-5
    assert s.backbone_lr == 0.0


def test_local_smoke_defaults():
    r = LocalSmokeRecipe()
    s = r.stage()
    assert s.max_steps == 20
    assert s.depth_branch_lr == 1e-4
    # Local smoke is CPU-friendly; AMP is off and dtype is fp32.
    assert s.amp is False
    assert s.amp_dtype == "fp32"


def test_recipe_from_yaml_depth_only():
    cfg = {
        "kind": "depth_only",
        "steps": 1234,
        "lr": 2e-4,
        "backbone_lr": 0.0,
        "seed": 7,
    }
    r = recipe_from_yaml_dict(cfg)
    assert isinstance(r, DepthBranchOnlyRecipe)
    s = r.stage()
    assert s.max_steps == 1234
    assert s.depth_branch_lr == 2e-4
    assert r.seed == 7


def test_recipe_from_yaml_joint_unfreeze():
    cfg = {
        "kind": "open_backbone",
        "steps": 555,
        "branch_lr": 5e-4,
        "backbone_lr": 1e-5,
    }
    r = recipe_from_yaml_dict(cfg)
    assert isinstance(r, JointUnfreezeRecipe)
    s = r.stage()
    assert s.max_steps == 555
    assert s.depth_branch_lr == 5e-4
    assert s.backbone_lr == 1e-5


def test_recipe_from_yaml_smoke():
    cfg = {
        "kind": "local_smoke",
        "steps": 5,
        "lr": 1e-4,
    }
    r = recipe_from_yaml_dict(cfg)
    assert isinstance(r, LocalSmokeRecipe)
    s = r.stage()
    assert s.max_steps == 5


def test_recipe_from_yaml_unknown_kind():
    with pytest.raises(RecipeConfigError):
        recipe_from_yaml_dict({"kind": "nope"})


def test_recipe_from_yaml_missing_kind():
    with pytest.raises(RecipeConfigError):
        recipe_from_yaml_dict({})


def test_recipe_to_dict_round_trip():
    r = DepthBranchOnlyRecipe(steps=10, lr=1e-3, seed=42)
    d = r.to_dict()
    assert d["stage"]["max_steps"] == 10
    assert d["seed"] == 42
    assert d["stage"]["depth_branch_lr"] == 1e-3
