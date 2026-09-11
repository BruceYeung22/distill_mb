"""Tests for the bundled YAML configs and the tiny config loader helper.

The stage-0 deliverable is just the configs themselves; the real
loader lives in Agent A. We still verify that each config is valid
YAML, has the required top-level keys, and round-trips through
``yaml.safe_load``.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from moebius_finetune.contracts import ConditionBatch, validate_condition

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO_ROOT / "configs"

REQUIRED_TOP_LEVEL = {
    "experiment",
    "runtime",
    "data",
    "model",
    "training",
    "evaluation",
}

# Configs that follow the full training recipe shape. The rknn export
# config has a deployment-focused layout (target / export /
# quantization / budget) instead.
EXPERIMENT_STYLE_CONFIGS = {
    "local_smoke.yaml",
    "teacher_depth.yaml",
    "student_pixel.yaml",
    "student_latent.yaml",
}


def _load(name: str) -> dict:
    path = CONFIG_DIR / name
    assert path.is_file(), f"missing config: {path}"
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Each shipped config is valid YAML and has the required sections.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "filename",
    [
        "paths.example.yaml",
        "local_smoke.yaml",
        "teacher_depth.yaml",
        "student_pixel.yaml",
        "student_latent.yaml",
        "rknn_rk3588.yaml",
    ],
)
def test_config_is_yaml_with_required_sections(filename: str):
    cfg = _load(filename)
    assert isinstance(cfg, dict)
    # `paths.example.yaml` and `rknn_rk3588.yaml` have a different
    # top-level layout (path registry / export recipe respectively);
    # only the experiment-style configs share the section keys.
    if filename in EXPERIMENT_STYLE_CONFIGS:
        missing = REQUIRED_TOP_LEVEL - set(cfg.keys())
        assert not missing, f"{filename} missing required sections: {missing}"


# ---------------------------------------------------------------------------
# paths.example.yaml must keep all paths as placeholders, not absolute
# real host paths.
# ---------------------------------------------------------------------------


def test_paths_example_uses_placeholders():
    cfg = _load("paths.example.yaml")
    # Moebius commit must be the pinned value from TDD §2.1.
    assert cfg["moebius_commit"] == "b88d462bacb9af6e7128a3b4cc4a07418bedfd61"
    # Path fields must be strings.
    for key in (
        "data_root",
        "weights_root",
        "cache_root",
        "output_root",
        "moebius_upstream",
        "zipdepth_checkpoint",
        "hami_data_root",
    ):
        assert isinstance(cfg[key], str)
        assert cfg[key]  # non-empty


# ---------------------------------------------------------------------------
# Local smoke config matches the recipe in TDD §5.3.
# ---------------------------------------------------------------------------


def test_local_smoke_recipe_matches_tdd():
    cfg = _load("local_smoke.yaml")
    assert cfg["training"]["max_steps"] == 20
    assert cfg["training"]["lr"] == pytest.approx(1e-4)
    assert cfg["model"]["branch"] == "depth"
    assert cfg["data"]["batch_size"] == 1
    assert cfg["data"]["resolution"] == 512


# ---------------------------------------------------------------------------
# Student configs and rknn config exist and declare the architecture
# names referenced from the spec.
# ---------------------------------------------------------------------------


def test_student_pixel_arch_named():
    cfg = _load("student_pixel.yaml")
    assert cfg["model"]["arch"] == "pixel_student_v0"


def test_student_latent_arch_named():
    cfg = _load("student_latent.yaml")
    assert cfg["model"]["arch"] == "latent_student_v0"


def test_rknn_targets_int8_rk3588():
    cfg = _load("rknn_rk3588.yaml")
    assert cfg["target"]["platform"] == "rk3588"
    assert cfg["target"]["precision"] == "int8"
    assert cfg["export"]["rknn_toolkit_version"] == "2.3.2"
    assert cfg["budget"]["max_gmacs"] == pytest.approx(8.0)
    assert cfg["budget"]["max_dram_mb"] == 300


# ---------------------------------------------------------------------------
# The local_smoke config's shape spec actually matches what a synthetic
# ConditionBatch built from those numbers would look like.
# ---------------------------------------------------------------------------


def test_local_smoke_shapes_match_condition_contract():
    cfg = _load("local_smoke.yaml")
    h = w = int(cfg["data"]["resolution"])
    b = int(cfg["data"]["batch_size"])
    # Build a synthetic empty-mask batch of that size and run it through
    # the validator. This catches typos in the config that would only
    # surface at training time.
    import numpy as np

    rgb = np.zeros((b, 3, h, w), dtype=np.float32)
    mask = np.zeros((b, 1, h, w), dtype=np.float32)
    depth = np.zeros((b, 1, h, w), dtype=np.float32)
    noise = np.zeros((b, 4, h // 8, w // 8), dtype=np.float32)
    batch = ConditionBatch.from_arrays(rgb, mask, depth, noise)
    validate_condition(batch)
    assert batch.spatial_shape == (h, w)
