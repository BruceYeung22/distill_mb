"""Tests for :mod:`moebius_finetune.teachers.loader`."""

from __future__ import annotations

import hashlib

import pytest

from moebius_finetune.teachers.loader import (
    MOEBIUS_PINNED_COMMIT,
    WeightHashMismatchError,
    WeightLoadError,
    get_weight_metadata,
    load_removal_model,
    sha256_of_file,
    verify_state_dict_hash,
)

from .conftest import moebius_weight_path


pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")


# ---------------------------------------------------------------------------
# Pinned commit
# ---------------------------------------------------------------------------


def test_moebius_pinned_commit_is_frozen():
    """TDD §2.1 — the Moebius commit must not drift in the loader."""
    assert MOEBIUS_PINNED_COMMIT == "b88d462bacb9af6e7128a3b4cc4a07418bedfd61"


# ---------------------------------------------------------------------------
# Strict load
# ---------------------------------------------------------------------------


def test_load_removal_model_strict_succeeds(moebius_weights_path):
    model = load_removal_model(moebius_weights_path, strict=True)
    n_params = sum(p.numel() for p in model.parameters())
    # TDD §2.1: 226,041,531 parameters
    assert n_params == 226_041_531
    # The model should be in eval() mode after load.
    assert not model.training


def test_load_removal_model_rejects_corrupt_state_dict(tmp_path):
    """A weight file with a different key set must fail strictly."""
    # Build a tiny fake state_dict that does NOT match the Moebius
    # model at all.
    fake = tmp_path / "fake.bin"
    import torch
    torch.save({"some.random.key": torch.zeros(1)}, str(fake))
    with pytest.raises((WeightLoadError, RuntimeError)):
        load_removal_model(fake, strict=True)


def test_load_removal_model_strict_false_emits_warning(moebius_weights_path, recwarn):
    """A non-strict load must emit a warning, not silently pass.

    The spec says strict=True is the default and ``strict=False`` is
    only for ablation; warnings make that explicit.
    """
    load_removal_model(moebius_weights_path, strict=False)
    msgs = [str(w.message) for w in recwarn.list
            if "load_removal_model(strict=False)" in str(w.message)]
    assert any("ablation" in m for m in msgs), msgs


# ---------------------------------------------------------------------------
# SHA-256 verification
# ---------------------------------------------------------------------------


def test_sha256_of_file_matches_hashlib(moebius_weights_path):
    expected = hashlib.sha256(moebius_weights_path.read_bytes()).hexdigest()
    assert sha256_of_file(moebius_weights_path) == expected


def test_verify_state_dict_hash_accepts_correct_hash(moebius_weights_path):
    expected = sha256_of_file(moebius_weights_path)
    verify_state_dict_hash(moebius_weights_path, expected)


def test_verify_state_dict_hash_rejects_mismatch(moebius_weights_path):
    with pytest.raises(WeightHashMismatchError):
        verify_state_dict_hash(moebius_weights_path, "0" * 64)


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------


def test_get_weight_metadata_shape(moebius_weights_path):
    meta = get_weight_metadata(moebius_weights_path)
    assert meta["moebius_commit"] == MOEBIUS_PINNED_COMMIT
    assert meta["param_count"] == 226_041_531
    assert meta["size_bytes"] == moebius_weights_path.stat().st_size
    assert isinstance(meta["sha256"], str)
    assert len(meta["sha256"]) == 64
    assert meta["sha256"] == sha256_of_file(moebius_weights_path)


def test_get_weight_metadata_rejects_mismatch(moebius_weights_path):
    with pytest.raises(WeightHashMismatchError):
        get_weight_metadata(moebius_weights_path, expected_sha256="0" * 64)


# ---------------------------------------------------------------------------
# Missing file
# ---------------------------------------------------------------------------


def test_load_missing_file_raises(tmp_path):
    missing = tmp_path / "does-not-exist.bin"
    with pytest.raises(WeightLoadError):
        load_removal_model(missing, strict=True)


def test_sha256_of_missing_file_raises(tmp_path):
    missing = tmp_path / "does-not-exist.bin"
    with pytest.raises(WeightLoadError):
        sha256_of_file(missing)
