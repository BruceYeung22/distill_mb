"""Shared fixtures for ``tests/training/teacher``."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import pytest


def _candidate_roots() -> list[Path]:
    roots: list[Path] = []
    env = os.environ.get("MOEBIUS_UPSTREAM_DIR")
    if env:
        roots.append(Path(env))
    roots.extend(
        [
            Path("/mnt/d/project/moebius_distill/Moebius"),
            Path("D:/project/moebius_distill/Moebius"),
        ]
    )
    return roots


def moebius_weight_path() -> Optional[Path]:
    for upstream in _candidate_roots():
        if not upstream.is_dir():
            continue
        for cand in [
            upstream / "weight/Moebius/ft_places2/diffusion_pytorch_model.bin",
            upstream / "pretrained/ft_places2.pt",
        ]:
            if cand.is_file():
                return cand
    return None


@pytest.fixture(scope="session")
def moebius_weights_path() -> Optional[Path]:
    p = moebius_weight_path()
    if p is None:
        pytest.skip(
            f"Moebius weights not found under "
            f"{[str(r) for r in _candidate_roots()]}"
        )
    return p


@pytest.fixture()
def moebius_model(moebius_weights_path):
    """A fresh Moebius RemovalModel per test.

    The smoke tests mutate the model (gradient updates), so we cannot
    share a session-scoped fixture.
    """
    torch = pytest.importorskip("torch")
    from moebius_finetune.teachers.loader import load_removal_model
    return load_removal_model(moebius_weights_path, strict=True)
