"""Shared fixtures for the ``tests/teachers`` package.

Heavy ML dependencies (torch, diffusers) are imported lazily inside
fixtures so that simply collecting the tests does not pull them in.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterator, Optional

import pytest


# ---------------------------------------------------------------------------
# Moebius upstream path
# ---------------------------------------------------------------------------


def _candidate_roots() -> list[Path]:
    roots: list[Path] = []
    env = os.environ.get("MOEBIUS_UPSTREAM_DIR")
    if env:
        roots.append(Path(env))
    # The dev box's Moebius checkout. The DGX Spark box keeps it at
    # /home/dog/project/moebius_distill/Moebius; the older WSL box used
    # the /mnt/d or D:/ forms.
    roots.extend(
        [
            Path("/home/dog/project/moebius_distill/Moebius"),
            Path("/mnt/d/project/moebius_distill/Moebius"),
            Path("D:/project/moebius_distill/Moebius"),
            Path("D:/project/Moebius"),
            Path("D:/moebius_distill/Moebius"),
        ]
    )
    return roots


def moebius_upstream_dir() -> Path:
    for root in _candidate_roots():
        if root.is_dir():
            return root
    return _candidate_roots()[0]


def moebius_weight_path() -> Optional[Path]:
    """Resolve the on-disk weight file for the tests.

    ``MOEBIUS_WEIGHTS_PATH`` overrides everything when set. Otherwise we
    probe the layouts we have seen on real boxes:

    * ``weights/moebius/pretrained/diffusion_pytorch_model.bin`` — the
      pretrained checkpoint as checked out on the DGX Spark box
      (aarch64/CUDA 13).
    * ``weight/Moebius/ft_places2/diffusion_pytorch_model.bin`` — the
      upstream release layout documented in the Moebius README.
    * ``pretrained/ft_places2.pt`` — the earlier spec's path; the same
      checkpoint under a different name.
    """
    override = os.environ.get("MOEBIUS_WEIGHTS_PATH")
    if override and Path(override).is_file():
        return Path(override)
    for upstream in _candidate_roots():
        if not upstream.is_dir():
            continue
        candidates = [
            upstream / "weights/moebius/pretrained/diffusion_pytorch_model.bin",
            upstream / "weight/Moebius/ft_places2/diffusion_pytorch_model.bin",
            upstream / "pretrained/ft_places2.pt",
            upstream / "weight/Moebius/ft_places2/ft_places2.pt",
        ]
        for cand in candidates:
            if cand.is_file():
                return cand
    return None


@pytest.fixture(scope="session")
def moebius_weights_path() -> Optional[Path]:
    p = moebius_weight_path()
    if p is None:
        pytest.skip(
            f"Moebius weight file not found under "
            f"{[str(r) for r in _candidate_roots()]}"
        )
    return p


@pytest.fixture(scope="session")
def moebius_model(moebius_weights_path):
    """The strict-loaded Moebius RemovalModel, shared across the session.

    Loading the 226M-param model takes ~30s on the dev box, so the
    wrapper / baseline / smoke tests share this fixture.
    """
    torch = pytest.importorskip("torch")
    from moebius_finetune.teachers.loader import load_removal_model
    return load_removal_model(moebius_weights_path, strict=True)
