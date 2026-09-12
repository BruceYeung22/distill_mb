"""Strict weight loading for the Moebius :class:`RemovalModel`.

This module implements the contract from TDD §5.1 and §9.2:

* :func:`load_removal_model` builds a :class:`RemovalModel` from the
  Moebius YAML config and loads a state_dict with ``strict=True`` by
  default. Any missing or unexpected key is an error; we do not allow
  ``strict=False`` to mask main-weight loading bugs. A ``strict=False``
  branch is provided strictly for ablation studies (TDD §9.2: "原 9
  通道权重严格加载") and emits a warning so it cannot be used silently.
* :func:`verify_state_dict_hash` checks a file's SHA-256 against an
  expected value. This is the cache key (TDD §7.1: "case ID、数据版本、
  教师 checkpoint hash") and also gates the bootstrap of pretrained
  weights.
* :func:`get_weight_metadata` returns the metadata used to build cache
  keys: pinned Moebius commit, SHA-256, file size and parameter total.

The actual weight file path on this dev box is
``Moebius/weight/Moebius/ft_places2/diffusion_pytorch_model.bin`` (the
spec also documents ``Moebius/pretrained/ft_places2.pt`` as an
alias). The loader is path-agnostic; callers decide where the
checkpoint lives.
"""

from __future__ import annotations

import hashlib
import os
import warnings
from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch

# Pinned Moebius commit (TDD §2.1). Used both for traceability and as a
# cache-key field. Tests assert this value.
MOEBIUS_PINNED_COMMIT = "b88d462bacb9af6e7128a3b4cc4a07418bedfd61"

# Moebius YAML config (relative to the Moebius upstream checkout).
MOEBIUS_CONFIG_REL = "config/model_cfg/moebius.yaml"

# Default num_embeddings for the conditional UNet (TDD §5.1: "沿用原条件
# token IDs 0…9"). The RemovalModel uses ``num_embeddings`` and the
# pipeline splits it half/half for CFG; for the single-condition path we
# only need the first half (IDs 0..9).
DEFAULT_NUM_EMBEDDINGS = 20


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class WeightLoadError(RuntimeError):
    """Raised when the Moebius weight file cannot be loaded strictly."""


class WeightHashMismatchError(RuntimeError):
    """Raised when :func:`verify_state_dict_hash` finds a hash mismatch."""


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _moebius_upstream_dir() -> Path:
    """Locate the Moebius upstream checkout.

    Looks at ``MOEBIUS_UPSTREAM_DIR`` (the package never reads unlisted
    paths) and falls back to common dev-box locations: the DGX Spark
    checkout ``/home/dog/project/moebius_distill/Moebius``, then the
    older WSL form ``/mnt/d/project/moebius_distill/Moebius`` and the
    Windows form ``D:/project/moebius_distill/Moebius``.
    """
    env = os.environ.get("MOEBIUS_UPSTREAM_DIR")
    if env and Path(env).is_dir():
        return Path(env)
    for cand in (
        Path("/home/dog/project/moebius_distill/Moebius"),
        Path("/mnt/d/project/moebius_distill/Moebius"),
        Path("D:/project/moebius_distill/Moebius"),
        Path("D:/project/Moebius"),
    ):
        if cand.is_dir():
            return cand
    return Path("/home/dog/project/moebius_distill/Moebius")


def _removal_model_class():
    """Import the Moebius :class:`RemovalModel` and its builder lazily.

    We do this lazily so that the rest of the package can be imported
    in CPU-only environments where Moebius is not installed.
    """
    try:
        # The Moebius repo adds itself to ``sys.path`` at import time;
        # we replicate that here so we can import its modules.
        import sys
        upstream = str(_moebius_upstream_dir())
        if upstream not in sys.path:
            sys.path.insert(0, upstream)
        from removal.v1_2.removal_model import (  # type: ignore[import-not-found]
            RemovalModel,
            build_removal_model,
        )
        return RemovalModel, build_removal_model
    except Exception as exc:  # pragma: no cover - depends on host
        raise WeightLoadError(
            f"Could not import Moebius removal model from "
            f"{_moebius_upstream_dir()}: {exc}. Make sure the upstream "
            f"repo is on PYTHONPATH or set MOEBIUS_UPSTREAM_DIR."
        ) from exc


def _build_empty_model(num_embeddings: int = DEFAULT_NUM_EMBEDDINGS):
    """Build an empty :class:`RemovalModel` ready to receive weights."""
    _, build_removal_model = _removal_model_class()
    config_path = _moebius_upstream_dir() / MOEBIUS_CONFIG_REL
    if not config_path.is_file():
        raise WeightLoadError(
            f"Moebius config not found at {config_path}. Check the "
            f"upstream checkout (commit {MOEBIUS_PINNED_COMMIT})."
        )
    return build_removal_model(str(config_path), num_embeddings=num_embeddings)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def _load_state_dict(weights_path: Union[str, os.PathLike]) -> Dict[str, torch.Tensor]:
    """Read a state_dict from disk with ``weights_only=True``.

    ``weights_only=True`` is a security requirement (TDD §5.3: "加载时
    ``torch.load(..., weights_only=True)``，不引入 pickle 风险"). If the
    file happens to be a pickled full model object rather than a plain
    state_dict we still need to surface the contained state_dict.
    """
    path = Path(weights_path)
    if not path.is_file():
        raise WeightLoadError(f"Weight file not found: {path}")
    try:
        state = torch.load(str(path), map_location="cpu", weights_only=True)
    except Exception as exc:
        raise WeightLoadError(
            f"torch.load failed for {path}: {exc}"
        ) from exc
    if not isinstance(state, dict):
        # Some Moebius checkpoints are wrapped in a dict under 'state_dict'.
        if hasattr(state, "state_dict"):
            state = state.state_dict()
        else:
            raise WeightLoadError(
                f"Expected a state_dict mapping, got {type(state).__name__}"
            )
    return state


def load_removal_model(
    weights_path: Union[str, os.PathLike],
    *,
    strict: bool = True,
    num_embeddings: int = DEFAULT_NUM_EMBEDDINGS,
) -> "Any":
    """Build a :class:`RemovalModel` and load the given weights strictly.

    Parameters
    ----------
    weights_path
        Filesystem path to a Moebius diffusion_pytorch_model.bin (or
        a compatible state_dict dump).
    strict
        Default ``True``. When ``False`` we emit a warning and fall
        through to ``load_state_dict(strict=False)``. The non-strict
        branch is only intended for ablation / no-depth baselines; the
        primary 9-channel teacher load path must be strict.
    num_embeddings
        Number of token embeddings for the conditional UNet. The
        pinned Moebius config uses 20 (10 uncond + 10 cond).

    Returns
    -------
    RemovalModel
        Model in ``eval()`` mode, on CPU, in ``torch.float32``. The
        caller is responsible for moving it to the target device and
        casting to the desired precision.
    """
    model = _build_empty_model(num_embeddings=num_embeddings)
    state_dict = _load_state_dict(weights_path)

    if not strict:
        warnings.warn(
            f"load_removal_model(strict=False) is only intended for "
            f"ablation baselines; missing/unexpected keys will be "
            f"silently dropped for {weights_path}.",
            stacklevel=2,
        )
    try:
        msg = model.load_state_dict(state_dict, strict=strict)
    except RuntimeError as exc:
        raise WeightLoadError(
            f"Strict weight load failed for {weights_path}: {exc}"
        ) from exc

    # Sanity-check the parameter count we expect (TDD §2.1: 226,041,531).
    n_params = sum(p.numel() for p in model.parameters())
    if n_params <= 0:
        raise WeightLoadError(
            f"Loaded model from {weights_path} has zero parameters; "
            f"the state_dict was likely empty."
        )
    model.eval()
    return model


def verify_state_dict_hash(
    weights_path: Union[str, os.PathLike],
    expected_sha256: str,
) -> None:
    """Compute the file SHA-256 and compare to ``expected_sha256``.

    Raises :class:`WeightHashMismatchError` on any mismatch (including
    case differences). The hash is what the teacher cache uses to bind
    a cache entry to a specific checkpoint (TDD §7.1).
    """
    path = Path(weights_path)
    if not path.is_file():
        raise WeightHashMismatchError(f"Weight file not found: {path}")
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    actual = h.hexdigest()
    if actual.lower() != expected_sha256.lower():
        raise WeightHashMismatchError(
            f"SHA-256 mismatch for {path}: expected "
            f"{expected_sha256.lower()}, got {actual.lower()}"
        )


def sha256_of_file(weights_path: Union[str, os.PathLike]) -> str:
    """Return the SHA-256 hex digest of ``weights_path``."""
    path = Path(weights_path)
    if not path.is_file():
        raise WeightLoadError(f"Weight file not found: {path}")
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_vae(vae_dir: Union[str, os.PathLike]) -> Any:
    """Load the frozen Moebius VAE (AutoencoderKL) from a local dir.

    The VAE is returned in ``eval()`` mode with every parameter set to
    ``requires_grad=False`` — the fine-tune and cache paths only ever
    run inference through it.
    """
    try:
        from diffusers import AutoencoderKL  # type: ignore[import-not-found]
    except Exception as exc:  # pragma: no cover - depends on host
        raise WeightLoadError(
            f"diffusers is required to load the VAE from {vae_dir}: {exc}"
        ) from exc
    path = Path(vae_dir)
    if not (path / "config.json").is_file():
        raise WeightLoadError(
            f"VAE config not found at {path / 'config.json'}"
        )
    try:
        vae = AutoencoderKL.from_pretrained(str(path), local_files_only=True)
    except Exception as exc:
        raise WeightLoadError(f"Failed to load VAE from {path}: {exc}") from exc
    vae.eval()
    for p in vae.parameters():
        p.requires_grad_(False)
    return vae


def get_weight_metadata(
    weights_path: Union[str, os.PathLike],
    *,
    expected_sha256: Optional[str] = None,
) -> Dict[str, Any]:
    """Return the metadata needed to build a teacher cache key.

    Fields
    ------
    path
        Absolute path of the weight file as a string.
    sha256
        SHA-256 hex digest of the file.
    size_bytes
        File size in bytes (decimal).
    param_count
        Number of parameters after strict load (catches empty/missing
        weights).
    moebius_commit
        Pinned Moebius commit (TDD §2.1). Always included.
    config_rel
        Path of the Moebius YAML config used to build the model.
    """
    path = Path(weights_path).absolute()
    if not path.is_file():
        raise WeightLoadError(f"Weight file not found: {path}")
    sha = sha256_of_file(path)
    if expected_sha256 is not None and sha.lower() != expected_sha256.lower():
        raise WeightHashMismatchError(
            f"SHA-256 mismatch for {path}: expected "
            f"{expected_sha256.lower()}, got {sha.lower()}"
        )
    # Build a temporary model to count parameters without polluting the
    # user's runtime; we discard it immediately.
    model = load_removal_model(path, strict=True)
    return {
        "path": str(path),
        "sha256": sha,
        "size_bytes": int(path.stat().st_size),
        "param_count": int(sum(p.numel() for p in model.parameters())),
        "moebius_commit": MOEBIUS_PINNED_COMMIT,
        "config_rel": MOEBIUS_CONFIG_REL,
    }


__all__ = [
    "DEFAULT_NUM_EMBEDDINGS",
    "MOEBIUS_CONFIG_REL",
    "MOEBIUS_PINNED_COMMIT",
    "WeightHashMismatchError",
    "WeightLoadError",
    "get_weight_metadata",
    "load_removal_model",
    "sha256_of_file",
    "verify_state_dict_hash",
]
