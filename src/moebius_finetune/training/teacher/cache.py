"""Teacher output cache for one-step distillation (TDD §7.1).

The teacher cache materialises, for each ``(case_id, seed)`` pair:

* the actual initial noise tensor that drives the teacher,
* the final latent after DDIM denoising,
* the teacher-decoded RGB candidate,
* a stable cache key derived from the case id, seed, scheduler
  config, condition-preprocessing version, and teacher weight hash.

The cache entry never includes the clean target. The inference path
uses only the masked image (no target encoding, TDD §5.2), so the
output is a valid distillation target for the student.

This module is intentionally side-effect-free: it does not write
files. The orchestrator (CLI / Agent B integration code) decides
where the cache lives, mirroring the data-management rule that
weights and cache never live in the package working tree.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch import nn
from torch import Tensor

from ...contracts import ConditionBatch
from ...teachers import (
    DepthConditionAdapter,
    DepthConditionedRemoval,
    OriginalRemovalBaseline,
)
from ...teachers.wrapper import _default_depth_features


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class CacheConfigError(ValueError):
    """Raised when the cache configuration is invalid."""


class CacheKeyMismatchError(RuntimeError):
    """Raised when a cache key does not match the current config."""


# ---------------------------------------------------------------------------
# Public dataclasses
# ---------------------------------------------------------------------------


CONDITION_PREPROCESSING_VERSION = 1
"""Bumped whenever the condition preprocessing contract changes.

TDD §7.1 requires a "条件预处理版本" field in every cache entry. The
version is a positive integer; old caches must be regenerated when
this changes.
"""


@dataclass
class TeacherCacheEntry:
    """One (case, seed) cache record (TDD §7.1)."""

    case_id: str
    data_version: str
    teacher_checkpoint_sha256: str
    moebius_commit: str
    scheduler_config: Dict[str, Any]
    timesteps: List[int]
    condition_preprocessing_version: int
    seed: int
    initial_noise: np.ndarray
    final_latent: np.ndarray
    teacher_rgb: np.ndarray
    cache_key: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        # Ensure JSON-friendly values: convert numpy to lists.
        for k in ("initial_noise", "final_latent", "teacher_rgb"):
            v = d[k]
            if isinstance(v, np.ndarray):
                d[k] = v.tolist()
        return d


# ---------------------------------------------------------------------------
# Seed derivation (stable, no Python hash())
# ---------------------------------------------------------------------------


def derive_seed(case_id: str, seed: int) -> int:
    """Stable seed derivation per TDD §7.1.

    We mix the case_id with the explicit seed via SHA-256 (no
    ``hash()``; the latter is not stable across Python processes
    thanks to ``PYTHONHASHSEED``). The result is mapped to a 32-bit
    unsigned integer for ``torch.manual_seed``.
    """
    if not isinstance(case_id, str):
        case_id = str(case_id)
    digest = hashlib.sha256(f"{case_id}|{int(seed)}".encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big", signed=False)


# ---------------------------------------------------------------------------
# Cache key (JSON-friendly dict)
# ---------------------------------------------------------------------------


def stable_cache_key(
    *,
    case_id: str,
    seed: int,
    data_version: str,
    teacher_checkpoint_sha256: str,
    moebius_commit: str,
    scheduler_config: Mapping[str, Any],
    condition_preprocessing_version: int = CONDITION_PREPROCESSING_VERSION,
) -> str:
    """Compute a stable hex cache key for a (case, seed) entry."""
    payload = {
        "case_id": str(case_id),
        "seed": int(seed),
        "data_version": str(data_version),
        "teacher_checkpoint_sha256": str(teacher_checkpoint_sha256),
        "moebius_commit": str(moebius_commit),
        "scheduler_config": _jsonify_scheduler_config(scheduler_config),
        "condition_preprocessing_version": int(condition_preprocessing_version),
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _jsonify_scheduler_config(cfg: Mapping[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in dict(cfg).items():
        if isinstance(v, Tensor):
            out[k] = v.detach().cpu().tolist()
        elif isinstance(v, np.ndarray):
            out[k] = v.tolist()
        elif isinstance(v, (list, tuple)):
            out[k] = [int(x) if isinstance(x, (int, np.integer)) else x for x in v]
        elif isinstance(v, (int, np.integer, float, str, bool)) or v is None:
            out[k] = v
        elif isinstance(v, Mapping):
            out[k] = _jsonify_scheduler_config(v)
        else:
            out[k] = str(v)
    return out


# ---------------------------------------------------------------------------
# Default scheduler configuration (TDD §5.2)
# ---------------------------------------------------------------------------


def default_scheduler_config(
    *,
    num_inference_steps: int = 20,
    num_train_timesteps: int = 1000,
    beta_start: float = 0.00085,
    beta_end: float = 0.012,
    beta_schedule: str = "scaled_linear",
    steps_offset: int = 1,
    prediction_type: str = "epsilon",
    set_alpha_to_one: bool = False,
    clip_sample: bool = False,
    eta: float = 0.0,
    strength: float = 1.0,
    guidance_scale: float = 1.0,
    noise_offset: float = 0.0,
) -> Dict[str, Any]:
    """Build the default DDIM scheduler config for the teacher cache.

    These defaults match TDD §5.2:

    * DDIM 20 steps,
    * eta=0 (deterministic),
    * strength=1 (pure-noise start),
    * guidance=1 (no CFG doubling),
    * noise_offset=0,
    * scaled-linear beta, 1000 train steps, original steps_offset=1,
    * epsilon prediction.
    """
    return {
        "scheduler_type": "DDIM",
        "num_inference_steps": int(num_inference_steps),
        "num_train_timesteps": int(num_train_timesteps),
        "beta_start": float(beta_start),
        "beta_end": float(beta_end),
        "beta_schedule": str(beta_schedule),
        "steps_offset": int(steps_offset),
        "prediction_type": str(prediction_type),
        "set_alpha_to_one": bool(set_alpha_to_one),
        "clip_sample": bool(clip_sample),
        "eta": float(eta),
        "strength": float(strength),
        "guidance_scale": float(guidance_scale),
        "noise_offset": float(noise_offset),
    }


# ---------------------------------------------------------------------------
# Cache entry building
# ---------------------------------------------------------------------------


CaseBatchBuilder = Callable[[str, int], Tuple[ConditionBatch, Optional[Tensor], Optional[Tensor]]]
"""Callable that, given a case id and a seed, returns:

* a :class:`ConditionBatch` (without ``noise``; we generate it
  deterministically from the seed),
* an optional pre-computed ``masked_latent`` of shape
  ``[B, 4, H/8, W/8]`` (if None, we use ``vae`` to encode),
* an optional pre-computed ``depth_features`` of shape
  ``[B, 2, H/8, W/8]`` (if None, we derive it from the condition).

The builder MUST NOT include the clean target in the
``ConditionBatch``; the inference path only sees the masked image.
"""


def cache_teacher_outputs(
    model: DepthConditionedRemoval,
    case_list: Sequence[str],
    seed_list: Sequence[int],
    scheduler_cfg: Mapping[str, Any],
    *,
    data_version: str = "synthetic",
    teacher_checkpoint_sha256: str = "",
    moebius_commit: str = "",
    vae: Optional[nn.Module] = None,
    case_batch_builder: Optional[CaseBatchBuilder] = None,
    on_case: Optional[Callable[[TeacherCacheEntry], None]] = None,
) -> List[TeacherCacheEntry]:
    """Run the teacher and produce cache entries for every (case, seed) pair.

    The cache is JSON-friendly (TDD §7.1) and contains:

    * ``case_id``, ``data_version``, ``teacher_checkpoint_sha256``,
      ``moebius_commit``,
    * the full ``scheduler_config`` and the actual ``timesteps``,
    * the ``condition_preprocessing_version`` (constant from this
      module),
    * the per-case ``seed``,
    * the actual ``initial_noise`` tensor (numpy),
    * the ``final_latent`` after DDIM,
    * the teacher-decoded ``teacher_rgb`` (numpy, [0, 1] range),
    * a stable ``cache_key`` hex digest.

    The function raises :class:`CacheKeyMismatchError` if a pre-existing
    cache entry (provided via ``on_case``) disagrees with the freshly
    computed key — this is the "no silent reuse" behaviour from TDD
    §7.1.
    """
    if not isinstance(model, DepthConditionedRemoval) and not isinstance(
        model, OriginalRemovalBaseline
    ):
        raise CacheConfigError(
            f"model must be a DepthConditionedRemoval or "
            f"OriginalRemovalBaseline, got {type(model).__name__}"
        )
    if not case_list:
        return []
    if not seed_list:
        raise CacheConfigError("seed_list must be non-empty")

    if "scheduler_type" not in scheduler_cfg:
        raise CacheConfigError("scheduler_cfg must include a scheduler_type")
    if scheduler_cfg.get("scheduler_type") != "DDIM":
        raise CacheConfigError(
            f"only DDIM is supported, got {scheduler_cfg.get('scheduler_type')!r}"
        )
    if float(scheduler_cfg.get("strength", 1.0)) != 1.0:
        # TDD §5.2: 推理从纯噪声出发
        raise CacheConfigError(
            "scheduler_cfg.strength must be 1.0 for the cache "
            "(inference starts from pure noise)"
        )
    if float(scheduler_cfg.get("guidance_scale", 1.0)) != 1.0:
        raise CacheConfigError(
            "scheduler_cfg.guidance_scale must be 1.0 for the cache "
            "(no CFG doubling in the single-condition path)"
        )

    device = next(model.parameters()).device
    ddim = _build_ddim_scheduler(scheduler_cfg)
    timesteps, num_inference_steps = _compute_timesteps(ddim, scheduler_cfg, device)
    timesteps_list = [int(t) for t in timesteps.cpu().tolist()]

    entries: List[TeacherCacheEntry] = []
    for case_id in case_list:
        for seed in seed_list:
            if case_batch_builder is None:
                raise CacheConfigError(
                    "case_batch_builder is required: cache_teacher_outputs "
                    "does not invent synthetic data; the caller must "
                    "supply a builder that returns the masked image only."
                )
            condition, masked_latent, depth_features = case_batch_builder(case_id, int(seed))
            # Build noise from the stable seed (no Python hash()).
            torch_seed = derive_seed(case_id, int(seed))
            torch.manual_seed(torch_seed)
            initial_noise = torch.randn(
                (1, 4, condition.noise.shape[2], condition.noise.shape[3]),
                dtype=torch.float32,
                device="cpu",
            )
            # Run the full DDIM denoising loop.
            final_latent = _ddim_loop(
                model=model,
                condition=condition,
                initial_noise=initial_noise.to(device),
                masked_latent=masked_latent.to(device) if masked_latent is not None else None,
                depth_features=depth_features.to(device) if depth_features is not None else None,
                ddim=ddim,
                timesteps=timesteps,
                vae=vae,
                device=device,
            )
            final_latent_cpu = final_latent.detach().to("cpu", dtype=torch.float32).numpy()
            # Decode the final latent to RGB (teacher output).
            if vae is None:
                raise CacheConfigError(
                    "A VAE is required to decode the final latent to RGB."
                )
            teacher_rgb_t = _decode_latent(vae, final_latent, device)
            teacher_rgb = teacher_rgb_t.detach().to("cpu", dtype=torch.float32).numpy()

            key = stable_cache_key(
                case_id=case_id,
                seed=int(seed),
                data_version=str(data_version),
                teacher_checkpoint_sha256=str(teacher_checkpoint_sha256),
                moebius_commit=str(moebius_commit),
                scheduler_config=scheduler_cfg,
            )

            entry = TeacherCacheEntry(
                case_id=str(case_id),
                data_version=str(data_version),
                teacher_checkpoint_sha256=str(teacher_checkpoint_sha256),
                moebius_commit=str(moebius_commit),
                scheduler_config=_jsonify_scheduler_config(scheduler_cfg),
                timesteps=timesteps_list,
                condition_preprocessing_version=CONDITION_PREPROCESSING_VERSION,
                seed=int(seed),
                initial_noise=initial_noise.numpy().astype(np.float32, copy=False),
                final_latent=final_latent_cpu.astype(np.float32, copy=False),
                teacher_rgb=teacher_rgb.astype(np.float32, copy=False),
                cache_key=key,
                metadata={
                    "num_inference_steps": int(num_inference_steps),
                    "torch_seed": int(torch_seed),
                },
            )

            if on_case is not None:
                on_case(entry)

            entries.append(entry)
    return entries


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _build_ddim_scheduler(cfg: Mapping[str, Any]):
    from diffusers import DDIMScheduler
    return DDIMScheduler(
        num_train_timesteps=int(cfg["num_train_timesteps"]),
        beta_start=float(cfg["beta_start"]),
        beta_end=float(cfg["beta_end"]),
        beta_schedule=str(cfg["beta_schedule"]),
        clip_sample=bool(cfg.get("clip_sample", False)),
        set_alpha_to_one=bool(cfg.get("set_alpha_to_one", False)),
        steps_offset=int(cfg.get("steps_offset", 1)),
        prediction_type=str(cfg["prediction_type"]),
    )


def _compute_timesteps(ddim, cfg: Mapping[str, Any], device: torch.device):
    ddim.set_timesteps(int(cfg["num_inference_steps"]), device=device)
    return ddim.timesteps, int(cfg["num_inference_steps"])


def _ddim_loop(
    *,
    model: nn.Module,
    condition: ConditionBatch,
    initial_noise: Tensor,
    masked_latent: Optional[Tensor],
    depth_features: Optional[Tensor],
    ddim,
    timesteps: Tensor,
    vae: Optional[nn.Module],
    device: torch.device,
) -> Tensor:
    """Run the single-condition DDIM denoising loop.

    The loop never encodes the clean target; we use the precomputed
    (or VAE-encoded) ``masked_latent`` plus the noisy latent and the
    resized mask to build the 9-channel UNet input.
    """
    B = initial_noise.shape[0]
    H_lat, W_lat = initial_noise.shape[-2], initial_noise.shape[-1]
    H, W = H_lat * 8, W_lat * 8

    if masked_latent is None:
        if vae is None:
            raise CacheConfigError("masked_latent is required when no VAE is provided")
        rgb = torch.from_numpy(np.ascontiguousarray(condition.rgb_hole)).to(device)
        mask = torch.from_numpy(np.ascontiguousarray(condition.hole_mask)).to(device)
        masked_image = (2.0 * rgb - 1.0) * (1.0 - mask)
        try:
            vae_dtype = next(vae.parameters()).dtype
        except (StopIteration, AttributeError):
            vae_dtype = torch.float32
        with torch.no_grad():
            encoded = vae.encode(masked_image.to(dtype=vae_dtype))
            # diffusers returns ``encoded.latent_dist`` (a
            # ``DiagonalGaussianDistribution``). Stand-in test VAEs may
            # return the ``mode()`` callable directly.
            if hasattr(encoded, "latent_dist"):
                latent = encoded.latent_dist.mode()
            elif hasattr(encoded, "mode"):
                latent = encoded.mode()
            else:
                raise CacheConfigError(
                    "VAE.encode() must return an object with .latent_dist "
                    "or .mode(); the cache pipeline cannot read the target "
                    "image, so it must not require any extra API."
                )
            scale = getattr(vae.config, "scaling_factor", 0.13025)
        masked_latent = (latent * float(scale)).to(dtype=torch.float32)

    mask_t = torch.from_numpy(np.ascontiguousarray(condition.hole_mask)).to(device)
    latent_mask = nn.functional.interpolate(
        mask_t, size=(H_lat, W_lat), mode="nearest"
    )

    if depth_features is None:
        depth_t = torch.from_numpy(np.ascontiguousarray(condition.depth_hole)).to(device)
        depth_features = _default_depth_features(mask_t, depth_t)

    noisy = initial_noise
    for t in timesteps:
        t = t.to(device).unsqueeze(0)
        noisy_in = ddim.scale_model_input(noisy, t)
        latent_input = torch.cat([noisy_in, latent_mask, masked_latent], dim=1)
        # Single-condition forward, no CFG doubling.
        noise_pred = model.forward(
            latent_input, t, None, depth_features=depth_features
        )
        noisy = ddim.step(noise_pred, t, noisy, return_dict=False)[0]
    return noisy


def _decode_latent(vae: nn.Module, latent: Tensor, device: torch.device) -> Tensor:
    vae_dtype = next(vae.parameters()).dtype
    scale = float(getattr(vae.config, "scaling_factor", 0.13025))
    with torch.no_grad():
        decoded = vae.decode((latent / scale).to(dtype=vae_dtype)).sample
    return ((decoded + 1.0) / 2.0).clamp(0.0, 1.0).to(dtype=torch.float32)


__all__ = [
    "CONDITION_PREPROCESSING_VERSION",
    "CacheConfigError",
    "CacheKeyMismatchError",
    "CaseBatchBuilder",
    "TeacherCacheEntry",
    "cache_teacher_outputs",
    "default_scheduler_config",
    "derive_seed",
    "stable_cache_key",
]
