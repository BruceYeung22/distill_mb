"""Public contracts for the moebius-finetune package.

This module defines the typed condition-batch, sample manifest, validation
and helper utilities shared by all subpackages (data, teachers, students,
training, evaluation, deployment). It is intentionally pure-numpy +
stdlib at module top level: importing it MUST NOT pull in torch, diffusers,
transformers, onnx, or rknn. Heavier dependencies are deferred to the
narrow functions that actually need them (e.g. :func:`to_torch`).

The contract is described in §4.1 and §4.2 of
``tdd/moebius-depth-finetune-distill-2026-09-12-05-38-54.md``.
"""

from __future__ import annotations

import dataclasses
import enum
import importlib
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional, Tuple

import numpy as np


__all__ = [
    "ConditionBatch",
    "ConditionContractError",
    "CoordFrame",
    "DepthNormalization",
    "Direction",
    "SampleManifest",
    "Split",
    "inpaint",
    "to_torch",
    "validate_condition",
]


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class ConditionContractError(ValueError):
    """Raised when a condition batch or sample manifest violates the contract.

    This is a ``ValueError`` subclass so generic validation callers can still
    catch ``ValueError``; downstream code should prefer catching
    :class:`ConditionContractError` so they only react to contract issues
    and not unrelated value errors.
    """


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class Split(str, enum.Enum):
    """Dataset split identifier (Hami-style; see TDD §4.2)."""

    TRAIN = "train"
    HOLDOUT_MASKS = "holdout_masks"
    HOLDOUT_IMAGES = "holdout_images"


class Direction(str, enum.Enum):
    """Source-side projection direction used to synthesise the GRT hole."""

    L2R = "L2R"
    R2L = "R2L"


class CoordFrame(str, enum.Enum):
    """Coordinate frame in which the sample's tensors live.

    First version of the project fixes everything to ``SOURCE``; ``TARGET``
    is reserved for future use.
    """

    SOURCE = "source"
    TARGET = "target"


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class DepthNormalization:
    """Per-image depth normalisation summary kept alongside the sample.

    ``p99`` is the 99-th percentile of the source-frame inverse depth used
    to bring the float depth map into a signed L-negative / R-positive
    representation fed to the network. ``sign_convention`` is a string
    that downstream code can pattern-match on; only the ``"L_neg_R_pos"``
    convention is supported in the first version.
    """

    p99: float
    sign_convention: str = "L_neg_R_pos"


@dataclass(frozen=True)
class ConditionBatch:
    """A batch of conditions for the inpainting network.

    See TDD §4.1 for the precise shapes, dtypes and value conventions. The
    constructor performs no validation; build via :meth:`from_arrays` or
    call :func:`validate_condition` explicitly.
    """

    rgb_hole: np.ndarray
    hole_mask: np.ndarray
    depth_hole: np.ndarray
    noise: np.ndarray

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------
    @classmethod
    def from_arrays(
        cls,
        rgb_hole: np.ndarray,
        hole_mask: np.ndarray,
        depth_hole: np.ndarray,
        noise: np.ndarray,
    ) -> "ConditionBatch":
        """Build a batch and run :func:`validate_condition` automatically."""
        batch = cls(
            rgb_hole=rgb_hole,
            hole_mask=hole_mask,
            depth_hole=depth_hole,
            noise=noise,
        )
        validate_condition(batch)
        return batch

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------
    @property
    def batch_size(self) -> int:
        return int(self.rgb_hole.shape[0])

    @property
    def spatial_shape(self) -> Tuple[int, int]:
        return (int(self.rgb_hole.shape[2]), int(self.rgb_hole.shape[3]))

    def to_dict(self) -> Dict[str, np.ndarray]:
        return {
            "rgb_hole": self.rgb_hole,
            "hole_mask": self.hole_mask,
            "depth_hole": self.depth_hole,
            "noise": self.noise,
        }


@dataclass
class SampleManifest:
    """Per-case manifest entry (see TDD §4.2).

    ``coord_frame`` is fixed to ``SOURCE`` in the first version. The
    ``depth_normalization`` is stored alongside the sample so that loaders
    do not silently re-derive it.
    """

    case_id: str
    source_id: str
    split: Split
    coord_frame: CoordFrame
    rgb_path: str
    depth_path: str
    mask_path: str
    target_path: str
    direction: Direction
    dmax_px: int
    depth_normalization: DepthNormalization

    # ------------------------------------------------------------------
    # Round-trip helpers
    # ------------------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        d = dataclasses.asdict(self)
        d["split"] = self.split.value
        d["coord_frame"] = self.coord_frame.value
        d["direction"] = self.direction.value
        return d

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SampleManifest":
        d = dict(data)
        d["split"] = Split(d["split"])
        d["coord_frame"] = CoordFrame(d["coord_frame"])
        d["direction"] = Direction(d["direction"])
        dn = d.get("depth_normalization")
        if isinstance(dn, Mapping):
            d["depth_normalization"] = DepthNormalization(**dn)
        elif dn is None:
            d["depth_normalization"] = DepthNormalization(p99=1.0)
        return cls(**d)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _as_float32_array(name: str, arr: np.ndarray) -> np.ndarray:
    if not isinstance(arr, np.ndarray):
        raise ConditionContractError(
            f"{name} must be a numpy.ndarray, got {type(arr).__name__}"
        )
    if arr.dtype != np.float32:
        raise ConditionContractError(
            f"{name} must be float32, got {arr.dtype}"
        )
    if not np.all(np.isfinite(arr)):
        raise ConditionContractError(f"{name} contains non-finite values")
    return arr


def validate_condition(batch: ConditionBatch) -> None:
    """Validate a :class:`ConditionBatch` against the TDD §4.1 contract.

    Checks performed:

    * All four tensors are contiguous ``float32`` numpy arrays.
    * Shapes: ``rgb_hole`` and ``depth_hole`` are ``[B, 3, H, W]`` /
      ``[B, 1, H, W]``; ``hole_mask`` is ``[B, 1, H, W]`` and ``noise`` is
      ``[B, 4, H/8, W/8]``. B, H, W match across the condition tensors.
    * ``hole_mask`` only contains values in ``{0, 1}`` (as floats).
    * Pixels where ``hole_mask == 1`` have ``rgb_hole`` and ``depth_hole``
      exactly equal to 0.
    """
    rgb = _as_float32_array("rgb_hole", batch.rgb_hole)
    mask = _as_float32_array("hole_mask", batch.hole_mask)
    depth = _as_float32_array("depth_hole", batch.depth_hole)
    noise = _as_float32_array("noise", batch.noise)

    if rgb.ndim != 4:
        raise ConditionContractError(
            f"rgb_hole must be 4D [B,3,H,W], got shape {rgb.shape}"
        )
    if mask.ndim != 4:
        raise ConditionContractError(
            f"hole_mask must be 4D [B,1,H,W], got shape {mask.shape}"
        )
    if depth.ndim != 4:
        raise ConditionContractError(
            f"depth_hole must be 4D [B,1,H,W], got shape {depth.shape}"
        )
    if noise.ndim != 4:
        raise ConditionContractError(
            f"noise must be 4D [B,4,H/8,W/8], got shape {noise.shape}"
        )

    b, c, h, w = rgb.shape
    if c != 3:
        raise ConditionContractError(
            f"rgb_hole channel dim must be 3, got {c} (shape={rgb.shape})"
        )
    if (b, 1, h, w) != mask.shape:
        raise ConditionContractError(
            f"hole_mask shape {mask.shape} incompatible with rgb_hole {rgb.shape}"
        )
    if (b, 1, h, w) != depth.shape:
        raise ConditionContractError(
            f"depth_hole shape {depth.shape} incompatible with rgb_hole {rgb.shape}"
        )
    expected_noise = (b, 4, h // 8, w // 8)
    if noise.shape != expected_noise:
        raise ConditionContractError(
            f"noise shape {noise.shape} incompatible with rgb_hole {rgb.shape}; "
            f"expected {expected_noise}"
        )
    if (h % 8) != 0 or (w % 8) != 0:
        raise ConditionContractError(
            f"spatial dims H={h}, W={w} must be multiples of 8 (VAE 8x downsample)"
        )

    # hole_mask values must be in {0, 1}
    mask_min = float(mask.min())
    mask_max = float(mask.max())
    if mask_min < 0.0 or mask_max > 1.0:
        raise ConditionContractError(
            f"hole_mask values must be in [0, 1], got [{mask_min}, {mask_max}]"
        )
    if not np.all((mask == 0.0) | (mask == 1.0)):
        raise ConditionContractError("hole_mask must contain only 0 or 1 values")

    # Hole pixels must have rgb_hole == 0 and depth_hole == 0
    if mask.any():
        rgb_in_hole_max = float(np.abs(rgb * mask).max())
        if rgb_in_hole_max != 0.0:
            raise ConditionContractError(
                f"rgb_hole must be exactly 0 inside the hole, got max |rgb_hole*mask|={rgb_in_hole_max}"
            )
        depth_in_hole_max = float(np.abs(depth * mask).max())
        if depth_in_hole_max != 0.0:
            raise ConditionContractError(
                f"depth_hole must be exactly 0 inside the hole, got max |depth_hole*mask|={depth_in_hole_max}"
            )


# ---------------------------------------------------------------------------
# inpaint / to_torch
# ---------------------------------------------------------------------------


def inpaint(condition: ConditionBatch, candidate: np.ndarray) -> np.ndarray:
    """Compose the final inpainted RGB image from condition + candidate.

    Behaviour (TDD §4.1):

    * ``candidate`` is the network's output for the hole region; shape must
      be ``[B, 3, H, W]`` matching ``condition``. The caller does the
      network forward; this function only handles the deterministic
      composition.
    * The known region is taken from ``condition.rgb_hole`` (i.e. the
      non-zero pixels of ``condition.rgb_hole``). The hole is filled with
      ``candidate`` clamped to ``[0, 1]``.
    * ``condition`` is validated; an empty mask simply returns the input.
    """
    if not isinstance(candidate, np.ndarray):
        raise ConditionContractError(
            f"candidate must be a numpy.ndarray, got {type(candidate).__name__}"
        )
    if candidate.dtype != np.float32:
        raise ConditionContractError(
            f"candidate must be float32, got {candidate.dtype}"
        )
    if candidate.shape != condition.rgb_hole.shape:
        raise ConditionContractError(
            f"candidate shape {candidate.shape} does not match rgb_hole "
            f"shape {condition.rgb_hole.shape}"
        )
    if not np.all(np.isfinite(candidate)):
        raise ConditionContractError("candidate contains non-finite values")

    validate_condition(condition)

    rgb = condition.rgb_hole
    mask = condition.hole_mask  # [B,1,H,W] in {0,1}
    clamped = np.clip(candidate, 0.0, 1.0).astype(np.float32, copy=False)
    # known region: (1 - mask) * rgb_hole
    return ((1.0 - mask) * rgb + mask * clamped).astype(np.float32, copy=False)


def to_torch(batch: ConditionBatch):
    """Convert a :class:`ConditionBatch` into a torch ``Tensor`` bundle.

    Torch is imported lazily inside this function so that simply importing
    :mod:`moebius_finetune.contracts` does not require torch. If torch is
    not installed a :class:`ConditionContractError` is raised with a
    descriptive message.

    Returns
    -------
    dict
        Mapping with keys ``rgb_hole``, ``hole_mask``, ``depth_hole`` and
        ``noise`` mapping to ``torch.Tensor`` of dtype ``float32``.
    """
    try:
        import torch  # type: ignore
    except ImportError as exc:  # pragma: no cover - import guard
        raise ConditionContractError(
            "to_torch requires torch, but it is not importable in this "
            "environment. Install the package's torch extra or the local "
            "training environment before using to_torch."
        ) from exc

    validate_condition(batch)

    def _to(arr: np.ndarray):
        return torch.from_numpy(np.ascontiguousarray(arr))

    return {
        "rgb_hole": _to(batch.rgb_hole),
        "hole_mask": _to(batch.hole_mask),
        "depth_hole": _to(batch.depth_hole),
        "noise": _to(batch.noise),
    }
