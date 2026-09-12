"""512-data preparation pipeline.

The pipeline takes a single COCO source image and produces a
fully-materialised 512² case:

1. Load the source RGB at 512×512 (PIL BILINEAR).
2. Run ZipDepth (via the cache adapter) to get a per-pixel
   inverse depth.
3. Compute the per-image p99 and freeze the
   :class:`DepthNormalization`.
4. Build the GRT mask **at 512 resolution** from the cached
   disparity (not by upsampling a 320 mask).
5. Generate ``rgb_hole`` (RGB with the mask region zeroed) and
   ``depth_hole`` (signed normalised depth with the mask region
   zeroed).
6. Persist the five on-disk artefacts and a
   :class:`SampleManifest` whose paths are all **relative** to the
   configured data root.

The pipeline is *deterministic* (no global random state) and is
safe to run in parallel. The ZipDepth cache is treated as the
ground-truth disparity source; if the file is missing, the
pipeline raises :class:`FileNotFoundError` with a clear hint to
run the cache step first.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Union

import numpy as np
from PIL import Image

from ..contracts import (
    ConditionContractError,
    CoordFrame,
    DepthNormalization,
    Direction,
    SampleManifest,
    Split,
)
from .depth_normalization import DepthNormalizationApplier
from .grt import compute_grt_mask
from .zipdepth_adapter import ZipDepthAdapter


__all__ = [
    "build_512_sample",
    "prepare_512_sample",
]


#: Default COCO image subdirectory. Matches Hami's junction.
_DEFAULT_COCO_SUBDIR_TRAIN = "coco_train2017"
_DEFAULT_COCO_SUBDIR_VAL = "coco_val2017"


def _load_rgb01(path: Path, size: int) -> np.ndarray:
    """Load a JPEG/PNG as ``(3, size, size)`` float32 in ``[0, 1]``."""
    with Image.open(path) as source:
        rgb = source.convert("RGB").resize(
            (size, size), Image.Resampling.BILINEAR
        )
    arr = np.asarray(rgb, dtype=np.float32) / np.float32(255.0)
    return arr.transpose(2, 0, 1).astype(np.float32, copy=False)


def _save_uint8(path: Path, arr: np.ndarray) -> None:
    """Save a uint8 ``(H, W)`` or ``(H, W, 3)`` array as PNG/JPEG."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if arr.ndim == 3 and arr.shape[0] in (1, 3):
        # Convert (C, H, W) -> (H, W, C)
        arr = arr.transpose(1, 2, 0)
    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr[..., 0]
    if arr.dtype != np.uint8:
        arr = arr.astype(np.uint8, copy=False)
    Image.fromarray(arr).save(path)


def _save_float_npy(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, arr.astype(np.float32, copy=False))


def build_512_sample(
    source_id: str,
    *,
    data_root: Union[str, Path],
    coco_dir: Union[str, Path],
    split: Split,
    direction: Direction,
    dmax_px: int,
    disparity_subdir: str = "disparity",
    case_id: Optional[str] = None,
    coco_subdir: Optional[str] = None,
    size: int = 512,
) -> SampleManifest:
    """Materialise a single 512 case for ``source_id``.

    The function does **not** verify the 320 mask is absent (we
    generate the mask from scratch); it merely returns a
    :class:`SampleManifest` describing what was written.

    Parameters
    ----------
    source_id : str
        The COCO image id (e.g. ``"000000033114"``).
    data_root : Path
        Where the case artefacts are written (e.g.
        ``benchmarks/coco32_grt_512``).
    coco_dir : Path
        Directory containing the source ``<source_id>.jpg`` files
        (Hami's junction: ``D:/project/Hami/data/images/coco_train2017``).
    split : Split
        Which split this case belongs to.
    direction : Direction
        GRT projection direction.
    dmax_px : int
        Maximum disparity in pixels (the multiplier applied to the
        p99-normalised disparity).
    disparity_subdir : str
        Sub-directory under ``data_root`` where the ZipDepth cache
        is read from.
    case_id : str, optional
        Override the default case id. When ``None`` the id is
        ``f"{source_id}__{dmax_px:02d}_{direction.value}"``.
    coco_subdir : str, optional
        Reserved for callers that need to override the source
        directory layout. Currently informational.
    size : int
        Output spatial size. Must be a multiple of 8. Default 512.

    Returns
    -------
    SampleManifest
        Manifest with paths **relative to ``data_root``**.
    """
    if size % 8 != 0:
        raise ConditionContractError(
            f"size must be a multiple of 8, got {size}"
        )
    if dmax_px <= 0:
        raise ConditionContractError(f"dmax_px must be positive, got {dmax_px}")

    data_root = Path(data_root)
    coco_dir = Path(coco_dir)
    case_id = case_id or f"{source_id}__{dmax_px:02d}_{direction.value}"

    # 1) Source RGB at 512²
    rgb_path = coco_dir / f"{source_id}.jpg"
    if not rgb_path.exists():
        raise FileNotFoundError(f"source image not found: {rgb_path}")
    rgb01 = _load_rgb01(rgb_path, size)  # (3, H, H) float32

    # 2) Disparity from the cache (already p99-normalised per the
    # Hami cache contract).
    adapter = ZipDepthAdapter(
        data_root=data_root, subdir=disparity_subdir, prefer_torch=True
    )
    cache_path = adapter.path_for(source_id)
    if not cache_path.exists():
        raise FileNotFoundError(
            f"disparity cache not found for source {source_id}: {cache_path}\n"
            "Run the ZipDepth cache step (Hami's "
            "scripts/cache_coco32_grt_disparity.py) first."
        )
    d_norm_cached = adapter.load(source_id).astype(np.float32, copy=False)
    if d_norm_cached.shape != (size, size):
        # Resize the cached depth to the target resolution using
        # BILINEAR (which preserves the p99 anchor in expectation).
        d_img = Image.fromarray(d_norm_cached).resize(
            (size, size), Image.Resampling.BILINEAR
        )
        d_norm_cached = np.asarray(d_img, dtype=np.float32)

    # 3) Re-derive the p99 *from the resized cache* so the
    # recorded statistics match the artefact exactly.
    applier = DepthNormalizationApplier.from_disparity(d_norm_cached)
    p99 = applier.p99
    depth_norm_record = applier.normalization

    # 4) GRT mask at 512 — DO NOT reuse the 320 mask.
    d_px = d_norm_cached * np.float32(dmax_px)
    mask_bool = compute_grt_mask(
        (size, size), d_px, direction, dmax_px=dmax_px
    )
    mask_u8 = mask_bool.astype(np.uint8) * 255

    # 5) Generate rgb_hole and depth_hole.
    mask_f = mask_bool.astype(np.float32)[None, :, :]  # (1, H, W)
    rgb_hole = (rgb01 * (1.0 - mask_f)).astype(np.float32, copy=False)
    sign = -1 if direction is Direction.L2R else 1
    depth_hole_signed = applier.apply(d_norm_cached, direction_sign=sign)
    depth_hole = (depth_hole_signed[None, :, :] * (1.0 - mask_f)).astype(
        np.float32, copy=False
    )

    # 6) Persist artefacts (paths RELATIVE to data_root).
    split_dir = data_root / split.value
    rgb_hole_rel = f"{split.value}/rgb_hole/{case_id}.png"
    depth_hole_rel = f"{split.value}/depth_hole/{case_id}.npy"
    mask_rel = f"{split.value}/masks/{case_id}.png"
    target_rel = f"{split.value}/targets/{case_id}.png"
    # The "target" for the DIBR task is the clean source RGB
    # (the ideal inpainting output is the source frame itself,
    # modulo the warp displacement which is already encoded in
    # the hole position). For training, the teacher fills the
    # hole from the target; we materialise the clean RGB so
    # downstream code has a stable on-disk target.
    _save_uint8(data_root / rgb_hole_rel, (rgb_hole * 255).astype(np.uint8))
    _save_float_npy(data_root / depth_hole_rel, depth_hole[0])
    _save_uint8(data_root / mask_rel, mask_u8)
    _save_uint8(data_root / target_rel, (rgb01 * 255).astype(np.uint8))

    # Per-case geometry JSON for downstream auditing.
    stats_rel = f"{split.value}/stats/{case_id}.json"
    stats_path = data_root / stats_rel
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.write_text(
        json.dumps(
            {
                "case_id": case_id,
                "source_id": source_id,
                "split": split.value,
                "direction": direction.value,
                "dmax_px": int(dmax_px),
                "size": int(size),
                "depth_normalization": {
                    "p99": float(p99),
                    "sign_convention": depth_norm_record.sign_convention,
                },
                "hole_ratio": float(mask_u8.astype(np.float32).mean() / 255.0),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    return SampleManifest(
        case_id=case_id,
        source_id=source_id,
        split=split,
        coord_frame=CoordFrame.SOURCE,
        rgb_path=rgb_hole_rel,
        depth_path=depth_hole_rel,
        mask_path=mask_rel,
        target_path=target_rel,
        direction=direction,
        dmax_px=int(dmax_px),
        depth_normalization=depth_norm_record,
    )


def prepare_512_sample(*args, **kwargs) -> SampleManifest:
    """Alias for :func:`build_512_sample` (matches the CLI naming)."""
    return build_512_sample(*args, **kwargs)
