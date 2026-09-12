"""Evaluation metrics for the DIBR hole-filling task (TDD §9.3).

All metrics operate on float32 numpy arrays and accept either a
single image ``(3, H, W)`` or a batched tensor ``(B, 3, H, W)``;
batch-level metrics are averaged over the valid cases (empty
masks are counted separately and excluded from the mean to avoid
inflating PSNR toward infinity).

Numerical-stability rules:

* :func:`hole_psnr` uses ``max(MSE, eps)`` to avoid log(0) and
  returns ``0.0`` for empty masks.
* :func:`boundary_l1` uses ``np.where(count > 0, ..., 0.0)`` for
  the band-mean to avoid 0/0.
* :func:`known_max_error` returns ``0.0`` for empty masks.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np


__all__ = [
    "aggregate_per_case",
    "boundary_l1",
    "hole_psnr",
    "known_max_error",
]


#: A small epsilon guarding against log(0) and 0/0 in metrics.
_EPS = 1e-12


def _ensure_4d(arr: np.ndarray, name: str) -> np.ndarray:
    if arr.ndim == 3:
        arr = arr[None, ...]
    if arr.ndim != 4:
        raise ValueError(
            f"{name} must be 3D [C, H, W] or 4D [B, C, H, W], got {arr.shape}"
        )
    return arr.astype(np.float32, copy=False)


def _ensure_mask_4d(arr: np.ndarray, name: str) -> np.ndarray:
    """Mask can be 2D (H, W), 3D (1, H, W), or 4D (B, 1, H, W)."""
    if arr.ndim == 2:
        arr = arr[None, None, ...]
    elif arr.ndim == 3:
        # Could be (1, H, W) or (B, H, W) — only (1, H, W) is valid
        # as a mask. Force 4D by adding the channel dim.
        if arr.shape[0] != 1 and arr.shape[0] > 1:
            # Treat as (B, H, W) by adding channel
            arr = arr[:, None, ...]
        else:
            arr = arr[None, ...] if arr.shape[0] != 1 else arr[None, ...]
    elif arr.ndim == 4:
        # Already 4D; pass through.
        pass
    else:
        raise ValueError(
            f"{name} must be 2D, 3D, or 4D, got ndim={arr.ndim}"
        )
    return arr.astype(np.float32, copy=False)


def hole_psnr(
    pred: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
    data_range: float = 1.0,
    per_case: bool = False,
) -> Union[float, np.ndarray]:
    """Hole-only PSNR (dB) for ``pred`` vs ``target`` inside ``mask``.

    Parameters
    ----------
    pred, target : np.ndarray
        ``(B, 3, H, W)`` (or ``(3, H, W)``) float32 arrays.
    mask : np.ndarray
        ``(B, 1, H, W)`` (or ``(1, H, W)`` or ``(H, W)``) 0/1 mask. 1 = hole.
    data_range : float
        Peak-to-peak range of the data; default ``1.0`` for [0, 1]
        RGB.
    per_case : bool
        If ``True``, return a ``(B,)`` array of per-case PSNR; empty
        cases are reported as 0.0. If ``False`` (default), return the
        scalar mean of the valid per-case PSNRs (or 0.0 if all are
        empty).

    Returns
    -------
    float or np.ndarray
    """
    p = _ensure_4d(pred, "pred")
    t = _ensure_4d(target, "target")
    m = _ensure_mask_4d(mask, "mask")
    if p.shape != t.shape:
        raise ValueError(f"pred/target shape mismatch: {p.shape} vs {t.shape}")
    # Mask must broadcast to the same B, H, W as p
    if m.shape[0] not in (1, p.shape[0]):
        raise ValueError(
            f"mask batch {m.shape[0]} incompatible with pred batch {p.shape[0]}"
        )
    if m.shape[2:] != p.shape[2:]:
        raise ValueError(
            f"mask spatial {m.shape[2:]} incompatible with pred spatial {p.shape[2:]}"
        )
    if m.shape[1] != 1:
        raise ValueError(f"mask channel dim must be 1, got {m.shape}")
    if data_range <= 0:
        raise ValueError(f"data_range must be positive, got {data_range}")

    B = p.shape[0]
    out = np.zeros(B, dtype=np.float32)
    for i in range(B):
        mi = m[0] if m.shape[0] == 1 else m[i]
        n_hole = float(mi.sum())
        if n_hole <= 0:
            out[i] = 0.0
            continue
        diff = (p[i] - t[i]) * mi
        # Per-element squared error, averaged over masked region
        mse = float(np.mean(diff * diff))
        if mse <= _EPS:
            out[i] = float("inf")
        else:
            out[i] = 10.0 * float(np.log10((data_range * data_range) / mse))
    if per_case:
        return out
    valid = out > 0
    if not valid.any():
        return 0.0
    return float(out[valid].mean())


def boundary_l1(
    pred: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
    band_px: int = 3,
    per_case: bool = False,
) -> Union[float, np.ndarray]:
    """Mean L1 error inside the mask boundary band.

    The band is the union of:
    * a ``band_px``-wide dilation of the hole;
    * a ``band_px``-wide erosion of the hole;

    so it captures the immediate inside and outside of the boundary.
    Pixels inside the hole but outside the band are excluded.

    Empty masks return 0.0.
    """
    p = _ensure_4d(pred, "pred")
    t = _ensure_4d(target, "target")
    m = _ensure_mask_4d(mask, "mask")
    if p.shape != t.shape:
        raise ValueError(f"pred/target shape mismatch: {p.shape} vs {t.shape}")
    if band_px <= 0:
        raise ValueError(f"band_px must be positive, got {band_px}")

    B = p.shape[0]
    out = np.zeros(B, dtype=np.float32)
    for i in range(B):
        mi = m[0] if m.shape[0] == 1 else m[i]
        m2d = mi[0]  # (H, W) 0/1
        if m2d.sum() <= 0:
            out[i] = 0.0
            continue
        band = _make_band(m2d, band_px)
        n = int(band.sum())
        if n <= 0:
            out[i] = 0.0
            continue
        # Compute per-pixel L1 across channels, then average over band.
        diff = np.abs(p[i] - t[i]).sum(axis=0)  # (H, W) per-pixel L1
        # Per-pixel average across the 3 channels: divide by 3
        diff = diff / np.float32(p.shape[1])
        out[i] = float(diff[band].mean())
    if per_case:
        return out
    valid = out > 0
    if not valid.any():
        return 0.0
    return float(out[valid].mean())


def known_max_error(
    pred: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
    per_case: bool = False,
) -> Union[float, np.ndarray]:
    """Maximum per-pixel L1 error in the known (non-hole) region.

    Empty masks return 0.0. If there is no known region (full
    hole), also returns 0.0.
    """
    p = _ensure_4d(pred, "pred")
    t = _ensure_4d(target, "target")
    m = _ensure_mask_4d(mask, "mask")
    if p.shape != t.shape:
        raise ValueError(f"pred/target shape mismatch: {p.shape} vs {t.shape}")
    B = p.shape[0]
    out = np.zeros(B, dtype=np.float32)
    for i in range(B):
        mi = m[0] if m.shape[0] == 1 else m[i]
        known = (1.0 - mi[0])  # (H, W)
        if known.sum() <= 0:
            out[i] = 0.0
            continue
        diff = np.abs(p[i] - t[i]).max(axis=0)  # (H, W) per-pixel max over C
        out[i] = float(diff[known > 0].max())
    if per_case:
        return out
    valid = out > 0
    if not valid.any():
        return 0.0
    return float(out[valid].max())


def _make_band(mask2d: np.ndarray, band_px: int) -> np.ndarray:
    """Union of erosion and dilation of a 0/1 mask.

    Uses a fast separable max/min (no scipy dependency) and
    supports ``band_px >= 1``; for ``band_px == 1`` this reduces to
    a 3×3 cross neighbourhood union.
    """
    mask = mask2d.astype(np.float32)
    eroded = _erode(mask, band_px)
    dilated = _dilate(mask, band_px)
    band = ((dilated > 0) & (eroded == 0)) | (mask > 0)
    return band.astype(bool)


def _erode(mask: np.ndarray, k: int) -> np.ndarray:
    """k-iteration min-filter (separable, no scipy)."""
    out = mask.copy()
    for _ in range(k):
        out = _separable_min(out)
    return out


def _dilate(mask: np.ndarray, k: int) -> np.ndarray:
    out = mask.copy()
    for _ in range(k):
        out = _separable_max(out)
    return out


def _separable_min(x: np.ndarray) -> np.ndarray:
    pad = 1
    xp = np.pad(x, pad, mode="edge")
    # min over [-1, 0, 1] in both axes
    a = np.stack([xp[:-2, :-2], xp[:-2, 1:-1], xp[:-2, 2:],
                  xp[1:-1, :-2], xp[1:-1, 1:-1], xp[1:-1, 2:],
                  xp[2:, :-2], xp[2:, 1:-1], xp[2:, 2:]], axis=0)
    return a.min(axis=0)


def _separable_max(x: np.ndarray) -> np.ndarray:
    pad = 1
    xp = np.pad(x, pad, mode="edge")
    a = np.stack([xp[:-2, :-2], xp[:-2, 1:-1], xp[:-2, 2:],
                  xp[1:-1, :-2], xp[1:-1, 1:-1], xp[1:-1, 2:],
                  xp[2:, :-2], xp[2:, 1:-1], xp[2:, 2:]], axis=0)
    return a.max(axis=0)


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def aggregate_per_case(
    metrics_list: Sequence[Dict[str, Any]],
    *,
    group_keys: Optional[Sequence[str]] = None,
    low_quantile: float = 0.1,
) -> Dict[str, Any]:
    """Aggregate a list of per-case metric dicts into a summary.

    Each element of ``metrics_list`` must be a dict with at least
    the following keys:

    * ``case_id`` (str)
    * ``split`` (str, optional but recommended for grouping)
    * ``direction`` ({"L2R", "R2L"}, optional but recommended)
    * ``dmax_px`` (int, optional but recommended)
    * ``hole_ratio`` (float, optional, used for "by hole width")
    * ``hole_psnr`` (float, or NaN/inf to indicate empty/skip)
    * ``boundary_l1`` (float)
    * ``known_max_error`` (float)
    * ``empty`` (bool, True when the case had no hole)

    Parameters
    ----------
    metrics_list : sequence of dict
        Per-case records.
    group_keys : sequence of str, optional
        Additional grouping keys to include in the report.
    low_quantile : float
        Quantile (0 < low < 1) used to compute the low-percentile
        PSNR. Default 0.1 (10th percentile).

    Returns
    -------
    dict
        Top-level keys:

        * ``"overall"``: ``{"n": int, "empty": int, "mean": float,
          "median": float, "low": float, "min": float, "max":
          float}`` for ``hole_psnr`` over the non-empty cases.
        * ``"empty_count"``: int.
        * ``"by_split"``: ``{split: summary}``
        * ``"by_direction"``: ``{direction: summary}``
        * ``"by_dmax"``: ``{dmax_px: summary}``
        * ``"by_hole_width"``: ``{"thin": summary, "medium":
          summary, "wide": summary}`` based on ``hole_ratio``.
        * ``"by_group"``: ``{key: {value: summary}}`` for each
          extra group key.
    """
    if not metrics_list:
        return {
            "overall": {
                "n": 0,
                "empty": 0,
                "mean": 0.0,
                "median": 0.0,
                "low": 0.0,
                "min": 0.0,
                "max": 0.0,
            },
            "empty_count": 0,
            "by_split": {},
            "by_direction": {},
            "by_dmax": {},
            "by_hole_width": {},
            "by_group": {},
        }

    # Index psnr by case + collect empties.
    valid_psnr: List[float] = []
    empty_count = 0
    by_split: Dict[str, List[float]] = {}
    by_dir: Dict[str, List[float]] = {}
    by_dmax: Dict[int, List[float]] = {}
    by_hole_width: Dict[str, List[float]] = {"thin": [], "medium": [], "wide": []}
    by_group: Dict[str, Dict[Any, List[float]]] = {}

    for m in metrics_list:
        if m.get("empty", False) or m.get("hole_psnr", 0.0) in (None, 0.0):
            # Treat 0.0 PSNR as the empty case marker (we use 0.0
            # for empty by convention in the per-case records).
            empty = bool(m.get("empty", False)) or float(m.get("hole_psnr", 0.0)) <= 0.0
        else:
            empty = False
        if empty:
            empty_count += 1
            continue
        psnr = float(m["hole_psnr"])
        valid_psnr.append(psnr)
        # Split
        s = m.get("split", "unspecified")
        by_split.setdefault(str(s), []).append(psnr)
        # Direction
        d = str(m.get("direction", "unspecified"))
        by_dir.setdefault(d, []).append(psnr)
        # Dmax
        dmax_key = m.get("dmax_px")
        if dmax_key is not None:
            try:
                by_dmax.setdefault(int(dmax_key), []).append(psnr)
            except (TypeError, ValueError):
                pass
        # Hole width bucket
        hr = float(m.get("hole_ratio", 0.0))
        if hr <= 0.05:
            bucket = "thin"
        elif hr <= 0.2:
            bucket = "medium"
        else:
            bucket = "wide"
        by_hole_width[bucket].append(psnr)
        # Extra groups
        if group_keys:
            for k in group_keys:
                if k in m:
                    by_group.setdefault(k, {}).setdefault(m[k], []).append(psnr)

    def _summary(values: Sequence[float]) -> Dict[str, Any]:
        if not values:
            return {"n": 0, "mean": 0.0, "median": 0.0, "low": 0.0, "min": 0.0, "max": 0.0}
        arr = np.asarray(values, dtype=np.float64)
        n = int(arr.size)
        return {
            "n": n,
            "mean": float(arr.mean()),
            "median": float(np.median(arr)),
            "low": float(np.quantile(arr, low_quantile)) if n >= 2 else float(arr.min()),
            "min": float(arr.min()),
            "max": float(arr.max()),
        }

    return {
        "overall": _summary(valid_psnr),
        "empty_count": int(empty_count),
        "by_split": {k: _summary(v) for k, v in by_split.items()},
        "by_direction": {k: _summary(v) for k, v in by_dir.items()},
        "by_dmax": {k: _summary(v) for k, v in sorted(by_dmax.items())},
        "by_hole_width": {k: _summary(v) for k, v in by_hole_width.items()},
        "by_group": {k: {vv: _summary(vv_vals) for vv, vv_vals in v.items()} for k, v in by_group.items()},
    }
