"""Comparison figure renderer (TDD §9.3).

The fixed arrangement is::

    [input RGB, mask, depth, target, baseline, teacher, student, error]

The renderer is a thin wrapper over :mod:`matplotlib` with two
guard rails:

* ``matplotlib.use("Agg")`` is set on import so the function works
  in headless test runs without a display.
* Failures (e.g. matplotlib not installed, bad input) print to
  stderr and return ``None`` instead of raising.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np

# Always use the headless backend so import works in tests.
import matplotlib
matplotlib.use("Agg")  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402


__all__ = ["plot_comparison_grid"]


#: Canonical column order for the comparison grid.
COLUMN_ORDER = (
    "input",
    "mask",
    "depth",
    "target",
    "baseline",
    "teacher",
    "student",
    "error",
)


def _to_hwc_uint8(arr: np.ndarray) -> np.ndarray:
    """Convert a tensor in ``(C, H, W)`` or ``(H, W)`` to ``(H, W, C)`` uint8."""
    if arr.ndim == 3 and arr.shape[0] in (1, 3):
        arr = arr.transpose(1, 2, 0)
    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr[..., 0]
    if arr.ndim == 2:
        arr = arr[..., None].repeat(3, axis=-1)
    if arr.ndim != 3:
        raise ValueError(f"cannot convert shape {arr.shape} to HWC uint8")
    if arr.shape[-1] == 4:
        arr = arr[..., :3]
    if arr.shape[-1] != 3:
        raise ValueError(f"expected 1 or 3 channels, got {arr.shape[-1]}")
    if arr.dtype != np.uint8:
        mn = float(arr.min())
        mx = float(arr.max())
        if mx - mn < 1e-6:
            arr = np.zeros_like(arr, dtype=np.uint8)
        else:
            arr = ((arr - mn) / (mx - mn) * 255.0).clip(0, 255).astype(np.uint8)
    return arr


def _depth_to_rgb(arr: np.ndarray) -> np.ndarray:
    """Map a 2D depth map to a viridis-coloured RGB uint8 image."""
    if arr.ndim == 3:
        arr = arr[0]
    arr = arr.astype(np.float32, copy=False)
    mn = float(arr.min())
    mx = float(arr.max())
    if mx - mn < 1e-6:
        norm = np.zeros_like(arr)
    else:
        norm = (arr - mn) / (mx - mn)
    cmap = plt.get_cmap("viridis")
    rgba = cmap(norm)
    return (rgba[..., :3] * 255).astype(np.uint8)


def _error_to_rgb(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Pixel-wise L1 error heat-map (red = high, white = none)."""
    if pred.ndim == 3:
        pred = pred.transpose(1, 2, 0)
    if target.ndim == 3:
        target = target.transpose(1, 2, 0)
    err = np.abs(pred - target).mean(axis=-1)  # (H, W)
    mx = float(err.max())
    if mx < 1e-6:
        return np.full((err.shape[0], err.shape[1], 3), 255, dtype=np.uint8)
    norm = err / mx
    rgba = plt.get_cmap("Reds")(norm)
    return (rgba[..., :3] * 255).astype(np.uint8)


def _ensure_image(item: Any, kind: str) -> np.ndarray:
    """Return a ``(H, W, 3)`` uint8 image for the given item + kind."""
    if not isinstance(item, np.ndarray):
        raise ValueError(f"{kind} entry must be a numpy array, got {type(item).__name__}")
    if kind in ("input", "target", "baseline", "teacher", "student"):
        return _to_hwc_uint8(item)
    if kind == "mask":
        return _to_hwc_uint8(item)
    if kind == "depth":
        return _depth_to_rgb(item)
    if kind == "error":
        # error needs a target reference; expect {pred, target}
        if not isinstance(item, dict) or "pred" not in item or "target" not in item:
            raise ValueError(
                f"'error' must be a dict with 'pred' and 'target' keys, got {type(item).__name__}"
            )
        return _error_to_rgb(item["pred"], item["target"])
    raise ValueError(f"unknown column kind {kind!r}")


def plot_comparison_grid(
    items: Sequence[Dict[str, Any]],
    out_path: Union[str, Path],
    *,
    title: str = "DIBR comparison",
    show: bool = False,
) -> Optional[Path]:
    """Write a multi-row comparison figure.

    Parameters
    ----------
    items : sequence of dict
        Each dict must have the same keys as :data:`COLUMN_ORDER`.
        The "error" entry may be either an image or a dict with
        ``{"pred", "target"}`` keys.
    out_path : str or Path
        Where the PNG is written.
    title : str
        Figure super-title.
    show : bool
        If True, call ``plt.show()`` (no-op on Agg backend).

    Returns
    -------
    Path or None
        The path of the written PNG, or ``None`` if rendering
        failed (a warning is printed to stderr in that case).
    """
    out = Path(out_path)
    try:
        n_rows = len(items)
        n_cols = len(COLUMN_ORDER)
        # Choose a row count of at least 1; if 0 rows, write an
        # empty placeholder image.
        if n_rows == 0:
            fig, ax = plt.subplots(figsize=(n_cols * 2, 1))
            ax.text(0.5, 0.5, "(no cases)", ha="center", va="center", transform=ax.transAxes)
            ax.axis("off")
            fig.suptitle(title)
            fig.savefig(out, bbox_inches="tight")
            plt.close(fig)
            return out

        fig, axes = plt.subplots(
            n_rows, n_cols, figsize=(n_cols * 2, n_rows * 2 + 1)
        )
        if n_rows == 1:
            axes = np.array([axes])
        for r, item in enumerate(items):
            for c, kind in enumerate(COLUMN_ORDER):
                ax = axes[r, c]
                value = item.get(kind, None)
                if value is None:
                    ax.text(0.5, 0.5, "(missing)", ha="center", va="center",
                            transform=ax.transAxes, fontsize=6, color="grey")
                    ax.axis("off")
                    ax.set_title(kind if r == 0 else "", fontsize=8)
                    continue
                try:
                    img = _ensure_image(value, kind)
                except (ValueError, TypeError) as exc:
                    print(
                        f"[plot_comparison_grid] row {r} col {kind}: {exc}; "
                        f"writing blank tile.",
                        file=sys.stderr,
                    )
                    img = np.full((32, 32, 3), 128, dtype=np.uint8)
                ax.imshow(img)
                ax.axis("off")
                if r == 0:
                    ax.set_title(kind, fontsize=8)
        fig.suptitle(title)
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, bbox_inches="tight")
        plt.close(fig)
        if show:
            plt.show()
        return out
    except Exception as exc:  # pragma: no cover - defensive
        print(
            f"[plot_comparison_grid] failed to write {out}: {exc}",
            file=sys.stderr,
        )
        return None
