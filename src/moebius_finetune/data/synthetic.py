"""Synthetic :class:`ConditionBatch` fixtures for B/C smoke runs.

The fixtures are built with **pure numpy** (no torch, no diffusers)
and strictly satisfy the contracts:

* ``rgb_hole`` and ``depth_hole`` are exactly 0 inside the hole.
* ``hole_mask`` only takes values in ``{0, 1}``.
* All four tensors are float32 and ``H, W`` are multiples of 8.
* Noise is the standard normal (no hidden state).

``hole_spec`` choices
---------------------

* ``"thin"`` — a single thin vertical strip (8 pixels wide).
* ``"wide"`` — a single wide rectangle covering ~30% of the area.
* ``"full"`` — every pixel is in the hole; the known region is
  empty so the model has to predict the whole image.
* ``"empty"`` — no hole at all; the inpaint helper should return
  the input unchanged.
"""

from __future__ import annotations

from typing import Literal, Optional

import numpy as np

from ..contracts import ConditionBatch, validate_condition


__all__ = ["make_synthetic_batch"]


HoleSpec = Literal["thin", "wide", "full", "empty"]


def _build_mask(H: int, W: int, hole_spec: HoleSpec, seed: int) -> np.ndarray:
    """Return a ``(1, H, W)`` 0/1 mask."""
    if hole_spec == "empty":
        return np.zeros((H, W), dtype=np.float32)
    if hole_spec == "full":
        return np.ones((H, W), dtype=np.float32)
    if hole_spec == "thin":
        # 8-px wide vertical strip in the middle.
        strip_w = max(8, min(W // 8, 16))
        x0 = (W - strip_w) // 2
        m = np.zeros((H, W), dtype=np.float32)
        m[:, x0 : x0 + strip_w] = 1.0
        return m
    if hole_spec == "wide":
        # Rectangular hole ~ 60% of width and 50% of height.
        hh, hw = H // 2, int(W * 0.6)
        y0 = (H - hh) // 2
        x0 = (W - hw) // 2
        m = np.zeros((H, W), dtype=np.float32)
        m[y0 : y0 + hh, x0 : x0 + hw] = 1.0
        return m
    raise ValueError(f"unknown hole_spec {hole_spec!r}")


def make_synthetic_batch(
    H: int = 512,
    B: int = 1,
    *,
    hole_spec: HoleSpec = "thin",
    seed: int = 0,
    signed_depth: bool = True,
    validate: bool = True,
) -> ConditionBatch:
    """Build a :class:`ConditionBatch` honouring the contracts.

    Parameters
    ----------
    H : int
        Spatial height. The width is also ``H`` (square images).
    B : int
        Batch size. All B samples share the same hole layout for
        determinism.
    hole_spec : {"thin", "wide", "full", "empty"}
        See module docstring.
    seed : int
        RNG seed.
    signed_depth : bool
        If ``True``, ``depth_hole`` is signed (negative left,
        positive right). The default matches the model's input
        convention. Set to ``False`` for unsigned debugging.
    validate : bool
        Run :func:`validate_condition` before returning.

    Returns
    -------
    ConditionBatch
    """
    if H % 8 != 0:
        raise ValueError(f"H must be a multiple of 8, got {H}")
    W = H
    rng = np.random.default_rng(seed)

    # Build a single 2-D mask then broadcast to (B, 1, H, W).
    mask2d = _build_mask(H, W, hole_spec, seed)
    mask = np.broadcast_to(mask2d[None, None, :, :], (B, 1, H, W)).astype(
        np.float32, copy=True
    )

    # RGB: random in [0, 1]; mask region must be exactly 0.
    rgb = rng.random((B, 3, H, W), dtype=np.float32)
    rgb = rgb * (1.0 - mask)
    # Sanity: hole pixels in rgb are 0.
    assert float((rgb * mask).max()) == 0.0

    # Depth: a smooth signed inverse depth.
    ys = np.linspace(-1.0, 1.0, H, dtype=np.float32)
    xs = np.linspace(-1.0, 1.0, W, dtype=np.float32)
    depth_field = (ys[:, None] * 0.4 + xs[None, :] * 0.6).astype(np.float32)
    if signed_depth:
        # Rescale to roughly [-0.5, 0.5] so the magnitude is well
        # below the saturating regime.
        depth_field = depth_field - depth_field.mean()
    depth = np.broadcast_to(depth_field[None, None, :, :], (B, 1, H, W)).astype(
        np.float32, copy=True
    )
    # Add a small per-batch jitter so each sample is distinct.
    jitter = rng.uniform(-0.05, 0.05, size=(B, 1, H, W)).astype(np.float32)
    depth = depth + jitter
    depth = np.clip(depth, -1.0, 1.0)
    # Mask region: 0.
    depth = depth * (1.0 - mask)
    assert float((depth * mask).max()) == 0.0

    # Noise: explicit, deterministic, stdnormal.
    noise = rng.standard_normal((B, 4, H // 8, W // 8)).astype(np.float32)

    batch = ConditionBatch(
        rgb_hole=rgb.astype(np.float32, copy=False),
        hole_mask=mask.astype(np.float32, copy=False),
        depth_hole=depth.astype(np.float32, copy=False),
        noise=noise.astype(np.float32, copy=False),
    )
    if validate:
        validate_condition(batch)
    return batch
