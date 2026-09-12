"""Pure-numpy GRT (Geometric Reciprocity Theorem) warp-hole mask functions.

Ported from Hami's ``hami.data.grt_mask`` (commit TBD-pin). The functions in
this module are deliberately torch-free so they can be unit-tested and used
in the data preparation pipeline without pulling torch into the
contracts-level import graph.

Conventions (matching Hami / TDD §4.2)
-------------------------------------

- ``disparity`` is non-negative; larger = closer.
- ``d_px = normalised_disparity * dmax_px`` is the per-pixel horizontal
  shift in pixels.
- The target column in the synthesised view is ``t = round(x + d_px)``
  (half-to-even rounding matches ``torch.round``).
- The mask marks the *source* pixels that are lost: out-of-bounds OR
  occluded by a strictly nearer pixel projecting to the same target
  column. Ties survive (no spurious holes on flat walls).
- Direction ``"L"`` (L2R) is implemented as
  ``hflip(mask_R(hflip(d)))`` to mirror Hami's behaviour.
"""

from __future__ import annotations

from typing import Literal, Tuple

import numpy as np

from ..contracts import Direction


__all__ = [
    "apply_p99_normalization",
    "compute_grt_mask",
    "disparity_to_signed_normalized",
]


# ---------------------------------------------------------------------------
# p99 normalisation
# ---------------------------------------------------------------------------

#: Threshold below which p99 (or the fallback ``max``) is considered
#: too small to be a useful anchor. Hami uses 1e-6.
_P99_EPS = 1e-6


def apply_p99_normalization(disparity: np.ndarray) -> np.ndarray:
    """Normalise a non-negative disparity to ``[0, 1]`` via per-image p99.

    Mirrors ``Hami.normalize_disparity``:

    .. code-block:: text

        p = quantile(d, 0.99)
        if p <= 1e-6: p = d.max()
        if p <= 1e-6: return zeros_like(d)
        else: return (d / p).clamp(0, 1)

    Parameters
    ----------
    disparity : np.ndarray
        Non-negative disparity map. Any shape is supported but the
        per-image p99 is taken over the flattened array, matching
        Hami's torch behaviour.

    Returns
    -------
    np.ndarray
        ``float32`` array with the same shape as ``disparity`` and
        values clipped to ``[0, 1]``.
    """
    d = np.asarray(disparity, dtype=np.float32)
    if d.ndim == 0:
        # Scalar inputs are not a real image but be lenient.
        d = d.reshape(1)
        p = float(np.quantile(d, 0.99)) if d.size else 0.0
    else:
        p = float(np.quantile(d, 0.99))
    if not np.isfinite(p) or p <= _P99_EPS:
        p = float(d.max()) if d.size else 0.0
    if not np.isfinite(p) or p <= _P99_EPS:
        return np.zeros_like(d, dtype=np.float32)
    out = d / np.float32(p)
    return np.clip(out, 0.0, 1.0).astype(np.float32, copy=False)


def disparity_to_signed_normalized(
    disparity: np.ndarray,
    p99: float,
    sign_convention: str = "L_neg_R_pos",
) -> np.ndarray:
    """Convert a per-image p99-normalised disparity to a signed feature.

    The pipeline stored in :class:`moebius_finetune.contracts.DepthNormalization`
    is:

    - ``d_norm = apply_p99_normalization(d)`` (range ``[0, 1]``).
    - For ``direction = L2R`` (a.k.a. ``"L"``): ``signed = -d_norm`` so
      the left half of the field (where the *target* is to the right) is
      negative.
    - For ``direction = R2L`` (a.k.a. ``"R"``): ``signed = +d_norm``.

    Parameters
    ----------
    disparity : np.ndarray
        Per-pixel disparity in pixels. Must be non-negative, finite.
    p99 : float
        Anchor to apply first (see :func:`apply_p99_normalization`).
    sign_convention : str
        Currently only ``"L_neg_R_pos"`` is supported.

    Returns
    -------
    np.ndarray
        ``float32`` array in ``[-1, 1]`` after the p99 clip and sign
        flip, ready to be passed to the model as a signed depth feature.
    """
    if sign_convention != "L_neg_R_pos":
        raise ValueError(
            f"unsupported sign_convention {sign_convention!r}; "
            "expected 'L_neg_R_pos'"
        )
    d = np.asarray(disparity, dtype=np.float32)
    if not np.all(np.isfinite(d)):
        raise ValueError("disparity contains non-finite values")
    if (d < 0).any():
        raise ValueError("disparity must be non-negative for GRT")
    if not np.isfinite(p99) or p99 <= _P99_EPS:
        return np.zeros_like(d, dtype=np.float32)
    clipped = np.clip(d / np.float32(p99), 0.0, 1.0).astype(np.float32, copy=False)
    return clipped  # caller chooses the sign via the L/R path


# ---------------------------------------------------------------------------
# GRT mask
# ---------------------------------------------------------------------------


def _grt_mask_core_r(d_px: np.ndarray) -> np.ndarray:
    """Pure-numpy R-direction core (mirror of Hami._grt_mask_core).

    Returns a boolean ``(H, W)`` array, ``True`` where the source pixel
    is lost (out-of-bounds OR strictly occluded).
    """
    H, W = d_px.shape
    xs = np.broadcast_to(np.arange(W, dtype=np.int64), (H, W))
    # Half-to-even rounding matches torch.round; numpy uses banker's
    # rounding via np.round which is equivalent for our use.
    t = np.round(xs.astype(np.float32) + d_px).astype(np.int64)
    valid = (t >= 0) & (t < W)
    t_safe = np.clip(t, 0, W - 1)
    # Buffer initialised to -inf so out-of-bounds entries never win.
    buf = np.full((H, W), -np.inf, dtype=np.float32)
    # Scatter the source values into the buffer at row*W + t_safe.
    # We only scatter valid (in-bounds) sources so that -inf wins for
    # out-of-bounds entries, matching Hami's `torch.where(valid, d_px, -inf)`.
    flat_buf = buf.reshape(-1)
    row_offsets = (np.arange(H, dtype=np.int64) * W)[:, None]
    flat_target = (row_offsets + t_safe).reshape(-1)
    flat_src = d_px.reshape(-1)
    flat_valid = valid.reshape(-1)
    if flat_valid.any():
        np.maximum.at(flat_buf, flat_target[flat_valid], flat_src[flat_valid])
    buf = flat_buf.reshape(H, W)
    # Gather the buffer at t_safe for every source pixel.
    B_at_src = np.take_along_axis(buf, t_safe, axis=1)
    # Strict occlusion: a source pixel is occluded if a strictly nearer
    # pixel projects to the same target column. Ties survive.
    occl = valid & (d_px < B_at_src)
    oob = ~valid
    return oob | occl


def compute_grt_mask(
    rgb_shape: Tuple[int, int],
    disparity: np.ndarray,
    direction: Direction,
    dmax_px: int,
    occlusion_test: Literal["strict"] = "strict",
) -> np.ndarray:
    """Compute the GRT warp-hole mask for a 2-D disparity field.

    Parameters
    ----------
    rgb_shape : (H, W)
        Spatial shape of the source image. Used as a sanity check.
    disparity : np.ndarray
        ``(H, W)`` per-pixel disparity in **pixels** (already
        multiplied by ``dmax_px``). Must be non-negative and finite.
    direction : Direction
        ``L2R`` or ``R2L`` from the contracts module.
    dmax_px : int
        Maximum disparity scale used to convert the per-image
        normalised disparity into pixels. When ``disparity`` is already
        in pixels, ``dmax_px`` is only used for the dmax bookkeeping
        recorded in the manifest. The function itself does not apply
        any extra scaling; the caller is expected to feed the scaled
        field.
    occlusion_test : {"strict"}
        Hami's "strict" comparison (ties survive). Reserved for future
        use; only the strict variant is implemented.

    Returns
    -------
    np.ndarray
        ``bool`` array of shape ``(H, W)``; ``True`` marks a hole pixel.
    """
    if occlusion_test != "strict":
        raise ValueError(
            f"unsupported occlusion_test {occlusion_test!r}; only 'strict' is implemented"
        )
    if not isinstance(direction, Direction):
        # Accept plain strings for convenience at test boundaries.
        try:
            direction = Direction(direction)
        except ValueError as exc:
            raise ValueError(
                f"direction must be a Direction enum or one of its values, got {direction!r}"
            ) from exc
    H, W = rgb_shape
    d = np.asarray(disparity, dtype=np.float32)
    if d.shape != (H, W):
        raise ValueError(
            f"disparity shape {d.shape} does not match rgb_shape {(H, W)}"
        )
    if not np.all(np.isfinite(d)):
        raise ValueError("disparity contains non-finite values")
    if (d < 0).any():
        raise ValueError("disparity must be non-negative for GRT")
    if int(dmax_px) <= 0:
        raise ValueError(f"dmax_px must be positive, got {dmax_px}")

    if direction is Direction.R2L:
        return _grt_mask_core_r(d)
    if direction is Direction.L2R:
        return np.flip(_grt_mask_core_r(np.flip(d, axis=-1)), axis=-1)
    raise ValueError(f"unsupported direction: {direction!r}")
