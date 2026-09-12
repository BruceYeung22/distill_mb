"""Per-image p99 depth normalisation helpers.

The on-disk contract stores :class:`DepthNormalization` alongside each
sample so loaders never re-derive the statistics. This module
implements:

* :func:`DepthNormalizationApplier.from_disparity` — compute the
  p99 from a per-image disparity map and freeze the parameters.
* :meth:`DepthNormalizationApplier.apply` — apply the stored
  statistics to another disparity field (e.g. a `depth_hole` batch
  tensor after the mask is applied).

Key rule (TDD §4.2): the p99 is computed **before** the mask is
applied, on the source-frame disparity. The mask is then applied
on top of the already-normalised map and zeros the hole region.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from ..contracts import DepthNormalization


__all__ = [
    "DepthNormalizationApplier",
    "apply_depth_normalization",
]


_EPS = 1e-6


@dataclass(frozen=True)
class DepthNormalizationApplier:
    """A pair of (raw disparity, normalisation) bound together.

    The p99 is computed from ``raw_disparity`` via the same recipe as
    Hami's :func:`apply_p99_normalization`, so the stored
    :class:`DepthNormalization` is always consistent with the
    pre-normalisation field the mask was built from.

    The ``apply`` method is a pure function of the inputs — no
    implicit state is kept between calls. The instance is therefore
    safe to share across threads.
    """

    raw_disparity: np.ndarray  # (H, W) float32, the *un-masked* source disparity
    normalization: DepthNormalization

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------
    @classmethod
    def from_disparity(
        cls,
        disparity: np.ndarray,
        sign_convention: str = "L_neg_R_pos",
    ) -> "DepthNormalizationApplier":
        """Compute the p99 anchor and freeze the pair.

        ``disparity`` must be the un-masked, non-negative, finite
        source-frame disparity (typically a ZipDepth inverse-depth
        output). The returned :class:`DepthNormalization` records
        exactly the same p99 the GRT mask pipeline used.
        """
        d = np.asarray(disparity, dtype=np.float32)
        if not np.all(np.isfinite(d)):
            raise ValueError("disparity contains non-finite values")
        if (d < 0).any():
            raise ValueError("disparity must be non-negative for p99 normalisation")
        p99 = float(np.quantile(d, 0.99)) if d.size else 0.0
        if not np.isfinite(p99) or p99 <= _EPS:
            p99 = float(d.max()) if d.size else 0.0
        if not np.isfinite(p99) or p99 <= _EPS:
            p99 = 0.0  # stored as 0; the applier will return zeros
        if sign_convention != "L_neg_R_pos":
            raise ValueError(
                f"unsupported sign_convention {sign_convention!r}; "
                "expected 'L_neg_R_pos'"
            )
        return cls(
            raw_disparity=d.astype(np.float32, copy=True),
            normalization=DepthNormalization(
                p99=p99, sign_convention=sign_convention
            ),
        )

    @classmethod
    def from_normalization(
        cls,
        normalization: DepthNormalization,
        raw_disparity: np.ndarray | None = None,
    ) -> "DepthNormalizationApplier":
        """Build an applier from a stored :class:`DepthNormalization`.

        ``raw_disparity`` is optional: when provided, the instance can
        be used to (re-)normalise other disparity fields without
        re-deriving the anchor. When it is ``None`` the applier can
        only :func:`apply` a stored :class:`DepthNormalization`
        directly via :func:`apply_depth_normalization`.
        """
        if raw_disparity is None:
            raw = np.zeros((0, 0), dtype=np.float32)
        else:
            raw = np.asarray(raw_disparity, dtype=np.float32)
        return cls(raw_disparity=raw, normalization=normalization)

    # ------------------------------------------------------------------
    # Application
    # ------------------------------------------------------------------
    def apply(
        self,
        depth_hole: np.ndarray,
        direction_sign: int = 1,
    ) -> np.ndarray:
        """Apply the frozen normalisation to a ``depth_hole`` tensor.

        Parameters
        ----------
        depth_hole : np.ndarray
            Per-pixel disparity with the same shape as
            ``raw_disparity`` (H, W), or batched (B, 1, H, W). The
            field may be the source frame's disparity; the function
            only uses the p99 anchor from ``normalization``. The
            caller is responsible for zeroing the hole region in
            advance (the contract requires ``depth_hole[mask==1]==0``).
        direction_sign : {-1, +1}
            ``+1`` for R2L (R-direction) → signed value is positive.
            ``-1`` for L2R (L-direction) → signed value is negative.

        Returns
        -------
        np.ndarray
            ``float32`` array with the same shape as ``depth_hole``,
            clipped to ``[-1, 1]``.
        """
        if direction_sign not in (-1, 1):
            raise ValueError(
                f"direction_sign must be -1 (L2R) or +1 (R2L), got {direction_sign}"
            )
        if self.normalization.p99 <= _EPS:
            return np.zeros_like(depth_hole, dtype=np.float32)
        d = np.asarray(depth_hole, dtype=np.float32)
        # If the input has 3 or 4 dims, normalise per-image by broadcasting
        # the (H, W) anchor: this is the batched case.
        if d.ndim in (3, 4):
            # Expect (B, 1, H, W) for 4D or (1, H, W) for 3D
            d_norm = d / np.float32(self.normalization.p99)
        elif d.ndim == 2:
            d_norm = d / np.float32(self.normalization.p99)
        else:
            raise ValueError(
                f"depth_hole must be 2D, 3D, or 4D, got ndim={d.ndim}"
            )
        d_norm = np.clip(d_norm, 0.0, 1.0)
        out = (direction_sign * d_norm).astype(np.float32, copy=False)
        return out

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------
    @property
    def p99(self) -> float:
        return self.normalization.p99

    def __repr__(self) -> str:
        return (
            f"DepthNormalizationApplier(p99={self.normalization.p99}, "
            f"sign={self.normalization.sign_convention}, "
            f"shape={self.raw_disparity.shape})"
        )


# ---------------------------------------------------------------------------
# Stateless functional API (for callers that already have a normalisation)
# ---------------------------------------------------------------------------


def apply_depth_normalization(
    depth_hole: np.ndarray,
    depth_normalization: DepthNormalization,
    direction_sign: int = 1,
) -> np.ndarray:
    """Stateless form of :meth:`DepthNormalizationApplier.apply`.

    Useful when only the :class:`DepthNormalization` summary is
    available (e.g. when re-applying the anchor at load time).
    """
    if depth_normalization.p99 <= _EPS:
        return np.zeros_like(depth_hole, dtype=np.float32)
    if direction_sign not in (-1, 1):
        raise ValueError(
            f"direction_sign must be -1 (L2R) or +1 (R2L), got {direction_sign}"
        )
    d = np.asarray(depth_hole, dtype=np.float32)
    out = d / np.float32(depth_normalization.p99)
    out = np.clip(out, 0.0, 1.0)
    out = (direction_sign * out).astype(np.float32, copy=False)
    return out
