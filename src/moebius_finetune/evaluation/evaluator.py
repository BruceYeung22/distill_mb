"""Fixed-seed evaluator for the DIBR inpainting task.

The evaluator takes a :class:`moebius_finetune.contracts.ConditionBatch`
and a ``predictor`` callable, and produces the per-case metrics
defined in :mod:`moebius_finetune.evaluation.metrics` for a fixed
seed (default 0). Optionally it also reports the mean and std of
``hole_psnr`` over ``seeds 0..3`` as a stability proxy (TDD §7.1,
§9.3).

The predictor interface is intentionally minimal so a dummy
implementation (e.g. ``run_dummy_predictor``) can satisfy it
without any real network code:

.. code-block:: python

    def predictor(condition: ConditionBatch, *, seed: int) -> np.ndarray:
        # returns (B, 3, H, W) float32 in [0, 1]
        ...

The evaluator validates the predictor's output (shape, dtype,
finite) and raises :class:`ConditionContractError` on contract
violations.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from ..contracts import (
    ConditionBatch,
    ConditionContractError,
    SampleManifest,
    inpaint,
    validate_condition,
)
from ..data.synthetic import make_synthetic_batch
from .metrics import (
    boundary_l1,
    hole_psnr,
    known_max_error,
)


__all__ = [
    "FixedEvaluator",
    "evaluate_manifest",
    "run_dummy_predictor",
]


Predictor = Callable[..., np.ndarray]
CaseLoader = Callable[[SampleManifest], Tuple[ConditionBatch, np.ndarray]]


@dataclass
class FixedEvaluator:
    """Run a fixed-seed evaluation pass over a (small) manifest.

    Parameters
    ----------
    predictor : callable
        ``predictor(condition, *, seed) -> np.ndarray (B, 3, H, W) [0, 1]``.
    manifest : sequence of SampleManifest or None
        Manifest to evaluate. When ``None``, a synthetic batch is
        used instead (so the evaluator is testable without real
        data).
    seed : int
        Primary seed for the evaluation.
    n_seeds : int
        Number of seeds (0..n_seeds-1) used for the stability
        report. Default 4.
    H : int
        Spatial size for the synthetic fallback.
    B : int
        Batch size for the synthetic fallback.
    hole_spec : str
        Hole layout for the synthetic fallback.
    """

    predictor: Predictor
    manifest: Optional[Sequence[SampleManifest]] = None
    seed: int = 0
    n_seeds: int = 4
    H: int = 64
    B: int = 1
    hole_spec: str = "thin"
    case_loader: Optional[CaseLoader] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def run(self) -> Dict[str, Any]:
        """Run the evaluator and return ``{per_case, aggregate}``.

        The synthetic fallback produces a single virtual case with
        ``case_id == "__synthetic__"`` and the configured hole
        layout. For the stability report, only the synthetic case is
        re-run with each of the requested seeds; the result is
        expressed as the per-seed hole PSNR.
        """
        if self.manifest is not None and len(self.manifest) > 0:
            if self.case_loader is None:
                raise ConditionContractError(
                    "manifest evaluation requires case_loader; refusing to "
                    "silently substitute synthetic data"
                )
            return self._run_manifest()
        return self._run_synthetic()

    # ------------------------------------------------------------------
    # Synthetic fallback
    # ------------------------------------------------------------------
    def _run_synthetic(self) -> Dict[str, Any]:
        batch = make_synthetic_batch(
            H=self.H, B=self.B, hole_spec=self.hole_spec, seed=self.seed
        )
        target = self._build_synthetic_target(batch)
        per_case = self._eval_batch(batch, target, seed=self.seed, case_id="__synthetic__")
        stability = self._stability_for_synthetic(target)
        return {
            "per_case": [per_case],
            "aggregate": _aggregate([per_case]),
            "stability": stability,
        }

    def _stability_for_synthetic(self, target: np.ndarray) -> Dict[str, Any]:
        psnrs: List[float] = []
        for s in range(self.n_seeds):
            batch = make_synthetic_batch(
                H=self.H, B=self.B, hole_spec=self.hole_spec, seed=s
            )
            case = self._eval_batch(batch, target, seed=s, case_id=f"__synthetic_s{s}__")
            psnrs.append(float(case["hole_psnr"]))
        arr = np.asarray(psnrs, dtype=np.float64)
        return {
            "n_seeds": self.n_seeds,
            "psnr_mean": float(arr.mean()),
            "psnr_std": float(arr.std()),
            "psnr_per_seed": psnrs,
        }

    def _build_synthetic_target(self, batch: ConditionBatch) -> np.ndarray:
        """Build a synthetic target: rgb + a small fill inside the hole.

        The synthetic target is ``rgb_hole + mask * 0.5`` (clipped),
        which means the dummy predictor can hit a PSNR of roughly
        ``10 * log10(1 / 0.25^2) ≈ 24 dB`` by predicting the
        constant 0.5 inside the hole.
        """
        rgb = batch.rgb_hole
        mask = batch.hole_mask
        target = rgb + mask * np.float32(0.5)
        return np.clip(target, 0.0, 1.0).astype(np.float32, copy=False)

    # ------------------------------------------------------------------
    # Manifest-based run
    # ------------------------------------------------------------------
    def _run_manifest(self) -> Dict[str, Any]:
        per_case: List[Dict[str, Any]] = []
        stability: List[Dict[str, Any]] = []
        for case in self.manifest:  # type: ignore[union-attr]
            batch, target = self._load_case(case)
            per_case.append(self._eval_batch(batch, target, seed=self.seed, case_id=case.case_id,
                                             extra={"split": case.split.value,
                                                    "direction": case.direction.value,
                                                    "dmax_px": case.dmax_px}))
            # Stability over seeds 0..n_seeds-1
            seed_psnrs = []
            for s in range(self.n_seeds):
                case_metrics = self._eval_batch(batch, target, seed=s,
                                                 case_id=case.case_id)
                seed_psnrs.append(float(case_metrics["hole_psnr"]))
            arr = np.asarray(seed_psnrs, dtype=np.float64)
            stability.append({
                "case_id": case.case_id,
                "psnr_mean": float(arr.mean()),
                "psnr_std": float(arr.std()),
                "psnr_per_seed": seed_psnrs,
            })
        return {
            "per_case": per_case,
            "aggregate": _aggregate(per_case),
            "stability": stability,
        }

    def _load_case(self, case: SampleManifest) -> tuple[ConditionBatch, np.ndarray]:
        """Load one real case through the explicitly supplied loader."""
        if self.case_loader is None:  # defensive for direct private calls
            raise ConditionContractError(
                "case_loader is required when evaluating a manifest"
            )
        result = self.case_loader(case)
        if not isinstance(result, tuple) or len(result) != 2:
            raise ConditionContractError(
                "case_loader must return (ConditionBatch, target)"
            )
        batch, target = result
        if not isinstance(batch, ConditionBatch):
            raise ConditionContractError("case_loader returned an invalid ConditionBatch")
        target = np.asarray(target, dtype=np.float32)
        if target.shape != batch.rgb_hole.shape:
            raise ConditionContractError(
                f"case_loader target shape {target.shape} does not match "
                f"condition {batch.rgb_hole.shape}"
            )
        return batch, target

    # ------------------------------------------------------------------
    # Single evaluation
    # ------------------------------------------------------------------
    def _eval_batch(
        self,
        batch: ConditionBatch,
        target: np.ndarray,
        *,
        seed: int,
        case_id: str,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        validate_condition(batch)
        pred = self.predictor(batch, seed=seed)
        _validate_predictor_output(pred, batch)
        # Inpaint for the "composited" version (the final RGB that
        # the user sees). For metrics we measure hole-only PSNR, so
        # the composite is informative for boundary/known metrics.
        composited = inpaint(batch, pred)
        empty = float(batch.hole_mask.sum()) <= 0
        # hole_psnr/empty: pass empty through to record
        psnr_arr = hole_psnr(pred, target, batch.hole_mask, per_case=True)
        boundary_arr = boundary_l1(pred, target, batch.hole_mask, per_case=True)
        known_arr = known_max_error(pred, target, batch.hole_mask, per_case=True)
        rec: Dict[str, Any] = {
            "case_id": case_id,
            "empty": bool(empty),
            "hole_psnr": float(psnr_arr[0]) if not empty else 0.0,
            "boundary_l1": float(boundary_arr[0]),
            "known_max_error": float(known_arr[0]),
            "hole_ratio": float(batch.hole_mask.mean()),
        }
        if extra:
            rec.update(extra)
        rec["composited_max"] = float(composited.max()) if not empty else 0.0
        rec["composited_min"] = float(composited.min()) if not empty else 0.0
        return rec


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _validate_predictor_output(pred: np.ndarray, batch: ConditionBatch) -> None:
    if not isinstance(pred, np.ndarray):
        raise ConditionContractError(
            f"predictor must return a numpy array, got {type(pred).__name__}"
        )
    if pred.dtype != np.float32:
        raise ConditionContractError(
            f"predictor output must be float32, got {pred.dtype}"
        )
    if pred.shape != batch.rgb_hole.shape:
        raise ConditionContractError(
            f"predictor output shape {pred.shape} does not match rgb_hole "
            f"{batch.rgb_hole.shape}"
        )
    if not np.all(np.isfinite(pred)):
        raise ConditionContractError("predictor output contains non-finite values")


def _aggregate(per_case: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    from .metrics import aggregate_per_case
    return aggregate_per_case(list(per_case))


# ---------------------------------------------------------------------------
# Convenience: dummy predictor + manifest runner
# ---------------------------------------------------------------------------


def run_dummy_predictor(
    batch: ConditionBatch, *, seed: int = 0
) -> np.ndarray:
    """Constant-fill predictor: returns ``0.5`` inside the hole.

    Useful for §9.1 acceptance — passes shape/dtype/finite checks
    and produces a meaningful (non-trivial) PSNR for a synthetic
    target.
    """
    rng = np.random.default_rng(seed)
    B, _, H, W = batch.rgb_hole.shape
    fill = np.full((B, 3, H, W), 0.5, dtype=np.float32)
    # Add a small per-call jitter so different seeds produce
    # different predictions (helpful for stability tests).
    jitter = (rng.standard_normal((B, 3, H, W)) * 0.01).astype(np.float32)
    return np.clip(fill + jitter, 0.0, 1.0)


def evaluate_manifest(
    manifest: Sequence[SampleManifest],
    predictor: Predictor,
    *,
    seed: int = 0,
    n_seeds: int = 4,
    H: int = 64,
    B: int = 1,
    hole_spec: str = "thin",
    case_loader: Optional[CaseLoader] = None,
) -> Dict[str, Any]:
    """Convenience wrapper around :class:`FixedEvaluator`."""
    ev = FixedEvaluator(
        predictor=predictor,
        manifest=manifest,
        seed=seed,
        n_seeds=n_seeds,
        H=H,
        B=B,
        hole_spec=hole_spec,
        case_loader=case_loader,
    )
    return ev.run()
