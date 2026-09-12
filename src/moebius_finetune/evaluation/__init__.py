"""Evaluation subpackage.

Owned by Agent A. Provides:

* :mod:`moebius_finetune.evaluation.metrics` — hole PSNR,
  boundary L1, known max error, and per-case aggregations.
* :mod:`moebius_finetune.evaluation.evaluator` — fixed-seed
  evaluator with a dummy-predictor interface for §9.1 acceptance.
* :mod:`moebius_finetune.evaluation.plots` — eight-column
  comparison figure.
"""

from .metrics import (
    aggregate_per_case,
    boundary_l1,
    global_l1,
    hole_l1,
    hole_psnr,
    known_max_error,
)
from .evaluator import (
    FixedEvaluator,
    evaluate_manifest,
    run_dummy_predictor,
)
from .plots import plot_comparison_grid
from .lpips_masked import masked_lpips
from .teacher_eval import aggregate_rows, pick_cases, run_evaluation


__all__ = [
    "hole_psnr",
    "boundary_l1",
    "known_max_error",
    "aggregate_per_case",
    "FixedEvaluator",
    "evaluate_manifest",
    "run_dummy_predictor",
    "plot_comparison_grid",
    "masked_lpips",
    "pick_cases",
    "aggregate_rows",
    "run_evaluation",
]
