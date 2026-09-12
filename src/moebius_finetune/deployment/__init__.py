"""Deployment subpackage.

Implements the static budget tool, the LPDDR4X bandwidth model, the
ONNX export pipeline, the RKNN entry stub and the four-class
budget-report formatter. All modules are pure-Python / pure-torch
except :mod:`moebius_finetune.deployment.rknn_export`, which only
*attempts* the conversion and falls back to an explicit
"not available" error when the RKNN toolkit is missing.
"""

from __future__ import annotations

from moebius_finetune.deployment.budget import (
    BudgetOperatorReport,
    compare_to_budget,
    compute_algorithm_macs,
    compute_logical_traffic,
    estimate_dram_traffic,
    is_supported_by_rknn,
)
from moebius_finetune.deployment.lpddr import BandwidthModel, latency_lower_bound
from moebius_finetune.deployment.onnx_export import export_onnx
from moebius_finetune.deployment.report import BudgetReport, format_markdown
from moebius_finetune.deployment.rknn_export import (
    RKNNSupportReport,
    RknnNotAvailable,
    export_rknn,
    rknn_support_report,
)

__all__ = [
    "BandwidthModel",
    "BudgetOperatorReport",
    "BudgetReport",
    "RKNNSupportReport",
    "RknnNotAvailable",
    "compare_to_budget",
    "compute_algorithm_macs",
    "compute_logical_traffic",
    "estimate_dram_traffic",
    "export_onnx",
    "export_rknn",
    "format_markdown",
    "is_supported_by_rknn",
    "latency_lower_bound",
    "rknn_support_report",
]
