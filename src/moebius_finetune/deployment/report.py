"""Four-class budget report (TDD §8.2).

The report combines:

* **Algorithm MACs** — from :func:`compute_algorithm_macs`.
* **Logical traffic** — from :func:`compute_logical_traffic`.
* **DRAM scenarios** — from :func:`estimate_dram_traffic`, with
  ``reread_factor`` ∈ ``{1.0, 1.5, 2.0}``.
* **Actual tool / device results** — currently a stub: the
  static support report from :func:`rknn_support_report` and a
  ``traffic_device_verified`` flag fixed to ``"unknown"`` until a
  real RK3588 board is available.

The :func:`format_markdown` function returns a Markdown string that
includes all four sections and the per-axis budget compliance.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from moebius_finetune.deployment.budget import (
    compare_to_budget,
    compute_algorithm_macs,
    compute_logical_traffic,
    estimate_dram_traffic,
)
from moebius_finetune.deployment.rknn_export import rknn_support_report


__all__ = ["BudgetReport", "format_markdown"]


@dataclass
class BudgetReport:
    """A combined four-class report.

    The dataclass holds the per-class raw data so the report can be
    inspected programmatically (e.g. by tests). The Markdown
    formatting is provided by :func:`format_markdown`.
    """

    model_name: str
    input_spec: tuple
    algorithm: Dict[str, Any]
    logical: Dict[str, Any]
    dram: Dict[str, Any]
    rknn: Dict[str, Any]
    compliance: Dict[str, Any]

    def to_dict(self) -> dict:
        return {
            "model_name": self.model_name,
            "input_spec": list(self.input_spec),
            "algorithm": self.algorithm,
            "logical": self.logical,
            "dram": self.dram,
            "rknn": self.rknn,
            "compliance": self.compliance,
        }


def make_report(
    model: "torch.nn.Module",  # type: ignore[name-defined]
    *,
    model_name: str = "model",
    input_spec: tuple = (1, 5, 512, 512),
    reread_factor_options: tuple = (1.0, 1.5, 2.0),
    extra_bytes: int = 32 * 1024 * 1024,
    macs_limit_g: float = 8.0,
    traffic_limit_mb: float = 300.0,
) -> BudgetReport:
    """Build a :class:`BudgetReport` for ``model``.

    The function combines all four classes of data and adds the
    static RKNN support report. The :class:`BudgetReport` is what
    :func:`format_markdown` consumes.
    """
    algorithm = compute_algorithm_macs(model, input_spec)
    logical = compute_logical_traffic(model, input_spec)
    dram = estimate_dram_traffic(
        logical["total_bytes"],
        reread_factor_options=reread_factor_options,
        extra_bytes=extra_bytes,
    )
    rknn = rknn_support_report().to_dict()
    # Compose the full report dict used by compare_to_budget.
    full = {
        "total_macs_g": algorithm["total_macs_g"],
        "dram_traffic": dram,
    }
    compliance = compare_to_budget(
        full,
        macs_limit_g=macs_limit_g,
        traffic_limit_mb=traffic_limit_mb,
    )
    return BudgetReport(
        model_name=model_name,
        input_spec=tuple(input_spec),
        algorithm=algorithm,
        logical=logical,
        dram=dram,
        rknn=rknn,
        compliance=compliance,
    )


def format_markdown(report: BudgetReport) -> str:
    """Format a :class:`BudgetReport` as Markdown."""
    algo = report.algorithm
    logical = report.logical
    dram = report.dram
    rknn = report.rknn
    comp = report.compliance

    lines: List[str] = []
    lines.append(f"# Budget report — {report.model_name}")
    lines.append("")
    lines.append(f"Input spec: `{list(report.input_spec)}`")
    lines.append("")
    lines.append("## 1. Algorithm MACs")
    lines.append("")
    lines.append(f"- Total MACs: `{algo['total_macs']:,}` ({algo['total_macs_g']:.4f} GMACs)")
    lines.append(f"- Total weight parameters: `{algo['total_weight_params']:,}`")
    lines.append(f"- Output shape: `{algo['output_shape']}`")
    if algo.get("unsupported_ops"):
        lines.append(f"- Unsupported ops (static table): `{', '.join(algo['unsupported_ops'])}`")
    else:
        lines.append("- Unsupported ops (static table): _none_")
    lines.append("")
    lines.append("Per-op breakdown (top 20 by MACs):")
    lines.append("")
    lines.append("| Op | Input | Output | MACs |")
    lines.append("|---|---|---|---|")
    sorted_ops = sorted(algo["operators"], key=lambda r: -int(r["macs"]))[:20]
    for r in sorted_ops:
        inshape = r["inputs"][0] if r["inputs"] else "-"
        outshape = r["outputs"][0] if r["outputs"] else "-"
        lines.append(
            f"| {r['name']} | `{list(inshape)}` | `{list(outshape)}` | `{r['macs']:,}` |"
        )
    lines.append("")
    lines.append("## 2. Logical traffic")
    lines.append("")
    lines.append("| Item | Bytes | MB |")
    lines.append("|---|---|---|")
    for key in (
        "weights_bytes",
        "input_bytes",
        "output_bytes",
        "skip_lifetime_bytes",
        "resize_bytes",
        "concat_bytes",
        "type_conversion_bytes",
    ):
        v = logical.get(key, 0)
        lines.append(f"| {key} | `{v:,}` | `{v / 1e6:.3f}` |")
    lines.append(
        f"| **total** | `{logical.get('total_bytes', 0):,}` | "
        f"`{logical.get('total_bytes', 0) / 1e6:.3f}` |"
    )
    lines.append("")
    lines.append("Fusion assumptions:")
    for note in logical.get("fusion_assumptions", []):
        lines.append(f"- {note}")
    lines.append("")
    lines.append("## 3. DRAM traffic scenarios")
    lines.append("")
    lines.append("| reread_factor | logical (bytes) | extra (bytes) | estimated (MB) |")
    lines.append("|---|---|---|---|")
    for k, v in dram.items():
        lines.append(
            f"| {v['reread_factor']:.1f} | `{v['logical_bytes']:,}` | "
            f"`{v['extra_bytes']:,}` | `{v['estimated_mb']:.3f}` |"
        )
    lines.append("")
    lines.append("## 4. Actual tool / device results")
    lines.append("")
    lines.append(f"- RKNN target platform: `{rknn['target_platform']}`")
    lines.append(f"- RKNN quantization: `{rknn['quantization']}`")
    if rknn.get("fp16_only"):
        lines.append(f"- FP16-only operators: `{', '.join(rknn['fp16_only'])}`")
    if rknn.get("cpu_only"):
        lines.append(f"- CPU-only operators: `{', '.join(rknn['cpu_only'])}`")
    lines.append("")
    lines.append("## Budget compliance")
    lines.append("")
    lines.append(f"- `algorithm_budget_pass`: `{comp['algorithm_budget_pass']}`")
    lines.append(
        f"- MACs: `{comp['algorithm_macs_g']:.4f}` GMACs vs limit "
        f"`{comp['algorithm_macs_limit_g']:.1f}` GMACs"
    )
    if comp.get("traffic_estimate_pass") is not None:
        lines.append(
            f"- `traffic_estimate_pass`: `{comp['traffic_estimate_pass']}` "
            f"(estimate {comp['traffic_estimate_mb']:.3f} MB at reread 1.5, "
            f"limit {comp['traffic_estimate_limit_mb']:.1f} MB)"
        )
    else:
        lines.append("- `traffic_estimate_pass`: _n/a_")
    lines.append(f"- `traffic_device_verified`: `{comp['traffic_device_verified']}`")
    lines.append("")
    return "\n".join(lines) + "\n"
