"""Execution-based algorithm and logical traffic budgets.

Trace an isolated CPU eval copy at the requested input shape. Count convolution
and matrix MACs plus elementwise arithmetic, and list other kernels separately.
Logical INT8 traffic counts every executed tensor read/write without fusion;
DRAM scenarios are estimates, never device measurements. Unknown work prevents
budget compliance. RKNN support labels remain an unverified static reference.
"""
from __future__ import annotations

import math
import inspect
import copy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import torch
from torch import nn
from torch.utils._python_dispatch import TorchDispatchMode


__all__ = [
    "BudgetOperatorReport",
    "compare_to_budget",
    "compute_algorithm_macs",
    "compute_combined_macs",
    "compute_logical_traffic",
    "estimate_dram_traffic",
    "is_supported_by_rknn",
]


# ---------------------------------------------------------------------------
# Static operator support table (RKNN 2.3.2 reference; re-validate later).
# ---------------------------------------------------------------------------


# Each entry is one of:
#   "fully"   - native RK3588 NPU operator
#   "fp16"    - supported on NPU but only in FP16
#   "cpu"     - executed on CPU; tensor copy required
#   "partial" - supported with constraints
#   "unsupported" - cannot be lowered; falls back to CPU
_RKNN_OP_SUPPORT: Dict[str, str] = {
    "Conv": "fully",
    "ConvTranspose": "fully",
    "DepthwiseConv": "fully",
    "Add": "fully",
    "Mul": "fully",
    "ReLU": "fully",
    "ReLU6": "fully",
    "Clip": "fully",
    "Sigmoid": "fully",
    "Tanh": "fully",
    "Concat": "fully",
    "ResizeNearest": "fully",
    "ResizeBilinear": "fully",
    "AveragePool": "fully",
    "MaxPool": "fully",
    "Reshape": "fully",
    "Transpose": "fully",
    "Softmax": "fully",
    "LayerNorm": "fp16",
    "GroupNorm": "fp16",
    "InstanceNorm": "fp16",
    "BatchNorm": "fully",  # when fused into a Conv (the only mode we use)
    "Gather": "cpu",
    "TopK": "cpu",
    "NonMaxSuppression": "cpu",
    "RandomUniform": "unsupported",  # in-graph RNG is forbidden by the TDD
    "Dropout": "unsupported",  # should be eliminated in inference
    "OneHot": "cpu",
    "Einsum": "partial",
}


def is_supported_by_rknn(op_name: str) -> str:
    """Return one of the support level strings for ``op_name``."""
    return _RKNN_OP_SUPPORT.get(op_name, "unsupported")


# ---------------------------------------------------------------------------
# Operator report dataclass
# ---------------------------------------------------------------------------


@dataclass
class BudgetOperatorReport:
    """Per-operator breakdown of MACs and tensor shapes."""

    name: str  # e.g. "Conv", "DepthwiseConv", "Add", "Resize"
    inputs: List[Tuple[int, ...]]
    outputs: List[Tuple[int, ...]]
    macs: int
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "inputs": [list(t) for t in self.inputs],
            "outputs": [list(t) for t in self.outputs],
            "macs": int(self.macs),
            "extra": dict(self.extra),
        }


# ---------------------------------------------------------------------------
# Algorithm MACs
# ---------------------------------------------------------------------------


def estimate_dram_traffic(
    logical_bytes: int,
    *,
    reread_factor_options: Sequence[float] = (1.0, 1.5, 2.0),
    extra_bytes: int = 32 * 1_000_000,
) -> dict:
    """Estimate DRAM traffic for one or more ``reread_factor`` values.

    Returns a dict with one entry per ``reread_factor`` and the
    total for each. The default ``extra_bytes`` is 32 MB
    (TDD §8.2; MB is decimal — 10^6 bytes — per TDD §8.1).
    """
    out: Dict[str, Dict[str, int]] = {}
    for rf in reread_factor_options:
        est = int(math.ceil(logical_bytes * float(rf) + int(extra_bytes)))
        out[f"reread_{rf:.1f}"] = {
            "reread_factor": float(rf),
            "logical_bytes": int(logical_bytes),
            "extra_bytes": int(extra_bytes),
            "estimated_bytes": est,
            "estimated_mb": est / 1e6,
        }
    return out


# ---------------------------------------------------------------------------
# Budget compliance
# ---------------------------------------------------------------------------


def compare_to_budget(
    report: Mapping[str, Any],
    *,
    macs_limit_g: float = 8.0,
    traffic_limit_mb: float = 300.0,
) -> dict:
    """Compare a report to the default budget envelope.

    The function returns a small dict with the per-axis pass/fail and
    the device-verified status, which is fixed to ``"unknown"`` in
    this static version.
    """
    gmacs = float(report.get("total_macs_g", 0.0))
    complete = report.get("accounting_complete", True)
    algorithm_pass = bool(complete and gmacs <= macs_limit_g)
    # For the traffic estimate, use the default 1.5x scenario.
    traffic_mb: Optional[float] = None
    dram = report.get("dram_traffic")
    if isinstance(dram, dict):
        for k, v in dram.items():
            if k == "reread_1.5":
                traffic_mb = float(v["estimated_mb"])  # type: ignore[index]
                break
    traffic_pass: Optional[bool] = None
    if traffic_mb is not None:
        traffic_pass = bool(complete and traffic_mb <= traffic_limit_mb)
    return {
        "algorithm_budget_pass": algorithm_pass,
        "algorithm_macs_g": gmacs,
        "algorithm_macs_limit_g": float(macs_limit_g),
        "accounting_complete": bool(complete),
        "traffic_estimate_pass": traffic_pass,
        "traffic_estimate_mb": traffic_mb,
        "traffic_estimate_limit_mb": float(traffic_limit_mb),
        "traffic_device_verified": "unknown",
    }


# ---------------------------------------------------------------------------

def _tensors(value):
    if isinstance(value, torch.Tensor):
        return [value]
    if isinstance(value, dict):
        value = value.values()
    if isinstance(value, (tuple, list)) or type(value).__name__ == "dict_values":
        return [t for item in value for t in _tensors(item)]
    return []


def _storage(tensor):
    return (tensor.device, tensor.untyped_storage().data_ptr())


class _ExecutionTrace(TorchDispatchMode):
    """Record kernels once, including functional calls and repeated module calls."""
    def __init__(self, weights):
        super().__init__()
        self.weights = weights
        self.operators = []
        self.unknown = set()

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        result = func(*args, **kwargs)
        raw = func._schema.name.split("::")[-1]
        ins = _tensors(args) + _tensors(kwargs)
        outs = _tensors(result)
        # Metadata-only aliases do not materialize an activation.
        views = {"view", "_unsafe_view", "reshape", "transpose", "t", "permute",
                 "detach", "alias", "expand", "slice", "select", "as_strided",
                 "unbind", "squeeze", "unsqueeze", "lift_fresh"}
        if raw in views:
            return result
        kind, macs = None, 0
        n = outs[0].numel() if outs else 0
        if raw in {"convolution", "_convolution"}:
            x, weight = args[:2]
            transposed, groups = bool(args[6]), int(args[8])
            if transposed:
                kind = "ConvTranspose"
                macs = x.numel() * math.prod(weight.shape[1:])
            else:
                kind = "DepthwiseConv" if groups == x.shape[1] and groups > 1 else "Conv"
                macs = n * math.prod(weight.shape[1:])
        elif raw in {"mm", "bmm", "addmm", "linear"}:
            kind = "MatMul"
            matrix = args[1] if raw == "addmm" else args[0]
            macs = n * matrix.shape[-1]
        elif raw in {"add", "add_", "sub", "sub_", "mul", "mul_", "div", "div_"}:
            kind = {"add": "Add", "sub": "Sub", "mul": "Mul", "div": "Div"}[raw.rstrip("_")]
            macs = n
        elif raw.startswith("upsample_nearest"):
            kind = "ResizeNearest"
        elif raw.startswith("upsample_bilinear"):
            kind = "ResizeBilinear"
        elif raw in {"relu", "relu_", "hardtanh", "hardtanh_", "clamp", "clamp_", "sigmoid", "tanh"}:
            kind = {"relu": "ReLU", "hardtanh": "ReLU6", "clamp": "Clip", "sigmoid": "Sigmoid", "tanh": "Tanh"}[raw.rstrip("_")]
        elif raw in {"native_batch_norm", "_native_batch_norm_legit", "_native_batch_norm_legit_no_training"}:
            kind = "BatchNorm"
        elif raw == "cat":
            kind = "Concat"
        elif raw in {"_to_copy", "copy_", "clone", "contiguous", "lift_fresh_copy"}:
            kind = "TypeConversion" if raw == "_to_copy" else "Copy"
        elif raw in {"zeros", "zeros_like", "ones", "ones_like", "full", "full_like", "empty", "empty_like", "empty_strided", "new_empty", "new_zeros"}:
            kind = "Allocation"
        elif raw.startswith("rand") or raw in {"bernoulli", "bernoulli_", "normal_", "uniform_"}:
            kind = "RandomUniform"
            self.unknown.add(str(func))
        if kind is None:
            kind = "Unknown"
            self.unknown.add(str(func))
        # Tensor factory shape templates are metadata, not data reads.
        read_inputs = [] if kind == "Allocation" else ins
        weights = sum(t.numel() for t in read_inputs if _storage(t) in self.weights)
        reads = sum(t.numel() for t in read_inputs if _storage(t) not in self.weights)
        writes = sum(t.numel() for t in outs)
        self.operators.append(BudgetOperatorReport(
            kind, [tuple(t.shape) for t in ins], [tuple(t.shape) for t in outs], int(macs),
            {"aten_op": str(func), "activation_read_elements": reads,
             "weight_read_elements": weights, "write_elements": writes,
             "source_read_bytes": sum(t.numel()*t.element_size() for t in read_inputs),
             "source_write_bytes": sum(t.numel()*t.element_size() for t in outs),
             "unaccounted": kind == "Unknown" or kind == "RandomUniform"},
        ).to_dict())
        return result


def _default_inputs(model, spec):
    from moebius_finetune.students import LatentStudentV0
    b, c, h, w = spec
    if isinstance(model, LatentStudentV0):
        import numpy as np
        from moebius_finetune.contracts import ConditionBatch
        condition = ConditionBatch.from_arrays(
            np.zeros((b,3,h,w), np.float32), np.ones((b,1,h,w), np.float32),
            np.zeros((b,1,h,w), np.float32), np.zeros((b,4,h//8,w//8), np.float32))
        return condition, condition.noise
    parameter = next(model.parameters(), None)
    dtype = parameter.dtype if parameter is not None else torch.float32
    inputs = [torch.zeros(spec, dtype=dtype)]
    required = [p for p in inspect.signature(model.forward).parameters.values()
                if p.default is inspect.Parameter.empty and p.kind in
                (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)]
    for p in required[1:]:
        if "noise" not in p.name.lower():
            raise ValueError(f"Provide an input adapter for required argument {p.name!r}")
        inputs.append(torch.zeros((b,4,h//8,w//8), dtype=dtype))
    return tuple(inputs)


def compute_algorithm_macs(model, input_spec, *, adapter=None):
    """Trace the supplied shape through an isolated CPU eval copy.

    ``adapter(copy, input_spec)`` may return positional forward inputs. MACs
    include conv/matmul and one operation per arithmetic output element;
    activation/normalization/resize work is listed separately with zero MACs.
    Transposed convolution counts dense input scatter MACs before cropping.
    Unknown costs fail closed. Only the executed path is covered.
    """
    traced = copy.deepcopy(model).cpu().eval()
    weight_storages = {_storage(t) for t in list(traced.parameters()) + list(traced.buffers())}
    trace = _ExecutionTrace(weight_storages)
    output = None
    try:
        with torch.random.fork_rng(devices=[]), torch.no_grad():
            inputs = adapter(traced, input_spec) if adapter else _default_inputs(traced, tuple(input_spec))
            if isinstance(inputs, torch.Tensor):
                inputs = (inputs,)
            with trace:
                output = traced(*inputs)
        from moebius_finetune.students import LatentStudentV0
        if isinstance(traced, LatentStudentV0):
            # Current full wrapper performs preprocessing in NumPy, outside ATen.
            trace.unknown.add("NumPy preprocessing and host copies outside tensor trace")
    except Exception as exc:
        trace.unknown.add(f"execution:{type(exc).__name__}: {exc}")
    total = sum(op["macs"] for op in trace.operators)
    final_tensors = _tensors(output)
    output_shape = list(final_tensors[0].shape) if final_tensors else list(getattr(output, "shape", ()))
    unsupported = {op["name"] for op in trace.operators if is_supported_by_rknn(op["name"]) == "unsupported"}
    return {"operators": trace.operators, "total_macs": total, "total_macs_g": total/1e9,
            "total_weight_params": sum(p.numel() for p in traced.parameters()),
            "unsupported_ops": sorted(unsupported | trace.unknown), "unknown_ops": sorted(trace.unknown),
            "accounting_complete": not trace.unknown, "input_spec": list(input_spec), "output_shape": output_shape,
            "scope": "Executed PyTorch eval path; CPU/NPU placement and fusion require compiler/device validation."}


def compute_logical_traffic(model, input_spec, *, dtype_bytes=None, adapter=None):
    """Unfused scenario: every executed operator reads inputs and writes outputs.

    Shared weights occupy storage once but are read on every execution. Views
    cost no transfer. All tensors use the requested INT8 scenario width; casts
    retain their actual source/destination widths. No device claim is made.
    """
    bpe = int((dtype_bytes or {"int8": 1}).get("int8", 1))
    r = compute_algorithm_macs(model, input_spec, adapter=adapter)
    reads = writes = weight_reads = casts = 0
    subsets = {"skip_lifetime_bytes": 0, "resize_bytes": 0, "concat_bytes": 0}
    for op in r["operators"]:
        e = op["extra"]
        if op["name"] == "TypeConversion":
            casts += e["source_read_bytes"] + e["source_write_bytes"]
            continue
        reads += e["activation_read_elements"] * bpe
        writes += e["write_elements"] * bpe
        weight_reads += e["weight_read_elements"] * bpe
        key = "skip_lifetime_bytes" if op["name"] == "Add" else "resize_bytes" if op["name"].startswith("Resize") else "concat_bytes" if op["name"] == "Concat" else None
        if key:
            subsets[key] += e["activation_read_elements"] * bpe
    return {"weights_bytes": r["total_weight_params"]*bpe, "weight_read_bytes": weight_reads,
            "input_bytes": reads, "output_bytes": writes, **subsets, "type_conversion_bytes": casts,
            "total_bytes": weight_reads+reads+writes+casts,
            "accounting_complete": r["accounting_complete"], "unknown_ops": r["unknown_ops"],
            "fusion_assumptions": ["No fusion; every kernel input read and output write is counted.",
                "Weights_bytes is storage; weight_read_bytes includes repeated execution and buffers.",
                "Skip/resize/concat rows are subsets of input_bytes, not additive extras.",
                "All non-cast tensors use the INT8 scenario width, including unsupported operators; compiler mixed precision can increase traffic.",
                "Unknown costs prevent a budget pass; actual device DRAM traffic remains unverified."]}


def compute_combined_macs(parts, *, adapters=None):
    """Sum explicitly supplied executions; this does not infer a full pipeline."""
    reports = [compute_algorithm_macs(m, spec, adapter=adapters[i] if adapters else None)
               for i, (m, spec) in enumerate(parts)]
    params = {id(p): p for m, _ in parts for p in m.parameters()}
    total = sum(r["total_macs"] for r in reports)
    return {"parts": reports, "total_macs": total, "total_macs_g": total/1e9,
            "total_weight_params": sum(p.numel() for p in params.values()),
            "unsupported_ops": sorted({op for r in reports for op in r["unsupported_ops"]}),
            "unknown_ops": sorted({op for r in reports for op in r["unknown_ops"]}),
            "accounting_complete": all(r["accounting_complete"] for r in reports)}
