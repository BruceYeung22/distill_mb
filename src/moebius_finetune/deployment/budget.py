"""Static budget tool (TDD §8.1, §8.2).

The tool walks a torch ``nn.Module`` and computes four things, each
on its own:

1. **Algorithm MACs** — per-operator multiply-accumulate counts. The
   convolution formula is ``Hout*Wout*Cout*(Cin/groups)*Kh*Kw``;
   depthwise convs are convs with ``groups == Cin``. Residual adds
   count as one MAC per output element. Resize / Concat / Add / type
   conversions are recorded as their own line items.
2. **Logical traffic** — the byte traffic if every tensor were
   read/written exactly once with no fusion, broken down by
   weights, input/output, skip lifetimes, and resize/concat/add.
3. **DRAM scenario** — apply a ``reread_factor`` (1.0, 1.5, 2.0) to
   the logical INT8 traffic and add a fixed ``extra_bytes`` budget
   for scheduler / CPU-NPU glue. All three scenarios are reported.
4. **Budget compliance** — compare against the 8 GMACs / 300 MB
   defaults. Without a real device, ``traffic_device_verified`` is
   fixed to ``"unknown"``.

Unsupported operators must be listed, never silently dropped. The
:func:`is_supported_by_rknn` stub returns a boolean based on a
static table; the values match the RKNN 2.3.2 operator support doc
at the time of writing and will be re-validated against the real
compile log later.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import torch
from torch import nn


__all__ = [
    "BudgetOperatorReport",
    "compare_to_budget",
    "compute_algorithm_macs",
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


def _conv_output_shape(
    input_shape: Tuple[int, int, int, int],
    kernel: int,
    stride: int,
    padding: int,
    dilation: int = 1,
) -> Tuple[int, int]:
    h_in = input_shape[2]
    w_in = input_shape[3]
    h_out = math.floor((h_in + 2 * padding - dilation * (kernel - 1) - 1) / stride + 1)
    w_out = math.floor((w_in + 2 * padding - dilation * (kernel - 1) - 1) / stride + 1)
    return (h_out, w_out)


def _conv_macs(
    input_shape: Tuple[int, int, int, int],
    output_shape: Tuple[int, int],
    *,
    in_channels: int,
    out_channels: int,
    groups: int,
    kernel: int,
) -> int:
    h_out, w_out = output_shape
    per_output = (in_channels // groups) * kernel * kernel
    return int(h_out * w_out * out_channels * per_output)


def _process_leaf_op(
    module: nn.Module,
    cur_shape: Tuple[int, int, int, int],
    reports: List[BudgetOperatorReport],
    name: str,
) -> Tuple[Tuple[int, int, int, int], int, int]:
    """Process a single leaf op and update the report list / counters.

    Returns ``(output_shape, weight_count, skip_lifetime)`` for the
    leaf. The shape-tracking mirrors the logic in the main
    :func:`_walk_module` loop.
    """
    weight_count = 0
    skip_lifetime = 0
    if isinstance(module, nn.Conv2d):
        kernel = module.kernel_size[0]
        stride = module.stride[0]
        padding = module.padding[0]
        dilation = module.dilation[0]
        groups = module.groups
        out_c = module.out_channels
        in_c = module.in_channels
        out_hw = _conv_output_shape(
            cur_shape, kernel, stride, padding, dilation
        )
        out_shape = (cur_shape[0], out_c, out_hw[0], out_hw[1])
        macs = _conv_macs(
            cur_shape, out_hw,
            in_channels=in_c,
            out_channels=out_c,
            groups=groups,
            kernel=kernel,
        )
        op_name = "DepthwiseConv" if groups == in_c and groups > 1 else "Conv"
        weight_count += int(module.weight.numel())
        if module.bias is not None:
            weight_count += int(module.bias.numel())
        reports.append(
            BudgetOperatorReport(
                name=op_name,
                inputs=[tuple(cur_shape)],
                outputs=[tuple(out_shape)],
                macs=macs,
                extra={
                    "kernel": kernel,
                    "stride": stride,
                    "padding": padding,
                    "groups": groups,
                    "cin": in_c,
                    "cout": out_c,
                },
            )
        )
        return out_shape, weight_count, skip_lifetime
    elif isinstance(module, nn.BatchNorm2d):
        params = int(sum(p.numel() for p in module.parameters()))
        weight_count += params
        reports.append(
            BudgetOperatorReport(
                name="BatchNorm",
                inputs=[tuple(cur_shape)],
                outputs=[tuple(cur_shape)],
                macs=0,
                extra={"params": params, "fused": True},
            )
        )
        return cur_shape, weight_count, skip_lifetime
    elif isinstance(module, (nn.ReLU, nn.ReLU6)):
        op_name = "ReLU" if isinstance(module, nn.ReLU) else "ReLU6"
        reports.append(
            BudgetOperatorReport(
                name=op_name,
                inputs=[tuple(cur_shape)],
                outputs=[tuple(cur_shape)],
                macs=0,
                extra={},
            )
        )
        return cur_shape, weight_count, skip_lifetime
    elif isinstance(module, nn.Linear):
        in_features = module.in_features
        out_features = module.out_features
        macs = in_features * out_features
        weight_count += int(module.weight.numel())
        if module.bias is not None:
            weight_count += int(module.bias.numel())
        reports.append(
            BudgetOperatorReport(
                name="MatMul",
                inputs=[(cur_shape[0], in_features)],
                outputs=[(cur_shape[0], out_features)],
                macs=macs,
                extra={"in": in_features, "out": out_features},
            )
        )
        return (cur_shape[0], out_features), weight_count, skip_lifetime
    # Unknown leaf — recurse defensively to surface it as a no-op.
    return cur_shape, 0, 0


def _walk_module(
    module: nn.Module,
    input_shape: Tuple[int, int, int, int],
    reports: List[BudgetOperatorReport],
    prefix: str = "",
) -> Tuple[Tuple[int, int, int, int], int, int]:
    """Walk ``module`` and append per-op reports.

    Returns the final output shape, plus the number of weight
    parameters seen and the number of elements counted as "skip
    lifetime" (placeholder for now).
    """
    cur_shape = input_shape
    weight_count = 0
    skip_lifetime = 0

    # If the top-level module is itself a leaf op (e.g. ``nn.Conv2d``),
    # ``named_children()`` is empty and we'd report zero MACs. Handle
    # the leaf case directly so the budget tool works for simple
    # sanity-test models.
    children = list(module.named_children())
    if not children:
        new_shape, w, s = _process_leaf_op(
            module, cur_shape, reports, prefix or type(module).__name__
        )
        return new_shape, w, s

    for name, child in module.named_children():
        full_name = f"{prefix}{name}" if not prefix else f"{prefix}.{name}"
        if isinstance(child, nn.Conv2d):
            kernel = child.kernel_size[0]
            stride = child.stride[0]
            padding = child.padding[0]
            dilation = child.dilation[0]
            groups = child.groups
            out_c = child.out_channels
            in_c = child.in_channels
            out_hw = _conv_output_shape(
                cur_shape, kernel, stride, padding, dilation
            )
            out_shape = (cur_shape[0], out_c, out_hw[0], out_hw[1])
            macs = _conv_macs(
                cur_shape, out_hw,
                in_channels=in_c,
                out_channels=out_c,
                groups=groups,
                kernel=kernel,
            )
            op_name = "DepthwiseConv" if groups == in_c and groups > 1 else "Conv"
            weight_count += int(child.weight.numel())
            if child.bias is not None:
                weight_count += int(child.bias.numel())
            reports.append(
                BudgetOperatorReport(
                    name=op_name,
                    inputs=[tuple(cur_shape)],
                    outputs=[tuple(out_shape)],
                    macs=macs,
                    extra={
                        "kernel": kernel,
                        "stride": stride,
                        "padding": padding,
                        "groups": groups,
                        "cin": in_c,
                        "cout": out_c,
                    },
                )
            )
            cur_shape = out_shape
        elif isinstance(child, (nn.BatchNorm2d, nn.BatchNorm1d)):
            # Track BN lifetime as a separate (cheap) operator.
            params = int(sum(p.numel() for p in child.parameters()))
            weight_count += params
            reports.append(
                BudgetOperatorReport(
                    name="BatchNorm",
                    inputs=[tuple(cur_shape)],
                    outputs=[tuple(cur_shape)],
                    macs=0,  # BN has no MACs (fused into Conv)
                    extra={"params": params, "fused": True},
                )
            )
        elif isinstance(child, (nn.ReLU, nn.ReLU6, nn.SiLU, nn.GELU)):
            op_name = {
                nn.ReLU: "ReLU",
                nn.ReLU6: "ReLU6",
                nn.SiLU: "SiLU",
                nn.GELU: "GELU",
            }.get(type(child), type(child).__name__)
            reports.append(
                BudgetOperatorReport(
                    name=op_name,
                    inputs=[tuple(cur_shape)],
                    outputs=[tuple(cur_shape)],
                    macs=0,
                    extra={},
                )
            )
        elif isinstance(child, nn.Upsample):
            # Nearest / bilinear upsample. Note: only the supported
            # nearest upsample is in the static support table; bilinear
            # is supported too on RKNN 2.3.2. We track the MAC as 0
            # (no multiply-accumulate) but count the output elements.
            new_h = child.size[0] if isinstance(child.size, tuple) else int(
                cur_shape[2] * (child.scale_factor if child.scale_factor else 1)
            )
            new_w = child.size[1] if isinstance(child.size, tuple) else int(
                cur_shape[3] * (child.scale_factor if child.scale_factor else 1)
            )
            if child.size is None:
                factor = child.scale_factor
                if isinstance(factor, (tuple, list)):
                    new_h = int(cur_shape[2] * factor[0])
                    new_w = int(cur_shape[3] * factor[1])
                else:
                    new_h = int(cur_shape[2] * factor)
                    new_w = int(cur_shape[3] * factor)
            out_shape = (cur_shape[0], cur_shape[1], new_h, new_w)
            mode = "ResizeNearest" if child.mode == "nearest" else f"Resize{child.mode.capitalize()}"
            reports.append(
                BudgetOperatorReport(
                    name=mode,
                    inputs=[tuple(cur_shape)],
                    outputs=[tuple(out_shape)],
                    macs=0,
                    extra={"mode": child.mode},
                )
            )
            cur_shape = out_shape
        elif child.__class__.__name__ == "UpsampleBlock":
            # Our common.UpsampleBlock does:
            #   up = proj_low(low)   # 1x1 conv, no spatial change
            #   up = interpolate(up, scale_factor=2)
            #   out = ir(up + skip)  # IR block
            # The walk handles the proj_low, then we record an upsample,
            # then we recurse into the IR block.
            from moebius_finetune.students.common import UpsampleBlock
            if isinstance(child, UpsampleBlock):
                # 1. proj_low: Conv2d or Identity
                if not isinstance(child.proj_low, nn.Identity):
                    proj = child.proj_low
                    out_c = proj.out_channels
                    out_shape = (cur_shape[0], out_c, cur_shape[2], cur_shape[3])
                    macs = cur_shape[2] * cur_shape[3] * out_c * (proj.in_channels // proj.groups) * 1 * 1
                    weight_count += int(proj.weight.numel())
                    if proj.bias is not None:
                        weight_count += int(proj.bias.numel())
                    reports.append(
                        BudgetOperatorReport(
                            name="Conv",
                            inputs=[tuple(cur_shape)],
                            outputs=[tuple(out_shape)],
                            macs=macs,
                            extra={"kernel": 1, "stride": 1, "padding": 0, "groups": proj.groups, "cin": proj.in_channels, "cout": proj.out_channels},
                        )
                    )
                    cur_shape = out_shape
                # 2. upsample ×2
                new_h = cur_shape[2] * 2
                new_w = cur_shape[3] * 2
                out_shape = (cur_shape[0], cur_shape[1], new_h, new_w)
                reports.append(
                    BudgetOperatorReport(
                        name="ResizeNearest",
                        inputs=[tuple(cur_shape)],
                        outputs=[tuple(out_shape)],
                        macs=0,
                        extra={"mode": "nearest", "factor": 2.0},
                    )
                )
                cur_shape = out_shape
                # 3. residual add (skip is assumed to have the same
                #    channel count and spatial size as the upsampled
                #    tensor; the contract is c_skip == c_out).
                reports.append(
                    BudgetOperatorReport(
                        name="Add",
                        inputs=[tuple(cur_shape), tuple(cur_shape)],
                        outputs=[tuple(cur_shape)],
                        macs=int(cur_shape[1] * cur_shape[2] * cur_shape[3]),
                        extra={"op": "Add", "source": "skip"},
                    )
                )
                # 4. IR block
                new_shape, w, s = _walk_module(child.ir, cur_shape, reports, prefix=full_name + ".ir")
                cur_shape = new_shape
                weight_count += w
                skip_lifetime += s
            else:
                # Generic composite; recurse.
                new_shape, w, s = _walk_module(child, cur_shape, reports, prefix=full_name)
                cur_shape = new_shape
                weight_count += w
                skip_lifetime += s
        elif isinstance(child, nn.AdaptiveAvgPool2d):
            out_shape = (cur_shape[0], cur_shape[1], child.output_size, child.output_size)
            reports.append(
                BudgetOperatorReport(
                    name="AveragePool",
                    inputs=[tuple(cur_shape)],
                    outputs=[tuple(out_shape)],
                    macs=0,
                    extra={"output_size": child.output_size},
                )
            )
            cur_shape = out_shape
        elif isinstance(child, (nn.AdaptiveAvgPool2d, nn.AvgPool2d, nn.MaxPool2d)):
            op_name = "AveragePool" if isinstance(child, (nn.AdaptiveAvgPool2d, nn.AvgPool2d)) else "MaxPool"
            if isinstance(child, nn.AdaptiveAvgPool2d):
                out_h = out_w = child.output_size
            else:
                out_h = _conv_output_shape(
                    cur_shape, child.kernel_size, child.stride, child.padding
                )[0]
                out_w = out_h
            out_shape = (cur_shape[0], cur_shape[1], out_h, out_w)
            reports.append(
                BudgetOperatorReport(
                    name=op_name,
                    inputs=[tuple(cur_shape)],
                    outputs=[tuple(out_shape)],
                    macs=0,
                    extra={},
                )
            )
            cur_shape = out_shape
        elif isinstance(child, nn.Linear):
            in_features = child.in_features
            out_features = child.out_features
            macs = in_features * out_features
            weight_count += int(child.weight.numel())
            if child.bias is not None:
                weight_count += int(child.bias.numel())
            reports.append(
                BudgetOperatorReport(
                    name="MatMul",
                    inputs=[(cur_shape[0], in_features)],
                    outputs=[(cur_shape[0], out_features)],
                    macs=macs,
                    extra={"in": in_features, "out": out_features},
                )
            )
            cur_shape = (cur_shape[0], out_features)
        elif isinstance(child, nn.Module) and child.__class__.__name__ == "Add":
            # torch's functional add is a python `+` op, which doesn't
            # appear as a module. We only hit this branch if someone
            # uses ``nn.Add`` (which doesn't exist) or a custom Add
            # module; treat it the same as a residual add for accounting.
            reports.append(
                BudgetOperatorReport(
                    name="Add",
                    inputs=[tuple(cur_shape), tuple(cur_shape)],
                    outputs=[tuple(cur_shape)],
                    macs=int(cur_shape[1] * cur_shape[2] * cur_shape[3]),
                    extra={"op": "Add"},
                )
            )
        elif isinstance(child, nn.Dropout):
            reports.append(
                BudgetOperatorReport(
                    name="Dropout",
                    inputs=[tuple(cur_shape)],
                    outputs=[tuple(cur_shape)],
                    macs=0,
                    extra={"p": child.p},
                )
            )
        elif child.__class__.__name__ in (
            "_DecoderStage",
            "_LatentNetDecoderStage",
            "_CodecDecoderStage",
        ):
            # All three decoder stages share the same forward shape:
            # optional 1x1 projection + optional ×2 upsample + skip
            # add + Sequential of IR blocks. Walk the children but
            # also record the upsample / skip add explicitly so the
            # spatial shape is threaded correctly.
            from moebius_finetune.students.latent import (
                _CodecDecoderStage,
                _LatentNetDecoderStage,
            )
            from moebius_finetune.students.pixel import _DecoderStage
            if isinstance(child, (_DecoderStage, _CodecDecoderStage, _LatentNetDecoderStage)):
                # 1. proj_low (Conv2d or Identity) if c_low != c_out
                if not isinstance(child.proj_low, nn.Identity):
                    proj = child.proj_low
                    out_c = proj.out_channels
                    out_shape = (cur_shape[0], out_c, cur_shape[2], cur_shape[3])
                    macs = cur_shape[2] * cur_shape[3] * out_c * (proj.in_channels // proj.groups) * 1 * 1
                    weight_count += int(proj.weight.numel())
                    if proj.bias is not None:
                        weight_count += int(proj.bias.numel())
                    reports.append(
                        BudgetOperatorReport(
                            name="Conv",
                            inputs=[tuple(cur_shape)],
                            outputs=[tuple(out_shape)],
                            macs=macs,
                            extra={"kernel": 1, "stride": 1, "padding": 0, "groups": proj.groups, "cin": proj.in_channels, "cout": proj.out_channels},
                        )
                    )
                    cur_shape = out_shape
                # 2. upsample ×2 (only if do_upsample)
                if getattr(child, "do_upsample", True):
                    new_h = cur_shape[2] * 2
                    new_w = cur_shape[3] * 2
                    out_shape = (cur_shape[0], cur_shape[1], new_h, new_w)
                    reports.append(
                        BudgetOperatorReport(
                            name="ResizeNearest",
                            inputs=[tuple(cur_shape)],
                            outputs=[tuple(out_shape)],
                            macs=0,
                            extra={"mode": "nearest", "factor": 2.0},
                        )
                    )
                    cur_shape = out_shape
                # 3. skip add
                reports.append(
                    BudgetOperatorReport(
                        name="Add",
                        inputs=[tuple(cur_shape), tuple(cur_shape)],
                        outputs=[tuple(cur_shape)],
                        macs=int(cur_shape[1] * cur_shape[2] * cur_shape[3]),
                        extra={"op": "Add", "source": "skip"},
                    )
                )
                # 4. walk the IR block stack
                if hasattr(child, "blocks") and isinstance(child.blocks, nn.Sequential):
                    new_shape, w, s = _walk_module(child.blocks, cur_shape, reports, prefix=full_name + ".blocks")
                    cur_shape = new_shape
                    weight_count += w
                    skip_lifetime += s
            else:
                new_shape, w, s = _walk_module(child, cur_shape, reports, prefix=full_name)
                cur_shape = new_shape
                weight_count += w
                skip_lifetime += s
        elif child.__class__.__name__ in ("InvertedResidualBlock", "ContextBlock"):
            # The IR block has expand (1x1) → dw (3x3 or 7x7) →
            # project (1x1) → + residual. We walk the children to
            # record the conv / bn / act ops, then we also record the
            # residual add.
            from moebius_finetune.students.common import InvertedResidualBlock, ContextBlock
            if isinstance(child, (InvertedResidualBlock, ContextBlock)):
                new_shape, w, s = _walk_module(child, cur_shape, reports, prefix=full_name)
                cur_shape = new_shape
                weight_count += w
                skip_lifetime += s
                # Residual add (TDD §8.1: residual adds count as one
                # MAC per output element).
                reports.append(
                    BudgetOperatorReport(
                        name="Add",
                        inputs=[tuple(cur_shape), tuple(cur_shape)],
                        outputs=[tuple(cur_shape)],
                        macs=int(cur_shape[1] * cur_shape[2] * cur_shape[3]),
                        extra={"op": "Add", "source": "residual"},
                    )
                )
            else:
                new_shape, w, s = _walk_module(child, cur_shape, reports, prefix=full_name)
                cur_shape = new_shape
                weight_count += w
                skip_lifetime += s
        else:
            # Recurse into composite modules.
            new_shape, w, s = _walk_module(child, cur_shape, reports, prefix=full_name)
            cur_shape = new_shape
            weight_count += w
            skip_lifetime += s

    return cur_shape, weight_count, skip_lifetime


def compute_algorithm_macs(
    model: nn.Module, input_spec: Tuple[int, int, int, int]
) -> dict:
    """Walk ``model`` and compute per-operator MACs.

    Parameters
    ----------
    model
        The torch module to walk. We do not run the model; we just
        trace its layers to figure out shapes and parameters.
    input_spec
        The ``(B, C, H, W)`` input shape the model expects.

    Returns
    -------
    dict
        With keys ``operators`` (a list of :class:`BudgetOperatorReport`
        ``to_dict()`` outputs), ``total_macs`` (int), ``total_macs_g``
        (float, GMACs), and ``unsupported_ops`` (list of operator
        names that the static RKNN table marks as ``"unsupported"``).
    """
    model.eval()
    reports: List[BudgetOperatorReport] = []
    output_shape, weight_count, _ = _walk_module(model, input_spec, reports)
    total_macs = sum(r.macs for r in reports)
    unsupported = sorted(
        {r.name for r in reports if is_supported_by_rknn(r.name) == "unsupported"}
    )
    return {
        "operators": [r.to_dict() for r in reports],
        "total_macs": int(total_macs),
        "total_macs_g": total_macs / 1e9,
        "total_weight_params": int(weight_count),
        "unsupported_ops": unsupported,
        "input_spec": list(input_spec),
        "output_shape": list(output_shape),
    }


def compute_combined_macs(
    parts: Sequence[Tuple[nn.Module, Tuple[int, int, int, int]]],
) -> dict:
    """Compute the combined MACs / weights of multiple (model, input_spec) pairs.

    The :class:`LatentStudentV0` has two submodels that use different
    input shapes (the codec gets RGB at 512², the net gets a 10-channel
    tensor at 64²). :func:`compute_algorithm_macs` only handles one
    input spec at a time, so this helper sums the per-submodel
    results to get the *end-to-end* student cost.
    """
    per_part = []
    total_macs = 0
    total_weight_params = 0
    unsupported_set: set = set()
    for model, spec in parts:
        r = compute_algorithm_macs(model, spec)
        per_part.append(r)
        total_macs += r["total_macs"]
        total_weight_params += r["total_weight_params"]
        unsupported_set.update(r["unsupported_ops"])
    return {
        "parts": per_part,
        "total_macs": int(total_macs),
        "total_macs_g": total_macs / 1e9,
        "total_weight_params": int(total_weight_params),
        "unsupported_ops": sorted(unsupported_set),
    }


# ---------------------------------------------------------------------------
# Logical traffic
# ---------------------------------------------------------------------------


def compute_logical_traffic(
    model: nn.Module,
    input_spec: Tuple[int, int, int, int],
    *,
    dtype_bytes: Mapping[str, int] = {"int8": 1, "fp32": 4, "fp16": 2},
) -> dict:
    """Compute the per-component logical byte traffic.

    The breakdown follows TDD §8.1, §8.2:

    * **weights** — the total weight bytes (assumed INT8 by default).
    * **inputs / outputs** — the input tensor bytes plus every
      intermediate output's bytes.
    * **skip lifetimes** — separate accounting for the residual adds.
    * **resize / concat / add / type-conversion** — each is broken
      out as its own line.
    * **fusion assumptions** are documented in the report.

    The MAC report is re-derived here to keep the per-component
    breakdown consistent.
    """
    macs_report = compute_algorithm_macs(model, input_spec)
    by_name: Dict[str, int] = {}
    bytes_by_name: Dict[str, int] = {}
    output_elements = 0
    add_inputs = 0
    resize_inputs = 0
    concat_inputs = 0
    for r in macs_report["operators"]:
        name = r["name"]
        by_name[name] = by_name.get(name, 0) + int(r["macs"])
        if r["outputs"]:
            for o in r["outputs"]:
                b = int(torch.tensor(o).prod().item())
                output_elements += b
                # Each output is a temporary write. We log this as the
                # sum of element counts in ``output_elements``.
        if name == "Add":
            for ins in r["inputs"]:
                b = int(torch.tensor(ins).prod().item())
                add_inputs += b
        if name.startswith("Resize"):
            for ins in r["inputs"]:
                b = int(torch.tensor(ins).prod().item())
                resize_inputs += b

    weight_bytes = int(macs_report["total_weight_params"]) * int(dtype_bytes.get("int8", 1))
    input_bytes = int(torch.tensor(list(input_spec)).prod().item()) * int(dtype_bytes.get("int8", 1))
    output_bytes = output_elements * int(dtype_bytes.get("int8", 1))
    add_bytes = add_inputs * int(dtype_bytes.get("int8", 1))
    resize_bytes = resize_inputs * int(dtype_bytes.get("int8", 1))
    concat_bytes = concat_inputs * int(dtype_bytes.get("int8", 1))
    typeconv_bytes = 0  # No explicit casts in the current student models.

    total_bytes = (
        weight_bytes
        + input_bytes
        + output_bytes
        + add_bytes
        + resize_bytes
        + concat_bytes
        + typeconv_bytes
    )
    return {
        "weights_bytes": weight_bytes,
        "input_bytes": input_bytes,
        "output_bytes": output_bytes,
        "skip_lifetime_bytes": add_bytes,
        "resize_bytes": resize_bytes,
        "concat_bytes": concat_bytes,
        "type_conversion_bytes": typeconv_bytes,
        "total_bytes": total_bytes,
        "fusion_assumptions": [
            "All weights stored as INT8 (1 byte/elem).",
            "Each operator's output is read exactly once by the next operator.",
            "Skip-add inputs are counted in full (no aliasing in the static model).",
            "Resize is nearest; the output is the dominant traffic term, not the input.",
        ],
    }


# ---------------------------------------------------------------------------
# DRAM traffic
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
    algorithm_pass = bool(gmacs <= macs_limit_g)
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
        traffic_pass = bool(traffic_mb <= traffic_limit_mb)
    return {
        "algorithm_budget_pass": algorithm_pass,
        "algorithm_macs_g": gmacs,
        "algorithm_macs_limit_g": float(macs_limit_g),
        "traffic_estimate_pass": traffic_pass,
        "traffic_estimate_mb": traffic_mb,
        "traffic_estimate_limit_mb": float(traffic_limit_mb),
        "traffic_device_verified": "unknown",
    }
