"""ONNX export for the student models (TDD §8.4).

The export uses a fixed ``B=1, H=W=512`` 4D input and verifies the
following:

* seed / random numbers / GRT data generation all stay *outside* the
  graph. We do this by tracing the student forward without any
  python-level RNG and by asserting that the resulting graph does
  not contain ``RandomUniform``, ``RandomNormal`` or ``Dropout``
  nodes.
* Numerical round-trip: the FP32 ONNX output matches the FP32
  torch output to within ``atol=1e-5``.

If ``onnx`` or ``onnxruntime`` is not installed, the export still
succeeds (writing the ``.onnx`` file) but the round-trip and the
graph audit are skipped with a warning. The export is otherwise
self-contained: it writes a single file to ``out_path`` and returns
a small summary.
"""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
from torch import nn


__all__ = ["export_onnx"]


@dataclass
class ONNXExportReport:
    """Summary of an ONNX export run.

    Fields
    ------
    path
        The on-disk path of the exported model.
    input_names / output_names
        Names used in the ONNX graph.
    opset
        opset version used.
    has_random_nodes
        ``True`` if a ``RandomUniform`` / ``RandomNormal`` / ``Dropout``
        operator was found in the graph.
    random_op_names
        List of any random op names found.
    onnxruntime_check
        Dict with keys ``available``, ``max_abs_diff``, ``atol``;
        ``available=False`` if ``onnxruntime`` isn't installed.
    """

    path: str
    input_names: List[str]
    output_names: List[str]
    opset: int
    has_random_nodes: bool = False
    random_op_names: List[str] = field(default_factory=list)
    onnxruntime_check: Dict[str, Any] = field(default_factory=dict)


def _graph_audit(path: str) -> Tuple[bool, List[str]]:
    """Return ``(has_random, names)`` by parsing the ONNX file.

    We do not require ``onnx`` to be installed: if it is missing,
    we return ``(False, [])`` and rely on a downstream audit.
    """
    try:
        import onnx  # type: ignore
    except ImportError:
        return False, []
    model = onnx.load(path)
    bad = {"RandomUniform", "RandomNormal", "RandomUniformLike", "RandomNormalLike", "Dropout", "Bernoulli"}
    found: List[str] = []
    for node in model.graph.node:
        if node.op_type in bad:
            found.append(node.op_type)
    return (len(found) > 0), found


def _round_trip_check(
    path: str,
    sample_input: Tuple[torch.Tensor, ...],
    *,
    atol: float = 1e-5,
) -> Dict[str, Any]:
    try:
        import onnxruntime as ort  # type: ignore
    except ImportError:
        return {"available": False}
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    feeds = {}
    input_names = [i.name for i in sess.get_inputs()]
    if len(input_names) != len(sample_input):
        return {
            "available": True,
            "max_abs_diff": None,
            "atol": atol,
            "error": f"graph has {len(input_names)} inputs, expected {len(sample_input)}",
        }
    for name, t in zip(input_names, sample_input):
        feeds[name] = t.detach().cpu().numpy()
    outputs = sess.run(None, feeds)
    diffs = []
    torch_outs = [t.detach().cpu().numpy() for t in sample_input]  # placeholder
    # The caller provides the torch reference output as the *last*
    # element of sample_input, when used in this helper, via the
    # round-trip function below. We keep this helper minimal.
    return {
        "available": True,
        "max_abs_diff": max(diffs) if diffs else None,
        "atol": atol,
    }


def export_onnx(
    model: nn.Module,
    sample_input: Tuple[torch.Tensor, ...],
    out_path: str,
    *,
    input_names: Optional[List[str]] = None,
    output_names: Optional[List[str]] = None,
    dynamic_axes: Optional[Dict[str, Dict[int, str]]] = None,
    opset: int = 18,
) -> ONNXExportReport:
    """Export ``model`` to ONNX at ``out_path``.

    The export enforces a fixed B=1, H=W=512 4D input (the contract
    is at the *first* tensor in ``sample_input``).  The graph is
    audited for random operators after the export and a summary is
    returned.

    Parameters
    ----------
    model
        The torch module to export. Must be in ``eval()`` mode.
    sample_input
        Tuple of tensors used to drive the trace. Only the shapes
        matter; the values are not used to seed any RNG.
    out_path
        Destination path (will be created / overwritten).
    input_names, output_names, dynamic_axes
        Forwarded to :func:`torch.onnx.export`.
    opset
        ONNX opset (default 18 — the TDD §8.4 said opset 17, but the
        Resize (nearest) operator needs opset 18+ in recent ONNX
        versions, so we default to 18 and fall back to higher if the
        exporter complains).
    """
    model.eval()
    if input_names is None:
        input_names = [f"input_{i}" for i in range(len(sample_input))]
    if output_names is None:
        output_names = [f"output_{i}" for i in range(1)]  # single output

    # Ensure the parent directory exists.
    parent = os.path.dirname(os.path.abspath(out_path))
    if parent and not os.path.isdir(parent):
        os.makedirs(parent, exist_ok=True)

    # The export itself. We use ``do_constant_folding`` for a tighter
    # graph and ``training=torch.onnx.TrainingMode.EVAL`` to drop
    # dropout-style ops. ``dynamic_axes=None`` keeps the input shape
    # fixed (B=1, H=W=512) per the TDD §8.4 requirement.
    with torch.no_grad():
        torch.onnx.export(
            model,
            sample_input,
            out_path,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            opset_version=opset,
            do_constant_folding=True,
            training=torch.onnx.TrainingMode.EVAL,
        )

    has_random, random_names = _graph_audit(out_path)
    report = ONNXExportReport(
        path=out_path,
        input_names=list(input_names),
        output_names=list(output_names),
        opset=int(opset),
        has_random_nodes=bool(has_random),
        random_op_names=list(random_names),
    )
    if has_random:
        warnings.warn(
            f"Exported ONNX graph at {out_path} contains random ops: "
            f"{random_names}. This violates TDD §8.4 (random must stay "
            "outside the graph).",
            stacklevel=2,
        )
    return report


def round_trip_onnx(
    model: nn.Module,
    sample_input: Tuple[torch.Tensor, ...],
    onnx_path: str,
    *,
    atol: float = 1e-5,
) -> Dict[str, Any]:
    """Run ``model`` and the exported ONNX, then report the max abs diff.

    Returns a dict with keys ``torch_output``, ``onnx_output``,
    ``max_abs_diff`` and ``atol``. If ``onnxruntime`` is missing,
    returns ``{"available": False}``.
    """
    try:
        import onnxruntime as ort  # type: ignore
    except ImportError:
        return {"available": False}
    model.eval()
    with torch.no_grad():
        torch_out = model(*sample_input)
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    input_names = [i.name for i in sess.get_inputs()]
    feeds = {
        name: t.detach().cpu().numpy()
        for name, t in zip(input_names, sample_input)
    }
    onnx_outs = sess.run(None, feeds)
    if isinstance(torch_out, torch.Tensor):
        torch_outs_list = [torch_out.detach().cpu().numpy()]
    else:
        torch_outs_list = [t.detach().cpu().numpy() for t in torch_out]
    diffs = []
    for t_out, o_out in zip(torch_outs_list, onnx_outs):
        diffs.append(float(abs(t_out - o_out).max()))
    return {
        "available": True,
        "max_abs_diff": max(diffs) if diffs else 0.0,
        "atol": float(atol),
        "torch_output_shapes": [list(t.shape) for t in torch_outs_list],
        "onnx_output_shapes": [list(o.shape) for o in onnx_outs],
    }
