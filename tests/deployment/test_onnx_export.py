"""Tests for the ONNX export entry (TDD §8.4)."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch import nn

from moebius_finetune.deployment.onnx_export import (
    ONNXExportReport,
    export_onnx,
    round_trip_onnx,
)
from moebius_finetune.students import PixelStudentV0




onnxruntime = pytest.importorskip("onnxruntime")


class _TinyNet(nn.Module):
    """A simple 5-channel to 3-channel conv at 64x64 for the export test."""

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(5, 3, kernel_size=3, padding=1)

    def forward(self, condition: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        # We use only `condition` so the output is deterministic; the
        # noise input is required by the contract but not used here.
        return self.conv(condition)


def test_onnx_export_writes_file(tmp_path: Path):
    model = _TinyNet().eval()
    cond = torch.randn(1, 5, 64, 64)
    noise = torch.randn(1, 4, 8, 8)
    out = tmp_path / "tiny.onnx"
    rep = export_onnx(
        model,
        (cond, noise),
        str(out),
        input_names=["cond", "noise"],
        output_names=["rgb"],
        opset=18,
    )
    assert out.is_file()
    assert rep.path == str(out)
    assert rep.opset == 18
    assert rep.has_random_nodes is False
    assert "cond" in rep.input_names
    assert "noise" in rep.input_names


def test_onnx_round_trip_close_to_torch():
    """The exported ONNX produces the same output as the torch model
    within ``atol=1e-5`` (FP32)."""
    torch.manual_seed(0)
    model = _TinyNet().eval()
    cond = torch.randn(1, 5, 64, 64)
    noise = torch.randn(1, 4, 8, 8)
    with tempfile_TemporaryDirectory() as tmp:
        out = Path(tmp) / "tiny.onnx"
        export_onnx(model, (cond, noise), str(out), opset=18)
        rt = round_trip_onnx(model, (cond, noise), str(out))
    if not rt.get("available", False):
        pytest.skip("onnxruntime not available")
    assert rt["max_abs_diff"] is not None
    assert rt["max_abs_diff"] < 1e-5


def test_onnx_graph_has_no_random_nodes():
    """The exported graph must not contain ``RandomUniform`` or
    ``Dropout`` operators (TDD §8.4)."""
    try:
        import onnx  # type: ignore
    except ImportError:
        pytest.skip("onnx not installed")
    model = _TinyNet().eval()
    cond = torch.randn(1, 5, 64, 64)
    noise = torch.randn(1, 4, 8, 8)
    with tempfile_TemporaryDirectory() as tmp:
        out = Path(tmp) / "tiny.onnx"
        export_onnx(model, (cond, noise), str(out), opset=18)
        m = onnx.load(str(out))
    bad = {"RandomUniform", "RandomNormal", "Dropout"}
    found = [n.op_type for n in m.graph.node if n.op_type in bad]
    assert found == []


def test_onnx_export_pixel_student_at_512(tmp_path: Path):
    """The full PixelStudent can be exported at 512x512 with a noise
    input at 64x64."""
    torch.manual_seed(0)
    model = PixelStudentV0().eval()
    cond = torch.randn(1, 5, 512, 512)
    noise = torch.randn(1, 4, 64, 64)
    out = tmp_path / "pixel.onnx"
    rep = export_onnx(model, (cond, noise), str(out), opset=18)
    assert rep.has_random_nodes is False
    assert rep.opset == 18


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


import tempfile
from contextlib import contextmanager


@contextmanager
def tempfile_TemporaryDirectory():
    """Tiny wrapper so the tests above read as a one-liner."""
    with tempfile.TemporaryDirectory() as tmp:
        yield tmp
