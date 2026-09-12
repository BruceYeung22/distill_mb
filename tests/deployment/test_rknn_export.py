"""Tests for the RKNN entry stub (TDD §8.4)."""

from __future__ import annotations

import builtins
import sys
from pathlib import Path

import pytest

from moebius_finetune.deployment.rknn_export import (
    RKNNSupportReport,
    RknnNotAvailable,
    export_rknn,
    rknn_support_report,
)




def test_rknn_support_report_fp16_only_isolated():
    """The report separates FP16-only ops from CPU-only ops."""
    rpt = rknn_support_report()
    # FP16-only: normalisation layers.
    assert "LayerNorm" in rpt.fp16_only
    assert "GroupNorm" in rpt.fp16_only
    # CPU-only: rare / dynamic ops.
    assert "Gather" in rpt.cpu_only
    assert "TopK" in rpt.cpu_only
    # The two lists do not overlap.
    assert set(rpt.fp16_only).isdisjoint(set(rpt.cpu_only))


def test_rknn_support_report_default_platform_rk3588():
    rpt = rknn_support_report()
    assert rpt.target_platform == "rk3588"
    assert rpt.quantization == "int8"


def test_export_rknn_raises_when_toolkit_missing(monkeypatch):
    """Without the RKNN toolkit, ``export_rknn`` raises ``RknnNotAvailable``."""
    # Block the rknn import by making the import fail.
    import importlib

    def _raise(name, *args, **kwargs):
        if name.startswith("rknn"):
            raise ImportError("simulated missing rknn")
        return importlib.import_module(name, *args, **kwargs)

    monkeypatch.setattr(importlib, "import_module", _raise)
    with pytest.raises(RknnNotAvailable) as exc:
        export_rknn("does_not_matter.onnx")
    msg = str(exc.value)
    assert "RKNN" in msg
    assert "pip" in msg or "install" in msg


def test_rknn_not_available_error_message_includes_install_hint():
    """The error message is actionable: it tells the user how to install."""
    # Simulate the missing toolkit by attempting the import directly.
    try:
        from rknn.api import RKNN  # type: ignore  # noqa: F401
        pytest.skip("RKNN toolkit is installed in this environment")
    except ImportError:
        pass
    with pytest.raises(RknnNotAvailable) as exc:
        from moebius_finetune.deployment.rknn_export import _try_import_rknn
        _try_import_rknn()
    msg = str(exc.value)
    assert "rknn-toolkit2" in msg
    # The error message should mention the toolkit, the install command,
    # and the steps to follow. It does not need to mention the target
    # platform by name (the entry point is platform-configurable).
    assert "pip" in msg
    assert "install" in msg
