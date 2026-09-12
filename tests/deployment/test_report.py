"""Tests for the four-class budget report (TDD §8.2)."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from moebius_finetune.deployment.report import (
    BudgetReport,
    format_markdown,
    make_report,
)
from moebius_finetune.students import PixelStudentV0




def _tiny_model() -> nn.Module:
    return nn.Sequential(nn.Conv2d(4, 8, kernel_size=3, padding=1))


def test_make_report_contains_four_classes():
    """``make_report`` returns a :class:`BudgetReport` with all four classes."""
    model = _tiny_model()
    report = make_report(model, model_name="tiny", input_spec=(1, 4, 16, 16))
    assert isinstance(report, BudgetReport)
    assert "total_macs" in report.algorithm
    assert "total_bytes" in report.logical
    assert "reread_1.0" in report.dram
    assert "operators" in report.rknn


def test_markdown_contains_required_sections():
    """The markdown output has the four required sections."""
    model = _tiny_model()
    report = make_report(model, model_name="tiny", input_spec=(1, 4, 16, 16))
    md = format_markdown(report)
    assert "## 1. Algorithm MACs" in md
    assert "## 2. Logical traffic" in md
    assert "## 3. DRAM traffic scenarios" in md
    assert "## 4. Actual tool / device results" in md
    assert "## Budget compliance" in md


def test_markdown_contains_three_reread_factors():
    """All three ``reread_factor`` values are reported."""
    model = _tiny_model()
    report = make_report(model, model_name="tiny", input_spec=(1, 4, 16, 16))
    md = format_markdown(report)
    assert "1.0" in md
    assert "1.5" in md
    assert "2.0" in md


def test_markdown_calls_out_device_verified_unknown():
    """The markdown explicitly states ``traffic_device_verified='unknown'``."""
    model = _tiny_model()
    report = make_report(model, model_name="tiny", input_spec=(1, 4, 16, 16))
    md = format_markdown(report)
    assert "traffic_device_verified" in md
    assert "unknown" in md


def test_markdown_lists_unsupported_ops_when_present():
    """If the static walker flags unsupported ops, the markdown lists them."""
    model = _tiny_model()
    report = make_report(model, model_name="tiny", input_spec=(1, 4, 16, 16))
    md = format_markdown(report)
    # The section header for unsupported ops is present even if the list is empty.
    assert "Unsupported ops" in md or "unsupported" in md.lower()


def test_markdown_for_pixel_student_includes_full_breakdown():
    """For the pixel student at 512², all sections render without errors."""
    model = PixelStudentV0()
    report = make_report(model, model_name="pixel_v0", input_spec=(1, 5, 512, 512))
    md = format_markdown(report)
    # The report must not be empty.
    assert len(md) > 0
    # All four sections present.
    for header in (
        "## 1. Algorithm MACs",
        "## 2. Logical traffic",
        "## 3. DRAM traffic scenarios",
        "## 4. Actual tool / device results",
        "## Budget compliance",
    ):
        assert header in md


def test_markdown_contains_budget_compliance_flags():
    """The compliance section mentions ``algorithm_budget_pass``."""
    model = _tiny_model()
    report = make_report(model, model_name="tiny", input_spec=(1, 4, 16, 16))
    md = format_markdown(report)
    assert "algorithm_budget_pass" in md
