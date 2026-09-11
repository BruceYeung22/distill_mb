"""Tests for the moebius-finetune package.

Only CPU-only, numpy + stdlib tests live at the top level. Tests that
need torch / onnx / rknn must live under the owning subpackage and
skip themselves when those dependencies are unavailable.
"""
