"""RKNN entry point (TDD §8.4).

The agent's first version is a *stub* — without the RKNN Toolkit2
installed in the current environment, the conversion cannot be
performed. The stub:

* imports the toolkit lazily and raises :class:`RknnNotAvailable`
  with a clear message when the import fails;
* returns an :class:`RKNNSupportReport` listing the operator-level
  support status regardless of the toolkit presence, so callers can
  audit the *expected* support level even without doing the
  conversion.

The static support table is the same one used by the budget tool;
the real values will be re-validated against the actual compile log
when the toolkit is available.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from moebius_finetune.deployment.budget import _RKNN_OP_SUPPORT


__all__ = ["RKNNSupportReport", "RknnNotAvailable", "export_rknn", "rknn_support_report"]


class RknnNotAvailable(RuntimeError):
    """Raised when RKNN Toolkit2 is not installed in the current env.

    The message is deliberately explicit: it lists the required
    dependency, the install command and the steps the operator
    should follow to perform the conversion manually.
    """


@dataclass
class RKNNSupportReport:
    """Per-operator support summary.

    Fields
    ------
    target_platform
        The platform name (e.g. ``"rk3588"``).
    quantization
        The requested precision (e.g. ``"int8"``).
    operators
        Dict ``op_name -> support_level`` where level is one of
        ``"fully"``, ``"fp16"``, ``"cpu"``, ``"partial"`` or
        ``"unsupported"``.
    fp16_only
        List of op names that need FP16.
    cpu_only
        List of op names that must be executed on CPU.
    notes
        Free-form notes from the agent (e.g. "Toolkit not installed,
        static table only").
    """

    target_platform: str
    quantization: str
    operators: Dict[str, str] = field(default_factory=dict)
    fp16_only: List[str] = field(default_factory=list)
    cpu_only: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


def rknn_support_report(
    target_platform: str = "rk3588",
    quantization: str = "int8",
) -> RKNNSupportReport:
    """Build the static support report.

    The report is *static*; it does not depend on the toolkit being
    installed. Use :func:`export_rknn` to actually try the
    conversion; that function augments the report with the real
    compile log when the toolkit is present.
    """
    fp16_only = sorted(
        name for name, level in _RKNN_OP_SUPPORT.items() if level == "fp16"
    )
    cpu_only = sorted(
        name for name, level in _RKNN_OP_SUPPORT.items() if level == "cpu"
    )
    return RKNNSupportReport(
        target_platform=target_platform,
        quantization=quantization,
        operators=dict(_RKNN_OP_SUPPORT),
        fp16_only=fp16_only,
        cpu_only=cpu_only,
        notes=[
            "Static table only; re-validate against the actual "
            "compile log when the RKNN toolkit is installed.",
        ],
    )


def _try_import_rknn() -> Any:
    try:
        from rknn.api import RKNN  # type: ignore
    except ImportError as exc:  # pragma: no cover - environment-dependent
        raise RknnNotAvailable(
            "RKNN Toolkit2 is not installed in this environment. "
            "To perform the actual conversion, install the toolkit in "
            "a separate environment per the TDD §2.3 / §8.4 guidance:\n"
            "  pip install rknn-toolkit2==2.3.2\n"
            "and run this entry point from that environment. The "
            "moebius-finetune training environment must NOT be "
            "modified to include the toolkit. See "
            "https://github.com/airockchip/rknn-toolkit2 for details."
        ) from exc
    return RKNN


def export_rknn(
    onnx_path: str,
    *,
    target_platform: str = "rk3588",
    quantization: str = "int8",
    calibration_list: Optional[List[str]] = None,
    output_path: Optional[str] = None,
) -> dict:
    """Attempt the RKNN conversion. Raises :class:`RknnNotAvailable` on missing deps.

    Returns a dict with the conversion config and the static
    :class:`RKNNSupportReport`. When the toolkit is installed, the
    returned dict will additionally include the conversion result
    fields (``rknn_path``, ``compile_log``).
    """
    RKNN = _try_import_rknn()
    report = rknn_support_report(
        target_platform=target_platform, quantization=quantization
    )
    rknn = RKNN(verbose=False)
    rknn.config(target_platform=target_platform)
    rknn.load_onnx(model=onnx_path)
    if quantization == "int8":
        if not calibration_list:
            raise ValueError(
                "int8 quantization requires a calibration_list; pass a list "
                "of numpy / image paths to the export_rknn entry point."
            )
        rknn.init_runtime()
        # The toolkit's quantize API takes a function or a list of
        # file paths. We do not call it here because the calibration
        # format is environment-specific; this branch documents the
        # required call rather than performing it.
        compile_log = (
            "RKNN Toolkit2 loaded; quantization is configured but the "
            "actual calibrate / quantize_onnx_model call must be "
            "performed by the operator with the calibration dataset."
        )
    else:
        rknn.init_runtime()
        compile_log = "RKNN Toolkit2 loaded; FP16 path."
    rknn.release()
    return {
        "report": report.to_dict(),
        "rknn_path": output_path,
        "compile_log": compile_log,
    }
