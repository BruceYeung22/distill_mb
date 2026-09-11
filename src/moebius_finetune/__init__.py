"""moebius-finetune: DIBR hole-filling with depth-conditioned Moebius + RK3588 INT8 student.

See the project README and ``docs/architecture.md`` for an overview. The
public types and helpers live in :mod:`moebius_finetune.contracts`.
"""

from __future__ import annotations

from .contracts import (
    ConditionBatch,
    ConditionContractError,
    CoordFrame,
    DepthNormalization,
    Direction,
    SampleManifest,
    Split,
    inpaint,
    to_torch,
    validate_condition,
)

__all__ = [
    "ConditionBatch",
    "ConditionContractError",
    "CoordFrame",
    "DepthNormalization",
    "Direction",
    "SampleManifest",
    "Split",
    "inpaint",
    "to_torch",
    "validate_condition",
]
