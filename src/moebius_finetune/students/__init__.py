"""Students subpackage.

Implements the lightweight student models and the building blocks that
compose them.  See TDD sections 6.1-6.3 and 7.1-7.2 for the structure
and the offline one-step distillation contract.

Public surface (filled in incrementally by Agent C):

* :class:`InvertedResidualBlock` / :class:`ContextBlock` /
  :class:`DownsampleBlock` / :class:`UpsampleBlock` /
  :class:`TailRefineBlock` and the BN-folding utility :func:`fuse_bn`
  (see :mod:`moebius_finetune.students.common`).
* :class:`PixelStudentV0` (see :mod:`moebius_finetune.students.pixel`).
* :class:`LatentStudentV0` together with the
  :class:`LightweightCodec` and :class:`OneStepLatentNet` submodels
  (see :mod:`moebius_finetune.students.latent`).
"""

from __future__ import annotations

from moebius_finetune.students.common import (
    ContextBlock,
    DownsampleBlock,
    InvertedResidualBlock,
    TailRefineBlock,
    UpsampleBlock,
    fuse_bn,
)
from moebius_finetune.students.latent import (
    LatentStudentV0,
    LightweightCodec,
    OneStepLatentNet,
)
from moebius_finetune.students.pixel import PixelStudentV0

__all__ = [
    "ContextBlock",
    "DownsampleBlock",
    "InvertedResidualBlock",
    "LatentStudentV0",
    "LightweightCodec",
    "OneStepLatentNet",
    "PixelStudentV0",
    "TailRefineBlock",
    "UpsampleBlock",
    "fuse_bn",
]
