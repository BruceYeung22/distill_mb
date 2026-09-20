"""Gated depthwise 7×7 block (MambaOut-style) for the Moebius-small student.

The 64²/32² stages of :class:`moebius_finetune.students.moebius_small.MoebiusSmallStudent`
use no attention at all — attention only pays off at 16², where the token
count is small. This block replaces the transformer's "token mixer" with a
gated depthwise convolution (MambaOut): the gate branch is the only branch
that carries a non-ReLU6 activation (SiLU), keeping the op set NPU-friendly
(1×1 PW, 7×7 DW, elementwise multiply, foldable BatchNorm).

``expand`` is fixed at 1 (the branch splits the projected tensor into two
``channels``-wide halves, so the 1×1 projection is ``2c`` wide); a larger
expand would multiply the MAC cost of every 64²/32² block.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["GatedDW7Block"]


class GatedDW7Block(nn.Module):
    """``x + pw2(a * silu(dw(b)))`` with ``(a, b) = pw1(norm(x)).chunk(2)``.

    Parameters (``c = channels``): BatchNorm ``2c``, ``pw1`` ``2c²``,
    ``dw`` ``49c``, ``pw2`` ``c²`` → ``3c² + 51c``. All convolutions are
    bias-free; only the gate branch uses SiLU.
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        if channels <= 0:
            raise ValueError(f"channels must be positive, got {channels}")
        self.channels = int(channels)
        self.norm = nn.BatchNorm2d(channels)
        self.pw1 = nn.Conv2d(channels, 2 * channels, 1, bias=False)
        self.dw = nn.Conv2d(
            channels, channels, 7, padding=3, groups=channels, bias=False
        )
        self.pw2 = nn.Conv2d(channels, channels, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        a, b = self.pw1(h).chunk(2, dim=1)
        return self.pw2(a * F.silu(self.dw(b))) + x
