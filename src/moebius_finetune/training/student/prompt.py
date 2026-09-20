"""Hole-fill convention for the GRT prompt image.

The conditioning image must present the hole as **mid-gray**, not black.

The model's input space is ``[-1, 1]``, and the pretrained
Moebius/PixelHacker was trained with the hole at **0** in that space —
exactly what the project's own teacher-inference reference does
(``training/teacher/cache.py``)::

    masked_image = (2.0 * rgb - 1.0) * (1.0 - mask)     # hole -> 0

TAESDXL consumes ``[0, 1]`` images, where that same value is ``0.5``.

``grt_dataset.build_case`` instead returns ``rgb_hole = rgb * (1 - mask)``
— the hole is ``0`` in ``[0, 1]``, i.e. **black** once mapped to ``[-1, 1]``.
Feeding that to the model makes it leave the hole dark: measured in-hole
L1 over the 32 GRT holdout cases is ``0.562`` with black versus ``0.225``
with mid-gray (same teacher, same noise). ``build_case`` is left untouched
— it is the raw warp product and other consumers rely on its layout — and
the fill is applied here, at the point where a *prompt* is assembled.
"""

from __future__ import annotations

import torch

__all__ = ["HOLE_FILL_VALUE", "build_masked_prompt"]

#: Fill value in the encoder's ``[0, 1]`` input space (0.5 == 0 in [-1, 1]).
HOLE_FILL_VALUE = 0.5


def build_masked_prompt(rgb01: torch.Tensor, mask01: torch.Tensor) -> torch.Tensor:
    """Return the prompt image with hole pixels set to mid-gray.

    ``rgb01`` is the warped image in ``[0, 1]`` (hole zeroed by
    ``build_case``); ``mask01`` is broadcastable and is ``1`` inside the
    hole. The result is ``rgb * (1 - mask) + 0.5 * mask``.
    """
    return rgb01 * (1.0 - mask01) + HOLE_FILL_VALUE * mask01
