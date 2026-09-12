"""Masked pixel-domain LPIPS (TDD2 §5 / D4).

``masked_lpips`` computes full-image LPIPS after compositing the ground
truth into the *known* region of the prediction, so the only image
difference left is inside the hole — the standard "hole LPIPS" trick.

The ``lpips`` package is imported lazily (CPU test rule); tests inject
a stub model instead.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np


__all__ = ["masked_lpips", "build_lpips_alex"]


def build_lpips_alex(device: str = "cuda") -> Any:
    """Build the pixel LPIPS (AlexNet) model, lazily importing lpips."""
    import torch  # noqa: F401 — ensures the torch dependency is explicit
    import lpips

    model = lpips.LPIPS(net="alex", verbose=False)
    model = model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def masked_lpips(
    pred: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
    model: Optional[Any] = None,
    device: str = "cuda",
) -> float:
    """Hole-restricted LPIPS (batch mean).

    ``pred``/``target`` are ``[B,3,H,W]`` in [0,1]; ``mask`` is
    ``[B,1,H,W]`` in {0,1} (1 = hole). The known region of ``pred`` is
    replaced by ``target`` before scoring, so the value isolates the
    hole. ``model`` may be an injected ``lpips``-compatible callable
    ``(img0, img1) -> [B,1,1,1]`` for tests; the real AlexNet model is
    built lazily otherwise.
    """
    import torch

    p = np.asarray(pred, dtype=np.float32)
    t = np.asarray(target, dtype=np.float32)
    m = np.asarray(mask, dtype=np.float32)
    if p.ndim != 4 or t.ndim != 4 or m.ndim != 4:
        raise ValueError(
            f"pred/target/mask must be 4D [B,C,H,W], got "
            f"{p.shape}/{t.shape}/{m.shape}"
        )
    composited = m * p + (1.0 - m) * t
    # [0,1] → [-1,1], the domain lpips expects.
    img0 = torch.from_numpy(np.ascontiguousarray(composited * 2.0 - 1.0)).to(device)
    img1 = torch.from_numpy(np.ascontiguousarray(t * 2.0 - 1.0)).to(device)
    if model is None:
        model = build_lpips_alex(device)
    with torch.no_grad():
        value = model(img0, img1)
    return float(torch.as_tensor(value).detach().float().mean().item())
