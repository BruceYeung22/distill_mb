"""E-LatentLPIPS checkpoint staging (network-free on this box).

``elatentlpips`` fetches two files from ``huggingface.co`` into
``./ckpt/`` relative to the process CWD:

``<encoder>_latest_vgg16_tuned.pth``
    The tuned model (trunk + linear heads), loaded with ``strict=True``.
``<encoder>_latent_vgg16.pth.tar``
    The ImageNet-initialised ``LatentVGG16BN`` trunk used as a starting
    point — every one of its weights is overwritten by the tuned load.

``huggingface.co`` is unreachable from this box, but ``hf-mirror.com``
serves the same paths, so the tuned file is fetched from the mirror by
:func:`fetch_tuned_checkpoint`. The 1.1 GB ``.pth.tar`` is *not* fetched:
:func:`ensure_ckpt_dir` synthesises it by extracting the ``net.*``
entries of the tuned checkpoint, which is exactly the state the real
download would leave behind after the strict load (the file only exists
to satisfy the library's unconditional download branch).

Both files are staged as symlinks so any CWD can satisfy the library's
``./ckpt/<name>`` lookup.
"""

from __future__ import annotations

import os
import subprocess
from typing import Optional

import torch

__all__ = [
    "DEFAULT_CKPT_DIR",
    "CKPT_SUBDIR",
    "ensure_ckpt_dir",
    "fetch_tuned_checkpoint",
    "load_elatentlpips",
]

DEFAULT_CKPT_DIR = os.environ.get(
    "ELATENTLPIPS_CKPT_DIR",
    "/home/dog/datasets/moebius_finetune/elatentlpips_ckpt",
)
CKPT_SUBDIR = "ckpt"
ENCODER = "sdxl"
TUNED_NAME = f"{ENCODER}_latest_vgg16_tuned.pth"
TAR_NAME = f"{ENCODER}_latent_vgg16.pth.tar"
MIRROR = "https://hf-mirror.com/Mingguksky/elatentlpips/resolve/main/elatentlpips_ckpt"


def fetch_tuned_checkpoint(ckpt_dir: Optional[str] = None) -> str:
    """Download the tuned checkpoint from the mirror if it is not cached."""
    root = ckpt_dir or DEFAULT_CKPT_DIR
    os.makedirs(root, exist_ok=True)
    dst = os.path.join(root, TUNED_NAME)
    if os.path.exists(dst) and os.path.getsize(dst) > 0:
        return dst
    subprocess.run(
        ["curl", "-sL", "-C", "-", "-o", dst, f"{MIRROR}/{TUNED_NAME}"],
        check=True,
    )
    return dst


def _write_trunk_shim(tuned_path: str, ckpt_dir: str) -> str:
    """Extract the ``net.sliceK.<i>.*`` entries as a ``LatentVGG16BN`` trunk tar.

    The slice index in the tuned checkpoint *is* the feature index of
    ``vgg16_bn`` (the slices are contiguous ranges of ``features``), so
    ``net.sliceK.<i>.<rest>`` maps to ``model.features.<i>.<rest>``.
    The dead ``model.classifier.*`` head is deliberately omitted — the
    calibrated wrapper only ever consumes ``model.features``.
    """
    tar_path = os.path.join(ckpt_dir, TAR_NAME)
    if os.path.exists(tar_path) and os.path.getsize(tar_path) > 0:
        return tar_path
    state = torch.load(tuned_path, map_location="cpu", weights_only=False)
    trunk = {}
    for key, value in state.items():
        if not key.startswith("net.slice"):
            continue
        _, _, index, rest = key.split(".", 3)
        trunk[f"model.features.{index}.{rest}"] = value
    if not trunk:
        raise RuntimeError(f"no net.slice* weights found in {tuned_path}")
    torch.save({"state_dict": trunk, "best_acc1": 0.0}, tar_path)
    return tar_path


def ensure_ckpt_dir(cwd: Optional[str] = None, ckpt_dir: Optional[str] = None) -> str:
    """Stage both checkpoint files under ``<cwd>/ckpt`` and return that path."""
    cwd = cwd or os.getcwd()
    root = ckpt_dir or DEFAULT_CKPT_DIR
    tuned = fetch_tuned_checkpoint(root)
    tar = _write_trunk_shim(tuned, root)

    target = os.path.join(cwd, CKPT_SUBDIR)
    os.makedirs(target, exist_ok=True)
    for src in (tuned, tar):
        link = os.path.join(target, os.path.basename(src))
        if not os.path.exists(link):
            os.symlink(src, link)
    return target


def load_elatentlpips(*, cwd: Optional[str] = None, device: str = "cpu"):
    """Return a frozen, eval-mode E-LatentLPIPS (SDXL latent space).

    The trunk-tar load is relaxed to ``strict=False`` for the duration of
    construction only: the shim carries every ``features`` entry, and the
    omitted ``classifier`` head is never executed by the calibrated
    wrapper. The authoritative load is the library's own ``strict=True``
    pass over the tuned checkpoint, which runs unchanged right after.

    The StyleGAN2-derived CUDA plugin used by the ``bg`` augment is
    disabled: it is JIT-compiled with ninja at first use, and the package's
    reference implementation (plain PyTorch ops) is numerically equivalent.
    """
    ensure_ckpt_dir(cwd=cwd)
    from elatentlpips import ELatentLPIPS
    from elatentlpips.style_ops import upfirdn2d
    from elatentlpips.vgg16 import LatentVGG16BN

    upfirdn2d._init = lambda: False
    original = LatentVGG16BN.load_state_dict

    def _lenient(self, state_dict, *args, **kwargs):
        kwargs["strict"] = False
        return original(self, state_dict, *args, **kwargs)

    LatentVGG16BN.load_state_dict = _lenient
    try:
        model = ELatentLPIPS(encoder=ENCODER, augment="bg")
    finally:
        LatentVGG16BN.load_state_dict = original
    return model.eval().to(device).requires_grad_(False)
