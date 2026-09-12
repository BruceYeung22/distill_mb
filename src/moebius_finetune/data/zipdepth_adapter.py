"""ZipDepth cache adapter.

Reads pre-computed disparity maps from disk in the format Hami's
``cache_coco32_grt_disparity.py`` writes. The default extension is
``.pt`` (torch tensor) and ``.npy`` (numpy). This module is
**read-only**; it does not own model loading or inference.

Path convention
---------------

* ``disparity_path_for(image_id)`` returns the canonical on-disk
  path for a given source id, assuming Hami's layout::

    <data_root>/disparity/<image_id>.pt   (preferred)
    <data_root>/disparity/<image_id>.npy  (fallback)

* ``load_disparity(path)`` reads a single file and returns a
  ``(H, W)`` float32 array.

If neither file exists, :class:`FileNotFoundError` is raised with a
helpful message that points to the ZipDepth cache step.
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

import numpy as np


__all__ = [
    "ZipDepthAdapter",
    "load_disparity",
    "disparity_path_for",
]


#: Default sub-directory name (matches Hami's cache script).
DEFAULT_DISPARITY_SUBDIR = "disparity"


def disparity_path_for(
    image_id: str,
    *,
    data_root: Union[str, Path],
    subdir: str = DEFAULT_DISPARITY_SUBDIR,
    prefer_torch: bool = True,
) -> Path:
    """Return the canonical disparity file path for ``image_id``.

    By convention, the file is named ``<image_id>.<ext>``. With
    ``prefer_torch=True`` (default), ``.pt`` is preferred and
    ``.npy`` is the fallback. The returned path is what *would* be
    loaded; existence is not checked here.
    """
    root = Path(data_root) / subdir
    if prefer_torch:
        pt = root / f"{image_id}.pt"
        npy = root / f"{image_id}.npy"
        if pt.exists():
            return pt
        if npy.exists():
            return npy
        # Default to the .pt path so the error message matches what
        # the user expects to produce.
        return pt
    npy = root / f"{image_id}.npy"
    if npy.exists():
        return npy
    return root / f"{image_id}.pt"


def load_disparity(
    path: Union[str, Path],
    *,
    allow_torch: bool = True,
) -> np.ndarray:
    """Load a disparity file as a ``(H, W)`` float32 array.

    Supports ``.npy`` (pure numpy) and ``.pt`` (torch tensor, only
    if torch is importable in this process — torch is loaded lazily
    so this module stays torch-free for tests that only need the
    numpy path).

    Parameters
    ----------
    path : str or Path
        File to load. Must be ``.npy`` or ``.pt``.
    allow_torch : bool
        If ``False``, ``.pt`` files raise :class:`FileNotFoundError`
        with a hint to use ``.npy``. Default: ``True``.

    Returns
    -------
    np.ndarray
        ``(H, W)`` float32 array. Leading batch / channel dims of
        length 1 are squeezed.

    Raises
    ------
    FileNotFoundError
        If the file does not exist or its format is unsupported.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"disparity cache not found: {p}\n"
            "Run the ZipDepth cache step first (Hami's "
            "scripts/cache_coco32_grt_disparity.py or the local "
            "prepare-data command)."
        )
    suffix = p.suffix.lower()
    if suffix == ".npy":
        arr = np.load(p)
    elif suffix == ".pt":
        if not allow_torch:
            raise FileNotFoundError(
                f"disparity cache is a .pt file but torch is disabled: {p}"
            )
        try:
            import torch  # type: ignore
        except ImportError as exc:  # pragma: no cover - import guard
            raise FileNotFoundError(
                f"disparity cache {p} is a .pt file; torch is not importable. "
                "Re-cache as .npy or install torch."
            ) from exc
        tensor = torch.load(p, map_location="cpu", weights_only=True)
        if hasattr(tensor, "detach"):
            arr = tensor.detach().cpu().numpy()
        else:
            arr = np.asarray(tensor)
    else:
        raise FileNotFoundError(
            f"unsupported disparity extension {suffix!r}: {p}"
        )

    arr = np.asarray(arr)
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]
    elif arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr[..., 0]
    if arr.ndim != 2:
        raise FileNotFoundError(
            f"disparity cache {p} has unsupported shape {arr.shape}; "
            "expected (H, W) or (1, H, W)"
        )
    return arr.astype(np.float32, copy=False)


class ZipDepthAdapter:
    """Lightweight wrapper around :func:`load_disparity`.

    Holds the ``data_root`` and default ``subdir`` so callers can
    ask for a disparity by source id without re-deriving the path.
    """

    def __init__(
        self,
        data_root: Union[str, Path],
        subdir: str = DEFAULT_DISPARITY_SUBDIR,
        prefer_torch: bool = True,
    ) -> None:
        self.data_root = Path(data_root)
        self.subdir = subdir
        self.prefer_torch = prefer_torch

    def path_for(self, image_id: str) -> Path:
        return disparity_path_for(
            image_id,
            data_root=self.data_root,
            subdir=self.subdir,
            prefer_torch=self.prefer_torch,
        )

    def load(self, image_id: str) -> np.ndarray:
        return load_disparity(
            self.path_for(image_id),
            allow_torch=self.prefer_torch,
        )
