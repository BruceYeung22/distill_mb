"""On-the-fly GRT 512 case builder: manifest, train provider, eval cases.

TDD2 §3 (owner directives 2026-09-12): **no disparity disk cache** —
ZipDepth runs online per case, and the GRT warp derives the input
tensors from the fresh disparity map. Disparity levels are reduced to
two (D7 rev.2):

* small: ``dmax_px = 16`` @512
* large: ``dmax_px = 32`` @512

with both directions (L2R / R2L). Each training batch (=1 case) draws
one (direction, dmax) pair uniformly at random and builds the whole
case from it.

Case semantics mirror ``pipeline_512.build_512_sample``: p99 is derived
from the 512 disparity, warp displacement = raw disparity × ``dmax_px``,
the depth condition = ``(disparity / p99)`` clipped to [0, 1] with the
L-negative / R-positive sign, hole pixels are exactly zero in
``rgb_hole`` / ``depth_hole``, and case ids follow
``<source_id>__<dmax:02d>_<direction>``.

Split roles: ``train`` (first N images, all four combos),
``holdout_masks`` (same images, all four combos — in-domain reference,
since both dmax levels are trained), ``holdout_images`` (exclusive
trailing images × four combos). Cases with ``hole_ratio`` outside
(0.005, 0.6) are skipped (D8).

The module is torch-free: predictors are injected callables
(``bgr_uint8 (H,W,3) -> disparity (H,W) float``); the real one
(:class:`ZipDepthOnline`) defers the torch / ZipDepth import to first
use.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

import numpy as np
from PIL import Image

from ..contracts import Direction
from .depth_normalization import DepthNormalizationApplier
from .grt import compute_grt_mask


__all__ = [
    "CaseSpec",
    "ZipDepthOnline",
    "resolve_zipdepth_dir",
    "list_image_ids",
    "build_manifest",
    "save_manifest",
    "load_manifest",
    "build_case",
    "hole_ratio_of",
    "GrtTrainProvider",
    "HOLE_RATIO_MIN",
    "HOLE_RATIO_MAX",
    "DISPARITY_COMBOS",
]

HOLE_RATIO_MIN = 0.005
HOLE_RATIO_MAX = 0.6

#: D7 rev.2 — small (16 px) + large (32 px), both directions.
DISPARITY_COMBOS: tuple = ((16, "L2R"), (16, "R2L"), (32, "L2R"), (32, "R2L"))

#: Probed when ``ZIPDEPTH_UPSTREAM_DIR`` is unset (local box first).
_ZIPDEPTH_CANDIDATES = (
    "/home/dog/project/moebius_distill/ZipDepth",
    "/mnt/d/project/moebius_distill/ZipDepth",
    "D:/project/moebius_distill/ZipDepth",
)

#: Image extensions accepted as COCO sources.
_IMAGE_EXTS = {".jpg", ".jpeg", ".png"}

Predictor = Callable[[np.ndarray], np.ndarray]


def resolve_zipdepth_dir() -> Path:
    env = os.environ.get("ZIPDEPTH_UPSTREAM_DIR")
    if env and Path(env).is_dir():
        return Path(env)
    for cand in _ZIPDEPTH_CANDIDATES:
        if Path(cand).is_dir():
            return Path(cand)
    return Path(_ZIPDEPTH_CANDIDATES[0])


def list_image_ids(
    image_dir: Union[str, Path],
    ids: Optional[Sequence[str]] = None,
) -> List[str]:
    if ids is not None:
        return [str(i) for i in ids]
    root = Path(image_dir)
    return sorted(p.stem for p in root.iterdir() if p.suffix.lower() in _IMAGE_EXTS)


class ZipDepthOnline:
    """Lazily-loaded ZipDepth predictor (one instance per process).

    The torch / ZipDepth import happens on the first call, so merely
    constructing this class stays cheap and torch-free. Note (HAMI
    documented side effect): ``DepthInference.__init__`` sets
    ``torch.set_float32_matmul_precision("high")`` globally when
    ``device == "cuda"``.
    """

    def __init__(
        self,
        checkpoint: Optional[str] = None,
        device: str = "cuda",
    ) -> None:
        self._checkpoint = checkpoint
        self._device = device
        self._infer: Optional[Predictor] = None

    def __call__(self, bgr_uint8: np.ndarray) -> np.ndarray:
        if self._infer is None:
            zd = resolve_zipdepth_dir()
            if str(zd) not in sys.path:
                sys.path.insert(0, str(zd))
            from zipdepth.inference.predictor import DepthInference  # noqa: E402

            ckpt = self._checkpoint or str(zd / "checkpoints" / "zipdepth_base.pth")
            infer = DepthInference(checkpoint_path=ckpt, device=self._device)
            self._infer = infer.infer_image
        return self._infer(bgr_uint8)  # type: ignore[misc]


@dataclass(frozen=True)
class CaseSpec:
    source_id: str
    dmax_px: int
    direction: str
    split: str

    @property
    def case_id(self) -> str:
        return f"{self.source_id}__{self.dmax_px:02d}_{self.direction}"


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------


def build_manifest(
    *,
    image_dir: Union[str, Path],
    num_train: int,
    num_holdout_images: int,
    seed: int = 0,
    combos: tuple = DISPARITY_COMBOS,
) -> Dict[str, Any]:
    """Assign split roles over the sorted image ids (deterministic).

    Train = first ``num_train`` ids; ``holdout_images`` = the last
    ``num_holdout_images`` ids (exclusive — never overlapping train);
    ``holdout_masks`` = the train images (in-domain mask reference).
    Every role enumerates all ``combos``.
    """
    ids = list_image_ids(image_dir)
    if num_train + num_holdout_images > len(ids):
        raise ValueError(
            f"num_train({num_train}) + num_holdout_images({num_holdout_images}) "
            f"exceeds available images ({len(ids)})"
        )
    train_ids = ids[:num_train]
    hold_ids = ids[len(ids) - num_holdout_images:] if num_holdout_images else []
    cases: List[CaseSpec] = []
    for iid in train_ids:
        for dmax, direction in combos:
            cases.append(CaseSpec(iid, dmax, direction, "train"))
        for dmax, direction in combos:
            cases.append(CaseSpec(iid, dmax, direction, "holdout_masks"))
    for iid in hold_ids:
        for dmax, direction in combos:
            cases.append(CaseSpec(iid, dmax, direction, "holdout_images"))
    return {
        "seed": int(seed),
        "num_train": int(num_train),
        "num_holdout_images": int(num_holdout_images),
        "disparity_combos": [list(c) for c in combos],
        "cases": [asdict(c) for c in cases],
    }


def save_manifest(manifest: Dict[str, Any], path: Union[str, Path]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8")


def load_manifest(path: Union[str, Path]) -> List[CaseSpec]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return [CaseSpec(**c) for c in data["cases"]]


# ---------------------------------------------------------------------------
# On-the-fly case construction
# ---------------------------------------------------------------------------


def _load_source_rgb01(
    image_dir: Union[str, Path], source_id: str, size: int
) -> np.ndarray:
    root = Path(image_dir)
    src = None
    for ext in (".jpg", ".jpeg", ".png"):
        cand = root / f"{source_id}{ext}"
        if cand.exists():
            src = cand
            break
    if src is None:
        raise FileNotFoundError(f"source image not found for {source_id}")
    with Image.open(src) as im:
        rgb = np.asarray(
            im.convert("RGB").resize((size, size), Image.Resampling.BILINEAR)
        )
    return (rgb.astype(np.float32) / np.float32(255.0)).transpose(2, 0, 1)


def hole_ratio_of(mask_bool: np.ndarray) -> float:
    return float(mask_bool.mean())


def build_case(
    case: CaseSpec,
    *,
    image_dir: Union[str, Path],
    predictor: Predictor,
    size: int = 512,
) -> Dict[str, np.ndarray]:
    """Compute one GRT case's tensors with **online** ZipDepth (TDD2 §3).

    ``predictor`` maps ``bgr_uint8 (H,W,3)`` → disparity ``(H,W)``.

    Returns ``rgb_hole (3,s,s)``, ``hole_mask (1,s,s)``,
    ``depth_hole (1,s,s)`` (hole-zeroed, signed, p99-normalised),
    ``target (3,s,s)`` (the clean source), and scalar ``hole_ratio``.
    """
    rgb = _load_source_rgb01(image_dir, case.source_id, size)
    disp = np.asarray(
        predictor(rgb.transpose(1, 2, 0)[..., ::-1].astype(np.uint8)),
        dtype=np.float32,
    )
    if disp.shape != (size, size):
        raise ValueError(
            f"predictor returned shape {disp.shape}, expected {(size, size)}"
        )
    applier = DepthNormalizationApplier.from_disparity(disp)

    mask_bool = compute_grt_mask(
        (size, size),
        disp * np.float32(case.dmax_px),
        Direction(case.direction),
        dmax_px=int(case.dmax_px),
    )
    mask_f = mask_bool.astype(np.float32)[None]  # (1, s, s)

    sign = -1 if case.direction == "L2R" else 1
    depth_signed = applier.apply(disp, direction_sign=sign)  # (s, s) in [-1, 1]

    return {
        "rgb_hole": (rgb * (1.0 - mask_f)).astype(np.float32),
        "hole_mask": mask_f.astype(np.float32),
        "depth_hole": (depth_signed[None] * (1.0 - mask_f)).astype(np.float32),
        "target": rgb.astype(np.float32),
        "hole_ratio": hole_ratio_of(mask_bool),
    }


def _stable_seed(case_id: str, salt: int) -> int:
    h = hashlib.blake2b(f"{case_id}:{salt}".encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(h, "little") % (2 ** 31 - 1)


class GrtTrainProvider:
    """Random-unified batch provider over one manifest split (torch-free).

    Each call draws one case uniformly at random from the split's case
    list (D7 rev.2: one (direction, dmax) pair per batch), builds it
    with the injected online predictor, and returns the batch dict
    ``training.teacher.finetune`` expects: ``clean_rgb`` (the full
    clean target — the q_sample x0), ``hole_mask``, ``depth_hole``,
    and a per-call deterministic ``noise`` (the ε target). Cases whose
    hole ratio falls outside the acceptance band are redrawn.
    """

    def __init__(
        self,
        cases: Sequence[CaseSpec],
        *,
        image_dir: Union[str, Path],
        predictor: Predictor,
        split: str = "train",
        size: int = 512,
        seed: int = 0,
    ) -> None:
        self.cases = [c for c in cases if c.split == split]
        if not self.cases:
            raise ValueError(f"manifest has no cases for split {split!r}")
        self.image_dir = Path(image_dir)
        self.predictor = predictor
        self.size = int(size)
        self.seed = int(seed)
        self._rng = np.random.default_rng(seed)
        self._salt = 0
        self.skipped = 0
        self.drawn_combos: Dict[str, int] = {}

    def __call__(self) -> Dict[str, np.ndarray]:
        while True:
            spec = self.cases[int(self._rng.integers(len(self.cases)))]
            built = build_case(
                spec,
                image_dir=self.image_dir,
                predictor=self.predictor,
                size=self.size,
            )
            if HOLE_RATIO_MIN < built["hole_ratio"] < HOLE_RATIO_MAX:
                break
            self.skipped += 1
        combo_key = f"{spec.dmax_px}_{spec.direction}"
        self.drawn_combos[combo_key] = self.drawn_combos.get(combo_key, 0) + 1
        self._salt += 1
        noise_rng = np.random.default_rng(_stable_seed(spec.case_id, self._salt))
        noise = noise_rng.standard_normal(
            (1, 4, self.size // 8, self.size // 8)
        ).astype(np.float32)
        return {
            "clean_rgb": built["target"][None].astype(np.float32),
            "hole_mask": built["hole_mask"][None].astype(np.float32),
            "depth_hole": built["depth_hole"][None].astype(np.float32),
            "noise": noise,
        }
