"""Manifest I/O for the moebius-finetune pipeline.

Three responsibilities:

* :func:`load_manifest` / :func:`save_manifest` — round-trip a
  list of :class:`SampleManifest` to JSON. Paths are always stored
  as **relative** to the data root, never as absolute paths.
* :func:`from_hami_320` — migrate the existing 320 Hami manifest
  (e.g. ``benchmarks/coco32_grt/manifest.json``) into the
  contracts :class:`SampleManifest` form.
* :func:`to_relative` — utility for normalising any path string
  against a data root. Uses ``pathlib`` exclusively (no
  ``os.chdir``).

Field validation
----------------

Both loaders use :class:`ConditionContractError` (from the public
contracts) so the calling code does not need a second exception
type. Required fields are checked explicitly; unknown fields are
ignored to allow forward-compatible additions by B/C.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List, Mapping, Union

from ..contracts import (
    ConditionContractError,
    CoordFrame,
    DepthNormalization,
    Direction,
    SampleManifest,
    Split,
)


__all__ = [
    "load_manifest",
    "save_manifest",
    "to_relative",
    "from_hami_320",
]


_REQUIRED_FIELDS = (
    "case_id",
    "source_id",
    "split",
    "coord_frame",
    "rgb_path",
    "depth_path",
    "mask_path",
    "target_path",
    "direction",
    "dmax_px",
)


def to_relative(path: Union[str, Path], rel_to: Union[str, Path]) -> str:
    """Return ``path`` as a POSIX-style string relative to ``rel_to``.

    Uses :mod:`pathlib` only; does not call :func:`os.chdir`. When
    ``path`` cannot be expressed relative to ``rel_to`` (e.g. on a
    different drive on Windows), it is returned unchanged with a
    warning-style string preserved as a portable absolute POSIX path
    only when no relative form is possible — but the API is
    intended for paths that *do* live under ``rel_to``.

    Empty / equal paths are normalised to ``"."``.
    """
    p = Path(path)
    r = Path(rel_to)
    if not p.is_absolute():
        # Already relative — keep it as a POSIX string.
        return p.as_posix()
    try:
        rel = p.resolve().relative_to(r.resolve())
    except ValueError:
        # Different drive / outside the root — preserve the absolute
        # path as POSIX. Callers that need a hard-fail should check
        # the result.
        return p.as_posix()
    rel_str = rel.as_posix()
    return rel_str or "."


# ---------------------------------------------------------------------------
# Load / save
# ---------------------------------------------------------------------------


def _coerce_manifest_entry(raw: Mapping[str, object]) -> SampleManifest:
    missing = [k for k in _REQUIRED_FIELDS if k not in raw]
    if missing:
        raise ConditionContractError(
            f"manifest entry missing required field(s): {missing}; "
            f"got keys={list(raw.keys())}"
        )
    try:
        split = Split(str(raw["split"]))
    except ValueError as exc:
        raise ConditionContractError(
            f"invalid split {raw['split']!r}; expected one of {[s.value for s in Split]}"
        ) from exc
    try:
        coord_frame = CoordFrame(str(raw["coord_frame"]))
    except ValueError as exc:
        raise ConditionContractError(
            f"invalid coord_frame {raw['coord_frame']!r}"
        ) from exc
    try:
        direction = Direction(str(raw["direction"]))
    except ValueError as exc:
        raise ConditionContractError(
            f"invalid direction {raw['direction']!r}"
        ) from exc
    dmax_px = raw["dmax_px"]
    if not isinstance(dmax_px, (int, float)) or isinstance(dmax_px, bool):
        raise ConditionContractError(
            f"dmax_px must be a number, got {type(dmax_px).__name__}"
        )
    depth_norm_raw = raw.get("depth_normalization")
    if isinstance(depth_norm_raw, Mapping):
        try:
            depth_normalization = DepthNormalization(
                p99=float(depth_norm_raw.get("p99", 1.0)),
                sign_convention=str(
                    depth_norm_raw.get("sign_convention", "L_neg_R_pos")
                ),
            )
        except (TypeError, ValueError) as exc:
            raise ConditionContractError(
                f"invalid depth_normalization: {depth_norm_raw!r}"
            ) from exc
    elif depth_norm_raw is None:
        depth_normalization = DepthNormalization(p99=1.0)
    else:
        raise ConditionContractError(
            f"depth_normalization must be a mapping or null, got {type(depth_norm_raw).__name__}"
        )

    return SampleManifest(
        case_id=str(raw["case_id"]),
        source_id=str(raw["source_id"]),
        split=split,
        coord_frame=coord_frame,
        rgb_path=str(raw["rgb_path"]),
        depth_path=str(raw["depth_path"]),
        mask_path=str(raw["mask_path"]),
        target_path=str(raw["target_path"]),
        direction=direction,
        dmax_px=int(dmax_px),
        depth_normalization=depth_normalization,
    )


def load_manifest(path: Union[str, Path]) -> List[SampleManifest]:
    """Load a list of :class:`SampleManifest` from a JSON file.

    The JSON may be either a top-level list of manifest dicts or a
    dict with a ``"cases"`` key (Hami-style manifest); the
    convenience wrapper picks the right shape automatically.
    """
    p = Path(path)
    if not p.exists():
        raise ConditionContractError(f"manifest file not found: {p}")
    data = json.loads(p.read_text(encoding="utf-8"))
    if isinstance(data, list):
        entries = data
    elif isinstance(data, dict) and "cases" in data:
        entries = data["cases"]
    else:
        raise ConditionContractError(
            f"manifest must be a list or have a 'cases' key, got {type(data).__name__}"
        )
    return [_coerce_manifest_entry(e) for e in entries]


def save_manifest(
    samples: Iterable[SampleManifest],
    path: Union[str, Path],
    *,
    data_root: Union[str, Path] | None = None,
) -> None:
    """Serialise ``samples`` to JSON.

    All paths inside each :class:`SampleManifest` are written as
    POSIX strings. When ``data_root`` is given, the function
    additionally re-bases any absolute path it encounters so the
    written file is always portable.

    The output is a top-level list of plain dicts (one per case)
    to keep the format simple for tooling.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    out: List[dict] = []
    root = Path(data_root).resolve() if data_root is not None else None
    for s in samples:
        d = s.to_dict()
        for key in ("rgb_path", "depth_path", "mask_path", "target_path"):
            v = d[key]
            if not isinstance(v, str):
                raise ConditionContractError(
                    f"{key} must be a string, got {type(v).__name__}"
                )
            # Reject absolute paths outright; the contract forbids
            # them in portable manifests.
            if Path(v).is_absolute():
                if root is None:
                    raise ConditionContractError(
                        f"absolute path {v!r} in manifest but no data_root was given to "
                        "resolve it; pass data_root= to save_manifest"
                    )
                rel = to_relative(v, root)
                if Path(rel).is_absolute():
                    raise ConditionContractError(
                        f"path {v!r} is not under data_root {root}; refusing to write absolute"
                    )
                d[key] = rel
            else:
                # Keep the POSIX form so cross-platform reads are stable.
                d[key] = Path(v).as_posix()
        out.append(d)
    p.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Hami 320 migration
# ---------------------------------------------------------------------------


#: Mapping from Hami's split string to the contracts :class:`Split`.
_HAMI_SPLIT_TO_CONTRACT = {
    "train": Split.TRAIN,
    "holdout_masks": Split.HOLDOUT_MASKS,
    "holdout_images": Split.HOLDOUT_IMAGES,
}

#: Mapping from Hami's direction string to the contracts :class:`Direction`.
_HAMI_DIR_TO_CONTRACT = {
    "L": Direction.L2R,
    "R": Direction.R2L,
    "L2R": Direction.L2R,
    "R2L": Direction.R2L,
}


def from_hami_320(path: Union[str, Path]) -> List[SampleManifest]:
    """Convert Hami's 320 GRT manifest to a list of :class:`SampleManifest`.

    Hami's manifest has two relevant structures: a top-level
    ``"images"`` list (per-source-image records) and a top-level
    ``"cases"`` list (per-(image, scale, direction)). For each
    case, the migration:

    * picks up the case's ``case_id`` / ``split`` / ``dmax_px`` /
      ``direction``;
    * resolves the case's ``image_path`` and ``mask_path``;
    * looks up the source image's ``depth_stats.p99`` and records
      it as :class:`DepthNormalization`;
    * invents a deterministic ``target_path`` placeholder because
      the Hami manifest does not pre-compute ground-truth targets —
      the value is the same as the RGB hole path with a ``.target``
      suffix (B/C's training will produce the actual file).

    Parameters
    ----------
    path : str or Path
        Path to Hami's ``benchmarks/coco32_grt/manifest.json`` (or
        any compatible manifest).
    """
    p = Path(path)
    if not p.exists():
        raise ConditionContractError(f"Hami manifest not found: {p}")
    data = json.loads(p.read_text(encoding="utf-8"))
    images = data.get("images", [])
    cases = data.get("cases", [])

    # Build a lookup keyed by coco_image_id -> depth_stats.
    p99_by_coco_id: dict[int, float] = {}
    image_path_by_coco_id: dict[int, str] = {}
    split_by_coco_id: dict[int, Split] = {}
    for entry in images:
        try:
            coco_id = int(entry["coco_image_id"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ConditionContractError(
                f"Hami image entry missing/invalid coco_image_id: {entry!r}"
            ) from exc
        stats = entry.get("depth_stats") or {}
        p99_val = float(stats.get("p99", 0.0)) if isinstance(stats, dict) else 0.0
        p99_by_coco_id[coco_id] = p99_val
        image_path = entry.get("file_name") or entry.get("image_path") or ""
        if image_path:
            image_path_by_coco_id[coco_id] = str(image_path)
        try:
            split_by_coco_id[coco_id] = _HAMI_SPLIT_TO_CONTRACT[str(entry.get("split", "train"))]
        except (KeyError, ValueError) as exc:
            raise ConditionContractError(
                f"unknown Hami image split {entry.get('split')!r}"
            ) from exc

    if not cases:
        # Fall back: if the manifest only has the "images" list and
        # no cases, we still build a SampleManifest per image so the
        # caller can at least know the sources.
        out: List[SampleManifest] = []
        for entry in images:
            coco_id = int(entry["coco_image_id"])
            depth_norm = DepthNormalization(p99=p99_by_coco_id.get(coco_id, 0.0))
            out.append(
                SampleManifest(
                    case_id=str(entry.get("file_name", coco_id)),
                    source_id=str(coco_id),
                    split=split_by_coco_id.get(coco_id, Split.TRAIN),
                    coord_frame=CoordFrame.SOURCE,
                    rgb_path=str(entry.get("file_name") or ""),
                    depth_path=f"disparity/{coco_id}.pt",
                    mask_path="",
                    target_path=f"{entry.get('file_name', coco_id)}.target.png",
                    direction=Direction.R2L,
                    dmax_px=0,
                    depth_normalization=depth_norm,
                )
            )
        return out

    out = []
    for case in cases:
        try:
            coco_id = int(case["coco_image_id"])
            dmax_px = int(case["dmax_px"])
            direction = _HAMI_DIR_TO_CONTRACT[str(case["direction"])]
            split = _HAMI_SPLIT_TO_CONTRACT[str(case["split"])]
        except (KeyError, TypeError, ValueError) as exc:
            raise ConditionContractError(
                f"Hami case entry missing/invalid required fields: {case!r}"
            ) from exc
        case_id = str(case["case_id"])
        image_path = str(case.get("image_path", image_path_by_coco_id.get(coco_id, "")))
        mask_path = str(case.get("mask_path", ""))
        if not image_path:
            raise ConditionContractError(
                f"Hami case {case_id} has no image_path and no fallback in images[]"
            )
        # The Hami manifest writes image_path as the
        # ``<split>/images/<case_id>.jpg`` materialised file. The
        # case_id is the source stem with the combo suffix, so the
        # *source* rgb is the bare stem; we use that for rgb_path.
        stem = case_id.split("__", 1)[0]
        rgb_rel = image_path  # already a data-relative POSIX path
        target_rel = str(Path(image_path).with_suffix(".target.png"))
        out.append(
            SampleManifest(
                case_id=case_id,
                source_id=str(coco_id),
                split=split,
                coord_frame=CoordFrame.SOURCE,
                rgb_path=rgb_rel,
                depth_path=f"disparity/{coco_id}.pt",
                mask_path=mask_rel_to_data_root(mask_path, stem) if mask_path else "",
                target_path=target_rel,
                direction=direction,
                dmax_px=dmax_px,
                depth_normalization=DepthNormalization(
                    p99=p99_by_coco_id.get(coco_id, 0.0)
                ),
            )
        )
    return out


def mask_rel_to_data_root(mask_path: str, stem: str) -> str:
    """If ``mask_path`` is absolute, rebase it to a data-relative path.

    Hami writes absolute paths to the working tree in some
    configurations; the contract demands portable, data-relative
    paths. When the input is already relative, return as-is.
    """
    p = Path(mask_path)
    if p.is_absolute():
        # Try to find a sensible "data/..." form by looking for
        # common markers. We pick the last ``masks/`` ancestor to
        # be defensive.
        for parent in p.parents:
            if parent.name == "masks" or parent.name == "data":
                rel = p.relative_to(parent.parent if parent.name == "masks" else parent)
                return rel.as_posix()
        return p.name
    return p.as_posix()
