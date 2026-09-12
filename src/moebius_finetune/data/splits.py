"""Three-split source-id definitions + isolation checks.

The three splits follow Hami's convention (TDD §4.2):

* ``train`` — 32 source images from the 320 base set. Both the
  *image* and the *mask* are unseen by ``holdout_masks``/
  ``holdout_images``; here, the masks are seen.
* ``holdout_masks`` — same 32 sources as ``train`` but with
  unseen disparity scales. So the *source* IDs overlap with
  ``train`` (this is the "same-image different-mask" capacity
  benchmark).
* ``holdout_images`` — 32 *new* sources that do not appear in
  ``train`` at all.

Isolation invariant:

* ``holdout_images ∩ train == ∅``  (source-image ID disjoint)
* ``holdout_masks ⊆ train``        (sources are a subset)
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Mapping, Set, Union

from ..contracts import ConditionContractError, Split


__all__ = [
    "SPLIT_SOURCES",
    "assert_isolated",
    "build_default_splits",
    "split_sources_from_directory",
]


#: Set of COCO source IDs for each split. Populated lazily by
#: :func:`build_default_splits` from Hami's benchmark directories.
#: When the Hami data is unavailable (e.g. on a CI box without
#: ``D:/project/Hami``) the mapping is empty and downstream code
#: must fall back to its own data source.
SPLIT_SOURCES: Dict[Split, Set[str]] = {
    Split.TRAIN: set(),
    Split.HOLDOUT_MASKS: set(),
    Split.HOLDOUT_IMAGES: set(),
}


def split_sources_from_directory(
    image_dir: Union[str, Path],
    *,
    suffix: str = ".jpg",
) -> Set[str]:
    """Return the set of source stems (file names without extension)
    in ``image_dir``.

    The directory is read non-recursively; only files ending in
    ``suffix`` (default ``.jpg``) are considered.
    """
    d = Path(image_dir)
    if not d.exists():
        return set()
    return {p.stem for p in d.glob(f"*{suffix}") if p.is_file()}


def build_default_splits(
    hami_benchmark_root: Union[str, Path] = Path(r"D:/project/Hami/benchmarks/coco32_grt"),
) -> Dict[Split, Set[str]]:
    """Populate :data:`SPLIT_SOURCES` from the default Hami layout.

    The Hami benchmark materialises per-case files in
    ``<split>/images/<case_id>.jpg``. The source stem is the
    portion before the ``__`` separator (e.g.
    ``000000033114__00_d08L`` → ``000000033114``). We only
    consider files of the form ``<stem>__<idx>_<combo>.jpg`` so
    that the per-case image copies do not pollute the source-id
    sets.
    """
    root = Path(hami_benchmark_root)
    out: Dict[Split, Set[str]] = {
        Split.TRAIN: set(),
        Split.HOLDOUT_MASKS: set(),
        Split.HOLDOUT_IMAGES: set(),
    }
    for split in (Split.TRAIN, Split.HOLDOUT_MASKS, Split.HOLDOUT_IMAGES):
        image_dir = root / split.value / "images"
        stems = split_sources_from_directory(image_dir)
        # The Hami benchmark materialises the SAME source image as
        # multiple per-case files. Take only those whose filename
        # looks like ``<stem>__<idx>_<combo>.jpg`` and extract the
        # source stem.
        for stem in stems:
            # We can also see the bare ``<coco_id>.jpg`` if the user
            # has placed the source images alongside. Only add when
            # we have a proper combo file.
            pass
        # Take the per-case stems and reduce to unique sources.
        sources: Set[str] = set()
        for p in image_dir.glob("*.jpg"):
            if "__" in p.stem:
                src = p.stem.split("__", 1)[0]
                sources.add(src)
        out[split] = sources
    SPLIT_SOURCES.update(out)
    return out


def assert_isolated(splits: Mapping[Split, Set[str]]) -> None:
    """Validate the three-split isolation invariant.

    Raises :class:`ConditionContractError` on any violation. The
    accepted rules are:

    * ``holdout_images ∩ train == ∅``
    * ``holdout_masks ⊆ train`` (sources)
    """
    train = set(splits.get(Split.TRAIN, set()))
    holdout_masks = set(splits.get(Split.HOLDOUT_MASKS, set()))
    holdout_images = set(splits.get(Split.HOLDOUT_IMAGES, set()))

    overlap = holdout_images & train
    if overlap:
        raise ConditionContractError(
            f"holdout_images and train share {len(overlap)} source id(s); "
            f"first 5: {sorted(overlap)[:5]}"
        )
    leak = holdout_masks - train
    if leak:
        raise ConditionContractError(
            f"holdout_masks has {len(leak)} source id(s) not in train; "
            f"first 5: {sorted(leak)[:5]}"
        )
