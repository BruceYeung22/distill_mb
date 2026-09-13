"""Data subpackage — manifest, GRT, datasets, pipeline.

Owned by Agent A.
"""

from .grt import (
    apply_p99_normalization,
    compute_grt_mask,
    disparity_to_signed_normalized,
)
from .depth_normalization import (
    DepthNormalizationApplier,
    apply_depth_normalization,
)
from .zipdepth_adapter import (
    ZipDepthAdapter,
    disparity_path_for,
    load_disparity,
)
from .manifest_io import (
    from_hami_320,
    load_manifest,
    mask_rel_to_data_root,
    save_manifest,
    to_relative,
)
from .splits import (
    SPLIT_SOURCES,
    assert_isolated,
    build_default_splits,
    split_sources_from_directory,
)
from .synthetic import make_synthetic_batch
from .pipeline_512 import build_512_sample, prepare_512_sample
from .grt_dataset import (
    CaseSpec,
    GrtTrainProvider,
    ZipDepthOnline,
    build_case,
    build_manifest,
    hole_ratio_of,
    list_image_ids,
    load_manifest as load_online_manifest,
    resolve_zipdepth_dir,
    save_manifest as save_online_manifest,
)


__all__ = [
    # grt
    "compute_grt_mask",
    "apply_p99_normalization",
    "disparity_to_signed_normalized",
    # depth_normalization
    "DepthNormalizationApplier",
    "apply_depth_normalization",
    # zipdepth adapter
    "ZipDepthAdapter",
    "load_disparity",
    "disparity_path_for",
    # manifest_io
    "load_manifest",
    "save_manifest",
    "load_online_manifest",
    "save_online_manifest",
    "from_hami_320",
    "to_relative",
    "mask_rel_to_data_root",
    # splits
    "SPLIT_SOURCES",
    "assert_isolated",
    "build_default_splits",
    "split_sources_from_directory",
    # synthetic
    "make_synthetic_batch",
    # pipeline
    "build_512_sample",
    "prepare_512_sample",
    # online ZipDepth + GRT dataset (TDD2 §3)
    "CaseSpec",
    "ZipDepthOnline",
    "build_case",
    "build_manifest",
    "GrtTrainProvider",
    "hole_ratio_of",
    "list_image_ids",
    "resolve_zipdepth_dir",
]
