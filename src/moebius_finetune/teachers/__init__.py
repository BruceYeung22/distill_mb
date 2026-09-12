"""Teachers subpackage (Agent B).

Public surface:

* :class:`DepthConditionAdapter` — zero-init 2→hidden→target_dim adapter.
* :class:`DepthConditionedRemoval` — single-condition UNet wrapper.
* :class:`OriginalRemovalBaseline` — original 9-channel teacher, no
  adapter, for ablation alignment.
* :func:`load_removal_model` / :func:`get_weight_metadata` /
  :func:`verify_state_dict_hash` — strict weight loading.
* :data:`MOEBIUS_PINNED_COMMIT` / :data:`MOEBIUS_CONFIG_REL` — pinned
  upstream references.

The teachers package depends on the Moebius upstream (read-only) at
runtime. CPU-only tests can import this subpackage without the
upstream: the upstream import is deferred inside
:func:`load_removal_model` and friends.
"""

from __future__ import annotations

from .depth_adapter import DepthAdapterConfigError, DepthConditionAdapter
from .loader import (
    DEFAULT_NUM_EMBEDDINGS,
    MOEBIUS_CONFIG_REL,
    MOEBIUS_PINNED_COMMIT,
    WeightHashMismatchError,
    WeightLoadError,
    get_weight_metadata,
    load_removal_model,
    sha256_of_file,
    verify_state_dict_hash,
)
from .original_baseline import OriginalRemovalBaseline
from .wrapper import (
    DEFAULT_INPUT_IDS_HALF,
    DepthConditionedRemoval,
    PredictCandidateError,
    WrapperConfigError,
)

__all__ = [
    "DEFAULT_INPUT_IDS_HALF",
    "DEFAULT_NUM_EMBEDDINGS",
    "DepthAdapterConfigError",
    "DepthConditionAdapter",
    "DepthConditionedRemoval",
    "MOEBIUS_CONFIG_REL",
    "MOEBIUS_PINNED_COMMIT",
    "OriginalRemovalBaseline",
    "PredictCandidateError",
    "WeightHashMismatchError",
    "WeightLoadError",
    "WrapperConfigError",
    "get_weight_metadata",
    "load_removal_model",
    "sha256_of_file",
    "verify_state_dict_hash",
]
