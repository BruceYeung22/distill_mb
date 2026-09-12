"""Single-condition forward wrapper around the Moebius teacher.

The depth-conditioned teacher is a :class:`torch.nn.Module` that owns:

* a Moebius :class:`RemovalModel` (loaded strictly, see
  :mod:`moebius_finetune.teachers.loader`),
* a :class:`DepthConditionAdapter` (zero-initialized, see
  :mod:`moebius_finetune.teachers.depth_adapter`).

The wrapper **adds** the depth-adapter residual to the output of the
original ``conv_in``. We do this with a forward hook that is registered
and removed inside a single call, so no state ever leaks across calls
(TDD §5.1: "禁止可变全局状态、禁止跨调用残留的 forward hook").

The single-condition forward is a strict subset of the original
pipeline:

    latent_model_input = cat([noisy_latent, mask, masked_latent], 1)   # 9ch
    noise_pred = diff_model(latent_model_input, t, input_ids[0..9])   # no CFG doubling

The wrapper exposes :func:`predict_candidate` which takes the public
:class:`ConditionBatch` plus the explicit initial noise and returns a
single denoising step's noise prediction. The full DDIM loop lives in
:mod:`moebius_finetune.training.teacher.cache` and is run by callers
that need it; the wrapper itself is one UNet step.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator, Optional, Union

import numpy as np
import torch
from torch import nn

from ..contracts import ConditionBatch, validate_condition
from .depth_adapter import DepthConditionAdapter


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class WrapperConfigError(ValueError):
    """Raised when the wrapper is constructed with invalid arguments."""


class PredictCandidateError(RuntimeError):
    """Raised when :func:`predict_candidate` cannot complete a forward."""


# ---------------------------------------------------------------------------
# DepthConditionedRemoval
# ---------------------------------------------------------------------------


# Default input_ids: IDs 0..9 are the "conditional" half of the
# original 20-token embedding table (TDD §5.1). We use these for the
# single-condition (no CFG) path.
DEFAULT_INPUT_IDS_HALF = 10


class DepthConditionedRemoval(nn.Module):
    """Single-condition forward wrapper that adds a depth-adapter residual.

    Parameters
    ----------
    model
        A built Moebius :class:`RemovalModel`. We require the model to
        have a ``diff_model.conv_in`` attribute (this is the case for
        the pinned config). The model is held by reference; do not
        mutate its parameters from outside the wrapper.
    depth_adapter
        A :class:`DepthConditionAdapter` instance. The wrapper exposes
        ``enable_depth_branch`` / ``disable_depth_branch`` context
        managers that toggle ``depth_adapter.enabled`` for the
        duration of the ``with`` block.
    input_ids_half
        Number of conditional input IDs to use (TDD §5.1 default: 10,
        giving the IDs ``0..9``). Set to the full
        ``model.num_embeddings`` if you want unconditional IDs.
    """

    def __init__(
        self,
        model: nn.Module,
        depth_adapter: DepthConditionAdapter,
        *,
        input_ids_half: int = DEFAULT_INPUT_IDS_HALF,
    ) -> None:
        super().__init__()
        if depth_adapter.target_dim != _conv_in_out_channels(model):
            raise WrapperConfigError(
                f"depth_adapter.target_dim={depth_adapter.target_dim} "
                f"does not match conv_in out channels "
                f"{_conv_in_out_channels(model)}; the adapter residual "
                f"cannot be added after conv_in."
            )
        if input_ids_half <= 0 or input_ids_half > model.num_embeddings:
            raise WrapperConfigError(
                f"input_ids_half must be in (0, num_embeddings="
                f"{model.num_embeddings}], got {input_ids_half}"
            )

        # Submodule assignment so the optimizer sees both sets of
        # parameters.
        self.model = model
        self.depth_adapter = depth_adapter
        self.input_ids_half = int(input_ids_half)
        # Whether the depth residual is enabled. The context managers
        # below mutate this flag in a stack-like way.
        self._branch_depth_enabled: bool = True
        self._depth_features_holder: list = []  # for the current call only

    # ------------------------------------------------------------------
    # Context managers
    # ------------------------------------------------------------------

    @contextmanager
    def disable_depth_branch(self) -> Iterator[None]:
        """Disable the depth residual for the duration of the block."""
        prev = self._branch_depth_enabled
        self._branch_depth_enabled = False
        prev_adapter = self.depth_adapter.enabled
        self.depth_adapter.enabled = False
        try:
            yield
        finally:
            self._branch_depth_enabled = prev
            self.depth_adapter.enabled = prev_adapter

    @contextmanager
    def enable_depth_branch(self) -> Iterator[None]:
        """Re-enable the depth residual for the duration of the block."""
        prev = self._branch_depth_enabled
        self._branch_depth_enabled = True
        prev_adapter = self.depth_adapter.enabled
        self.depth_adapter.enabled = True
        try:
            yield
        finally:
            self._branch_depth_enabled = prev
            self.depth_adapter.enabled = prev_adapter

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def _make_input_ids(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.arange(
            0, self.input_ids_half, dtype=torch.int64, device=device
        ).unsqueeze(0).expand(batch_size, -1).contiguous()

    def _conv_in_hook(self, module, inputs, output):
        """Add the depth-adapter residual to the conv_in output.

        The holder is a per-call list: the wrapper pushes the features
        in :func:`forward` and removes them after the call returns. We
        do not store any cross-call state here, satisfying TDD §5.1.
        """
        if not self._branch_depth_enabled:
            return output
        if not self._depth_features_holder:
            return output
        depth_features = self._depth_features_holder[-1]
        if depth_features is None:
            return output
        residual = self.depth_adapter(depth_features)
        if residual.shape != output.shape:
            raise WrapperConfigError(
                f"depth adapter residual shape {tuple(residual.shape)} "
                f"does not match conv_in output shape "
                f"{tuple(output.shape)}"
            )
        return output + residual

    def forward(
        self,
        noisy_latents: torch.Tensor,
        timesteps: torch.Tensor,
        input_ids: Optional[torch.Tensor],
        *,
        depth_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Run one single-condition denoising step.

        Parameters
        ----------
        noisy_latents
            ``[B, 4, H/8, W/8]`` noisy latent tensor.
        timesteps
            ``[B]`` or ``[1]`` int64 timestep tensor.
        input_ids
            ``[B, K]`` int64 input IDs. If ``None``, defaults to the
            conditional half (IDs 0..9) of the model's embedding table.
        depth_features
            ``[B, 2, H/8, W/8]`` low-resolution depth features
            (depth-mean + coverage from contracts §4.3). When ``None``,
            the depth branch is bypassed for this call.
        """
        B = noisy_latents.shape[0]
        if input_ids is None:
            input_ids = self._make_input_ids(B, noisy_latents.device)

        # Per-call holder: register, run, clear. No state survives
        # across calls.
        self._depth_features_holder.append(depth_features)
        conv_in = self.model.diff_model.conv_in
        handle = conv_in.register_forward_hook(self._conv_in_hook)
        try:
            noise_pred = self.model(noisy_latents, timesteps, input_ids)
        finally:
            handle.remove()
            self._depth_features_holder.pop()
        # The Moebius RemovalModel returns noise_pred directly (not a
        # UNet2DOutput); diffusers' UNet2DConditionModel wraps it in
        # ``.sample``. We normalize so the wrapper always returns a
        # plain tensor.
        if hasattr(noise_pred, "sample"):
            return noise_pred.sample
        return noise_pred

    # ------------------------------------------------------------------
    # Public inference helper
    # ------------------------------------------------------------------

    def predict_candidate(
        self,
        condition: ConditionBatch,
        noise: torch.Tensor,
        *,
        timesteps: Optional[torch.Tensor] = None,
        input_ids: Optional[torch.Tensor] = None,
        masked_latent: Optional[torch.Tensor] = None,
        depth_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """One single-condition UNet step from a public :class:`ConditionBatch`.

        The function:

        1. Validates ``condition`` against the §4.1 contract.
        2. Builds the 9-channel UNet input by concatenating
           ``noisy_latent`` (= ``noise``), the resized mask and the
           VAE-encoded masked image.
        3. Runs :func:`forward` with the depth-adapter residual.

        ``masked_latent`` may be passed directly to skip the VAE encode
        step. ``depth_features`` is the 2-channel H/8 features from
        contracts §4.3. Both are optional; the function never reads the
        "target" path (TDD §5.2).
        """
        validate_condition(condition)
        # Force the contract into torch float32; the user is responsible
        # for the VAE precision but we keep the inputs in fp32 here.
        rgb_hole = torch.from_numpy(np.ascontiguousarray(condition.rgb_hole))
        hole_mask = torch.from_numpy(np.ascontiguousarray(condition.hole_mask))
        depth_hole = torch.from_numpy(np.ascontiguousarray(condition.depth_hole))
        if not isinstance(noise, torch.Tensor):
            noise = torch.from_numpy(np.ascontiguousarray(noise))

        device = noise.device
        rgb_hole = rgb_hole.to(device)
        hole_mask = hole_mask.to(device)
        depth_hole = depth_hole.to(device)
        noise = noise.to(device)

        # Resize mask from H,W to H/8, W/8.
        B, _, H, W = hole_mask.shape
        latent_h, latent_w = H // 8, W // 8
        latent_mask = nn.functional.interpolate(
            hole_mask, size=(latent_h, latent_w), mode="nearest"
        )

        # Build the 9-channel UNet input.
        if masked_latent is None:
            raise PredictCandidateError(
                "predict_candidate requires either a pre-computed "
                "masked_latent or a VAE to encode the masked image. "
                "Pass masked_latent=... at call time, or use the "
                "HigherLevelTeacherPredictor which holds a VAE."
            )
        masked_latent = masked_latent.to(device)
        latent_input = torch.cat([noise, latent_mask, masked_latent], dim=1)

        # Depth features: 2-channel H/8 features (coverage + depth-mean).
        if depth_features is None:
            depth_features = _default_depth_features(hole_mask, depth_hole)
        depth_features = depth_features.to(device)

        if timesteps is None:
            timesteps = torch.zeros((B,), dtype=torch.int64, device=device)
        timesteps = timesteps.to(device)

        return self.forward(
            latent_input,
            timesteps,
            input_ids,
            depth_features=depth_features,
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _conv_in_out_channels(model: nn.Module) -> int:
    """Return the output channel count of ``model.diff_model.conv_in``."""
    diff = getattr(model, "diff_model", None)
    if diff is None:
        raise WrapperConfigError("model has no diff_model attribute")
    conv_in = getattr(diff, "conv_in", None)
    if conv_in is None:
        raise WrapperConfigError("model.diff_model has no conv_in attribute")
    # DepthwiseSeparableConv exposes conv_pw (the 1x1 projection).
    pw = getattr(conv_in, "conv_pw", None)
    if pw is not None and hasattr(pw, "out_channels"):
        return int(pw.out_channels)
    # Fallback: take the second dimension of conv_pw.weight if present.
    if pw is not None and hasattr(pw, "weight"):
        return int(pw.weight.shape[0])
    raise WrapperConfigError(
        f"Cannot determine conv_in out channels from {type(conv_in).__name__}"
    )


def _default_depth_features(
    hole_mask: torch.Tensor, depth_hole: torch.Tensor
) -> torch.Tensor:
    """Compute a default 2-channel H/8 depth feature from contracts §4.3.

    This is the area-downsampled **depth-mean followed by coverage**,
    i.e. the returned channel order is ``[depth_mean(1), coverage(1)]``.
    We expose it here so callers can fall back to a deterministic
    representation when the depth adapter input is not pre-computed
    (e.g. for tiny smoke tests that do not have a real ZipDepth cache
    available).
    """
    K = 1.0 - hole_mask
    coverage = nn.functional.avg_pool2d(K, kernel_size=8)
    depth_sum = nn.functional.avg_pool2d(depth_hole, kernel_size=8)
    eps = 1e-6
    depth_mean = depth_sum / torch.clamp(coverage, min=eps)
    # Where coverage == 0, depth_mean must be 0 (TDD §4.3).
    depth_mean = torch.where(coverage > 0, depth_mean, torch.zeros_like(depth_mean))
    return torch.cat([depth_mean, coverage], dim=1)


__all__ = [
    "DEFAULT_INPUT_IDS_HALF",
    "DepthConditionedRemoval",
    "PredictCandidateError",
    "WrapperConfigError",
    "_default_depth_features",
]
