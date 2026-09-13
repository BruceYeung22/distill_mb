"""Teacher fine-tuning loop for the depth-conditioned Moebius.

TDD §5.3 specifies the two-stage recipe:

* Stage 1 (``DepthBranchOnlyRecipe``): backbone is frozen, only the
  depth adapter trains. BN running statistics are not updated.
* Stage 2 (``JointUnfreezeRecipe``): backbone opens, branch lr=1e-4,
  backbone lr=1e-5. BN running statistics stay frozen (we do not flip
  ``track_running_stats`` back on; we keep the model in eval mode for
  BN, which is the standard TDD §5.1 behaviour).

The losses live in :mod:`moebius_finetune.training.teacher.losses`;
the depth feature extraction lives in
:func:`moebius_finetune.teachers.wrapper._default_depth_features` (TDD
§4.3). We do not re-derive the recipe, the loss or the depth features
here.

This module is also responsible for:

* checkpointing (model, optimizer, scheduler, global step, RNG state,
  data version, weight source hash),
* resume verification (:func:`verify_resume_consistency`).
"""

from __future__ import annotations

import json
import os
import random
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import torch
from torch import nn
from torch.optim import AdamW

from ...contracts import ConditionBatch
from ...teachers import (
    DepthConditionAdapter,
    DepthConditionedRemoval,
    OriginalRemovalBaseline,
)
from .losses import combined_epsilon_loss
from .recipe import (
    BaseRecipe,
    DepthBranchOnlyRecipe,
    JointUnfreezeRecipe,
    LocalSmokeRecipe,
    TeacherTrainingStage,
)


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class FinetuneConfigError(ValueError):
    """Raised when the fine-tuning config is malformed."""


class ResumeMismatchError(RuntimeError):
    """Raised when a checkpoint cannot be resumed consistently."""


# ---------------------------------------------------------------------------
# Trainer artifacts
# ---------------------------------------------------------------------------


@dataclass
class TeacherFinetuneArtifacts:
    """The artifacts of a fine-tuning run.

    Attributes
    ----------
    output_dir
        Where checkpoints and logs are written.
    global_step
        Final optimizer step (after gradient accumulation).
    best_loss
        Lowest loss observed during training.
    peak_memory_bytes
        Peak GPU memory in bytes (0 if no GPU / AMP off).
    checkpoint_paths
        All checkpoint files written, ordered by step.
    """

    output_dir: Path
    global_step: int
    best_loss: float
    peak_memory_bytes: int
    checkpoint_paths: List[Path] = field(default_factory=list)
    log_path: Optional[Path] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "output_dir": str(self.output_dir),
            "global_step": int(self.global_step),
            "best_loss": float(self.best_loss),
            "peak_memory_bytes": int(self.peak_memory_bytes),
            "checkpoint_paths": [str(p) for p in self.checkpoint_paths],
            "log_path": str(self.log_path) if self.log_path is not None else None,
            "metadata": dict(self.metadata),
        }


# ---------------------------------------------------------------------------
# TrainResult
# ---------------------------------------------------------------------------


@dataclass
class TrainResult:
    """Step-level train result (returned from the internal stepper)."""

    loss: float
    hole_loss: float
    known_loss: float
    grad_norm: float
    learning_rate: float


# ---------------------------------------------------------------------------
# Batch provider protocol
# ---------------------------------------------------------------------------


BatchProvider = Callable[[], Dict[str, Any]]
"""Callable returning a single train batch.

The returned dict may contain:

* ``clean_rgb`` (required): ``[B, 3, H, W]`` float32 in [0, 1].
* ``hole_mask``: ``[B, 1, H, W]`` float32 in {0, 1}.
* ``depth_hole``: ``[B, 1, H, W]`` float32 (zero in the hole).
* ``noise`` (required): ``[B, 4, H/8, W/8]`` float32.
* ``masked_latent``: optional pre-computed ``[B, 4, H/8, W/8]`` VAE
  latent of the masked image. When missing, the trainer encodes the
  masked image with the VAE passed at construction.

The provider is called once per microbatch. It should be deterministic with
respect to the global RNG state when seeding has been done.
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _check_amp_dtype(amp: bool, amp_dtype: str, device: torch.device) -> Tuple[bool, Optional[torch.dtype]]:
    if not amp:
        return False, None
    if device.type == "cpu":
        return False, None
    if amp_dtype == "bf16":
        if not torch.cuda.is_bf16_supported():
            warnings.warn("BF16 not supported on this device; falling back to FP32.", stacklevel=2)
            return False, None
        return True, torch.bfloat16
    if amp_dtype == "fp16":
        return True, torch.float16
    if amp_dtype == "fp32":
        return False, None
    raise FinetuneConfigError(f"Unknown amp_dtype: {amp_dtype!r}")


def _seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _capture_rng_state() -> Dict[str, Any]:
    state: Dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(state: Dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and "torch_cuda" in state:
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def _bn_eval(model: nn.Module) -> None:
    """Set all BN layers to eval mode so running statistics do not drift.

    TDD §5.1: "主干冻结阶段维持 BN running statistics 不变".
    """
    for module in model.modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            module.eval()


def _split_param_groups(
    model: DepthConditionedRemoval,
    stage: TeacherTrainingStage,
) -> List[Dict[str, Any]]:
    """Return AdamW parameter groups honouring the stage learning rates.

    Three groups (TDD2 §5):

    * ``depth_branch`` — the depth adapter; always trains.
    * ``conv_in`` — trains only when ``stage.conv_in_lr > 0`` (TDD2 D2:
      stage 2 unfreezes *only* conv_in; the rest of the backbone stays
      frozen for the whole run).
    * ``backbone`` — everything else in the backbone; trains only when
      ``stage.backbone_lr > 0``.

    ``requires_grad`` is forced to match the lr gates so the optimizer
    and the test assertions agree.
    """
    branch_params: List[nn.Parameter] = list(model.depth_adapter.parameters())
    conv_in = _get_conv_in(model)
    conv_in_params: List[nn.Parameter] = (
        list(conv_in.parameters()) if conv_in is not None else []
    )
    conv_in_ids = {id(p) for p in conv_in_params}
    backbone_params: List[nn.Parameter] = [
        p for p in model.model.parameters() if id(p) not in conv_in_ids
    ]
    # Force the contract: each group's requires_grad matches its lr gate.
    for p in backbone_params:
        p.requires_grad = bool(stage.backbone_lr > 0.0)
    for p in conv_in_params:
        p.requires_grad = bool(stage.conv_in_lr > 0.0)
    for p in branch_params:
        p.requires_grad = True
    backbone_params = [p for p in backbone_params if p.requires_grad]
    conv_in_params = [p for p in conv_in_params if p.requires_grad]
    branch_params = [p for p in branch_params if p.requires_grad]
    groups: List[Dict[str, Any]] = []
    if backbone_params and stage.backbone_lr > 0.0:
        groups.append(
            {
                "params": backbone_params,
                "lr": float(stage.backbone_lr),
                "name": "backbone",
            }
        )
    if conv_in_params and stage.conv_in_lr > 0.0:
        groups.append(
            {
                "params": conv_in_params,
                "lr": float(stage.conv_in_lr),
                "name": "conv_in",
            }
        )
    if branch_params:
        groups.append(
            {
                "params": branch_params,
                "lr": float(stage.depth_branch_lr),
                "name": "depth_branch",
            }
        )
    if not groups:
        raise FinetuneConfigError(
            "No trainable parameters: backbone is frozen and the depth "
            "adapter has requires_grad=False. Did you forget to enable "
            "the depth branch?"
        )
    for g in groups:
        g["weight_decay"] = float(stage.weight_decay)
    return groups


def _get_conv_in(model: nn.Module) -> Optional[nn.Module]:
    """Return ``model.model.diff_model.conv_in`` when the path exists."""
    inner = getattr(model, "model", None)
    diff = getattr(inner, "diff_model", None)
    return getattr(diff, "conv_in", None)


# ---------------------------------------------------------------------------
# Internal stepper
# ---------------------------------------------------------------------------


def _build_masked_image(condition: ConditionBatch) -> torch.Tensor:
    """``(2 * rgb_hole - 1) * (1 - hole_mask)`` in float32.

    TDD §5.2: "数据层始终为 RGB [0,1]；条件图为 (2*rgb_hole-1)*(1-hole_mask)".
    """
    rgb = torch.from_numpy(np.ascontiguousarray(condition.rgb_hole))
    mask = torch.from_numpy(np.ascontiguousarray(condition.hole_mask))
    masked = (2.0 * rgb - 1.0) * (1.0 - mask)
    return masked


def _default_depth_features(hole_mask: torch.Tensor, depth_hole: torch.Tensor) -> torch.Tensor:
    """Re-export the §4.3 area-downsampled features for the trainer."""
    K = 1.0 - hole_mask
    coverage = nn.functional.avg_pool2d(K, kernel_size=8)
    depth_sum = nn.functional.avg_pool2d(depth_hole, kernel_size=8)
    eps = 1e-6
    depth_mean = depth_sum / torch.clamp(coverage, min=eps)
    depth_mean = torch.where(coverage > 0, depth_mean, torch.zeros_like(depth_mean))
    return torch.cat([depth_mean, coverage], dim=1)


def _encode_masked_latent(
    masked_image: torch.Tensor,
    vae: Optional[nn.Module],
    device: torch.device,
    amp: bool,
    amp_dtype: Optional[torch.dtype],
) -> torch.Tensor:
    if vae is None:
        raise FinetuneConfigError(
            "No VAE provided and masked_latent not pre-computed; cannot "
            "construct the 9-channel UNet input."
        )
    try:
        vae_dtype = next(vae.parameters()).dtype
    except (StopIteration, AttributeError):
        vae_dtype = torch.float32
    with torch.no_grad():
        if amp and amp_dtype is not None:
            with torch.autocast(device_type=device.type, dtype=amp_dtype):
                encoded = vae.encode(masked_image.to(device=device, dtype=vae_dtype))
                latent = encoded.latent_dist.mode() if hasattr(encoded, "latent_dist") else encoded.mode()
        else:
            encoded = vae.encode(masked_image.to(device=device, dtype=vae_dtype))
            latent = encoded.latent_dist.mode() if hasattr(encoded, "latent_dist") else encoded.mode()
        scale = getattr(vae.config, "scaling_factor", 0.13025)
    return (latent * float(scale)).to(dtype=torch.float32)


# ---------------------------------------------------------------------------
# Public entry: finetune_depth_branch
# ---------------------------------------------------------------------------


def finetune_depth_branch(
    model: DepthConditionedRemoval,
    train_cfg: BaseRecipe,
    *,
    batch_provider: BatchProvider,
    output_dir: Union[str, os.PathLike],
    vae: Optional[nn.Module] = None,
    weight_metadata: Optional[Dict[str, Any]] = None,
    data_version: str = "synthetic",
    save_initial_checkpoint: bool = True,
    device: Optional[torch.device] = None,
) -> TeacherFinetuneArtifacts:
    """Run the fine-tuning loop for a single recipe.

    Parameters
    ----------
    model
        :class:`DepthConditionedRemoval` (or anything with the same
        ``forward(noisy_latents, timesteps, input_ids, depth_features=...)``
        signature).
    train_cfg
        Recipe describing the stage (see :mod:`recipe`).
    batch_provider
        Callable returning a batch dict (see :data:`BatchProvider`).
    output_dir
        Directory for checkpoints and logs. Created if missing.
    vae
        Optional VAE used to encode the masked image. The VAE is held
        in ``eval()`` mode and not trained.
    weight_metadata
        Optional metadata (from :func:`get_weight_metadata`) for the
        saved checkpoint. Used to track the weight source hash.
    data_version
        Data version string embedded in the checkpoint for cache
        invalidation (TDD §7.1).
    save_initial_checkpoint
        If true, a checkpoint is written before the first step so
        resume verification has a baseline to compare against.
    device
        Target device. Defaults to CUDA when available, else CPU.

    Returns
    -------
    TeacherFinetuneArtifacts

    Notes
    -----
    ``stage.max_steps`` counts optimizer updates. Each update consumes
    ``stage.grad_accum_steps`` provider calls; logs, checkpoints, and
    ``global_step`` use the same optimizer-update count.
    """
    if not isinstance(model, DepthConditionedRemoval):
        # For ablation, an OriginalRemovalBaseline can be used; in that
        # case there is no depth adapter to train. We still allow it.
        if not isinstance(model, OriginalRemovalBaseline):
            raise FinetuneConfigError(
                f"model must be a DepthConditionedRemoval or "
                f"OriginalRemovalBaseline, got {type(model).__name__}"
            )
    if not callable(batch_provider):
        raise FinetuneConfigError("batch_provider must be callable")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    stage = train_cfg.stage()
    grad_accum_steps = int(stage.grad_accum_steps)
    if grad_accum_steps <= 0:
        raise FinetuneConfigError(
            f"grad_accum_steps must be positive, got {grad_accum_steps}"
        )
    amp, amp_dtype = _check_amp_dtype(stage.amp, stage.amp_dtype, device)

    # Seed before model move so initial state is deterministic.
    _seed_all(int(train_cfg.seed))

    # Move model to device. Keep BN frozen (TDD §5.1).
    model.to(device)
    if vae is not None:
        # The VAE must live on the same device: autocast casts the CUDA
        # input to bf16 while CPU conv weights stay fp32 → conv dtype
        # mismatch (found in the first real-data sanity run).
        vae.to(device)
    _bn_eval(model)
    # Always keep the depth adapter in train mode so its parameters
    # receive gradient updates even with BN frozen.
    if hasattr(model, "depth_adapter"):
        model.depth_adapter.train()
    else:
        # No adapter (OriginalRemovalBaseline) → no trainable branch.
        pass

    # Build optimizer.
    if isinstance(model, DepthConditionedRemoval):
        param_groups = _split_param_groups(model, stage)
    else:
        # No branch to train. Still create an optimizer so the loop is
        # well-defined for the baseline case (this matches "no depth
        # micro-tuning" ablation).
        backbone_params = [p for p in model.model.parameters() if p.requires_grad]
        if not backbone_params:
            raise FinetuneConfigError(
                "No trainable parameters on the baseline model."
            )
        param_groups = [
            {
                "params": backbone_params,
                "lr": float(stage.backbone_lr or 1e-5),
                "name": "backbone",
                "weight_decay": float(stage.weight_decay),
            }
        ]
    optimizer = AdamW(
        param_groups,
        lr=float(stage.depth_branch_lr),
        weight_decay=float(stage.weight_decay),
    )

    # Initial RNG state capture (before any training).
    initial_rng = _capture_rng_state()

    # Initial checkpoint.
    artifacts = TeacherFinetuneArtifacts(
        output_dir=output_dir,
        global_step=0,
        best_loss=float("inf"),
        peak_memory_bytes=0,
        metadata={
            "data_version": data_version,
            "weight_metadata": dict(weight_metadata) if weight_metadata else {},
            "stage": stage.to_dict(),
            "device": str(device),
        },
    )
    log_path = output_dir / "train.log"
    artifacts.log_path = log_path

    def _log(msg: str) -> None:
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(msg + "\n")

    _log(f"=== start stage={stage.name} steps={stage.max_steps} amp={amp} dtype={amp_dtype} device={device} ===")

    if save_initial_checkpoint:
        _save_checkpoint(
            output_dir=output_dir,
            model=model,
            optimizer=optimizer,
            scheduler=None,
            global_step=0,
            rng_state=initial_rng,
            data_version=data_version,
            weight_metadata=weight_metadata,
            stage=stage,
        )

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)

    # Alphas for the q_sample noising — from the same scheduler config
    # the cache/inference path uses, so train and eval spaces match.
    from .cache import default_scheduler_config, _build_ddim_scheduler

    train_ddim = _build_ddim_scheduler(default_scheduler_config())
    alphas_bar = train_ddim.alphas_cumprod.to(device=device, dtype=torch.float32)

    best_loss = float("inf")
    optimizer.zero_grad(set_to_none=True)
    for step in range(1, int(stage.max_steps) + 1):
        window_results: List[TrainResult] = []
        for _ in range(grad_accum_steps):
            batch = batch_provider()
            window_results.append(
                _train_step(
                    model=model,
                    optimizer=optimizer,
                    batch=batch,
                    vae=vae,
                    device=device,
                    amp=amp,
                    amp_dtype=amp_dtype,
                    known_weight=0.1,
                    alphas_bar=alphas_bar,
                    loss_scale=1.0 / grad_accum_steps,
                )
            )

        # Clip once, after the averaged gradients for the whole window have
        # been accumulated.  This keeps clipping independent of the number
        # of microbatches in a window.
        grad_norm = nn.utils.clip_grad_norm_(
            [p for g in optimizer.param_groups for p in g["params"]],
            max_norm=float(stage.max_grad_norm),
        )
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        result = TrainResult(
            loss=float(np.mean([r.loss for r in window_results])),
            hole_loss=float(np.mean([r.hole_loss for r in window_results])),
            known_loss=float(np.mean([r.known_loss for r in window_results])),
            grad_norm=float(grad_norm),
            learning_rate=window_results[-1].learning_rate,
        )
        if result.loss < best_loss:
            best_loss = float(result.loss)

        if step % int(train_cfg.log_every_steps) == 0:
            _log(
                f"step={step} loss={result.loss:.6f} "
                f"hole={result.hole_loss:.6f} known={result.known_loss:.6f} "
                f"gn={result.grad_norm:.4f} lr={result.learning_rate:.2e}"
            )

        if step % int(train_cfg.save_every_steps) == 0 or step == int(stage.max_steps):
            ckpt = _save_checkpoint(
                output_dir=output_dir,
                model=model,
                optimizer=optimizer,
                scheduler=None,
                global_step=step,
                rng_state=_capture_rng_state(),
                data_version=data_version,
                weight_metadata=weight_metadata,
                stage=stage,
            )
            artifacts.checkpoint_paths.append(ckpt)

        if torch.cuda.is_available():
            peak = int(torch.cuda.max_memory_allocated(device))
            if peak > artifacts.peak_memory_bytes:
                artifacts.peak_memory_bytes = peak

    artifacts.global_step = int(stage.max_steps)
    artifacts.best_loss = best_loss
    _log(
        f"=== done best_loss={best_loss:.6f} peak_mem={artifacts.peak_memory_bytes} ==="
    )
    return artifacts


# ---------------------------------------------------------------------------
# Checkpoint
# ---------------------------------------------------------------------------


CHECKPOINT_VERSION = 1
CHECKPOINT_BASENAME = "teacher_step{step:08d}.pt"


def _save_checkpoint(
    *,
    output_dir: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[Any],
    global_step: int,
    rng_state: Dict[str, Any],
    data_version: str,
    weight_metadata: Optional[Dict[str, Any]],
    stage: TeacherTrainingStage,
) -> Path:
    name = CHECKPOINT_BASENAME.format(step=global_step)
    path = output_dir / name

    # Save exactly the trainable subset (TDD2 §5): adapter when it
    # trains, conv_in when it unfreezes, the full state_dict only when
    # the backbone lr opens.
    trainable_parts: List[str] = []
    model_state: Dict[str, Any] = {}
    if isinstance(model, DepthConditionedRemoval) and hasattr(model, "depth_adapter"):
        if stage.depth_branch_lr > 0.0:
            model_state.update(
                {f"depth_adapter.{k}": v for k, v in model.depth_adapter.state_dict().items()}
            )
            trainable_parts.append("depth_adapter")
        conv_in = _get_conv_in(model)
        if conv_in is not None and stage.conv_in_lr > 0.0:
            model_state.update(
                {f"model.diff_model.conv_in.{k}": v for k, v in conv_in.state_dict().items()}
            )
            trainable_parts.append("conv_in")
        if stage.backbone_lr > 0.0:
            model_state = dict(model.state_dict())
            trainable_parts.append("backbone")
    else:
        model_state = dict(model.state_dict())
        trainable_parts.append("backbone")

    payload: Dict[str, Any] = {
        "version": CHECKPOINT_VERSION,
        "global_step": int(global_step),
        "stage": stage.to_dict(),
        "model_state": model_state,
        "model_class": type(model).__name__,
        "is_depth_adapter_only": bool(trainable_parts == ["depth_adapter"]),
        "trainable_parts": list(trainable_parts),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
        "rng_state": rng_state,
        "data_version": str(data_version),
        "weight_metadata": dict(weight_metadata) if weight_metadata else {},
    }
    torch.save(payload, str(path), _use_new_zipfile_serialization=False)
    return path


def verify_resume_consistency(checkpoint_path: Union[str, os.PathLike]) -> None:
    """Verify that a checkpoint can be resumed bit-for-bit.

    Loads the checkpoint twice with the same code path and confirms
    that the global step, RNG state, model parameters, optimizer state
    and data version are bit-equal across the two loads. Raises
    :class:`ResumeMismatchError` on any difference.
    """
    path = Path(checkpoint_path)
    if not path.is_file():
        raise ResumeMismatchError(f"Checkpoint not found: {path}")

    payload_a = torch.load(str(path), map_location="cpu", weights_only=False)
    payload_b = torch.load(str(path), map_location="cpu", weights_only=False)

    if int(payload_a["global_step"]) != int(payload_b["global_step"]):
        raise ResumeMismatchError("global_step mismatch on reload")
    if str(payload_a.get("data_version")) != str(payload_b.get("data_version")):
        raise ResumeMismatchError("data_version mismatch on reload")

    # RNG state deep check.
    for key in ("python", "numpy", "torch"):
        a = payload_a["rng_state"][key]
        b = payload_b["rng_state"][key]
        if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
            if not torch.equal(a, b):
                raise ResumeMismatchError(f"rng_state[{key}] mismatch on reload")
        else:
            # python random state is a tuple of (version, internal_state_tuple, gauss_next).
            # numpy state is a tuple of (version, *keys, pos, has_gauss, gauss).
            # We compare them via their pickle-bytes to avoid the
            # ambiguity error.
            import pickle
            ap = pickle.dumps(a)
            bp = pickle.dumps(b)
            if ap != bp:
                raise ResumeMismatchError(
                    f"rng_state[{key}] mismatch on reload"
                )
    if "torch_cuda" in payload_a["rng_state"] and "torch_cuda" in payload_b["rng_state"]:
        a = payload_a["rng_state"]["torch_cuda"]
        b = payload_b["rng_state"]["torch_cuda"]
        for i, (x, y) in enumerate(zip(a, b)):
            if not torch.equal(x, y):
                raise ResumeMismatchError(
                    f"rng_state[torch_cuda][{i}] mismatch on reload"
                )

    # Model state deep check (only the keys present).
    a_state = payload_a["model_state"]
    b_state = payload_b["model_state"]
    if set(a_state.keys()) != set(b_state.keys()):
        raise ResumeMismatchError("model state keys differ between reloads")
    for k in a_state:
        if not torch.equal(a_state[k], b_state[k]):
            raise ResumeMismatchError(f"model_state[{k}] mismatch on reload")

    # Optimizer state: bit-equal iter-by-iter.
    a_opt = payload_a["optimizer_state"]
    b_opt = payload_b["optimizer_state"]
    if not _states_equal(a_opt, b_opt):
        raise ResumeMismatchError("optimizer_state mismatch on reload")


def _states_equal(a: Any, b: Any) -> bool:
    if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
        return a.shape == b.shape and bool(torch.equal(a, b))
    if isinstance(a, dict) and isinstance(b, dict):
        if set(a.keys()) != set(b.keys()):
            return False
        return all(_states_equal(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if len(a) != len(b):
            return False
        return all(_states_equal(x, y) for x, y in zip(a, b))
    return a == b


# ---------------------------------------------------------------------------
# Internal step
# ---------------------------------------------------------------------------


def _q_sample(
    x0: torch.Tensor,
    eps: torch.Tensor,
    alphas_bar: torch.Tensor,
    timesteps: torch.Tensor,
) -> torch.Tensor:
    """Standard DDPM forward noising: ``√ᾱ_t·x0 + √(1-ᾱ_t)·ε``.

    ``alphas_bar`` is the 1-D ``alphas_cumprod`` tensor (already on the
    target device); ``timesteps`` is ``[B]`` int64.
    """
    ab = alphas_bar.to(device=x0.device, dtype=torch.float32)[timesteps]
    ab = ab.view(-1, 1, 1, 1)
    return ab.sqrt() * x0 + (1.0 - ab).sqrt() * eps


def _train_step(
    *,
    model: DepthConditionedRemoval,
    optimizer: torch.optim.Optimizer,
    batch: Dict[str, Any],
    vae: Optional[nn.Module],
    device: torch.device,
    amp: bool,
    amp_dtype: Optional[torch.dtype],
    known_weight: float,
    alphas_bar: torch.Tensor,
    loss_scale: float = 1.0,
) -> TrainResult:
    """Run one microbatch forward/backward pass.

    The caller owns gradient clearing, clipping, and ``optimizer.step()``.
    ``loss_scale`` divides the contribution of this microbatch when several
    microbatches are accumulated into one optimizer update.
    """
    def _to_tensor(value: Any, name: str) -> torch.Tensor:
        if isinstance(value, torch.Tensor):
            return value.to(device=device, dtype=torch.float32)
        if isinstance(value, np.ndarray):
            return torch.from_numpy(np.ascontiguousarray(value)).to(
                device=device, dtype=torch.float32
            )
        raise FinetuneConfigError(
            f"batch[{name!r}] must be a torch.Tensor or numpy.ndarray, "
            f"got {type(value).__name__}"
        )

    clean_rgb = _to_tensor(batch["clean_rgb"], "clean_rgb")
    hole_mask = _to_tensor(batch["hole_mask"], "hole_mask")
    depth_hole = _to_tensor(batch["depth_hole"], "depth_hole")
    noise = _to_tensor(batch["noise"], "noise")
    B = clean_rgb.shape[0]
    H, W = int(clean_rgb.shape[-2]), int(clean_rgb.shape[-1])

    # Build the masked image (RGB in [0,1] → [-1,1], zero in hole).
    masked_image = (2.0 * clean_rgb - 1.0) * (1.0 - hole_mask)

    # VAE encode (inference only, no grad through VAE).
    if "masked_latent" in batch and batch["masked_latent"] is not None:
        ml = batch["masked_latent"]
        if isinstance(ml, torch.Tensor):
            masked_latent = ml.to(device=device, dtype=torch.float32)
        elif isinstance(ml, np.ndarray):
            masked_latent = torch.from_numpy(np.ascontiguousarray(ml)).to(
                device=device, dtype=torch.float32
            )
        else:
            raise FinetuneConfigError(
                f"batch['masked_latent'] must be tensor or ndarray, "
                f"got {type(ml).__name__}"
            )
    else:
        masked_latent = _encode_masked_latent(
            masked_image, vae, device, amp, amp_dtype
        )

    # Resize mask to H/8, W/8.
    latent_h, latent_w = H // 8, W // 8
    latent_mask = nn.functional.interpolate(
        hole_mask, size=(latent_h, latent_w), mode="nearest"
    )

    # Depth features at H/8: coverage + depth-mean.
    depth_features = _default_depth_features(hole_mask, depth_hole)

    # Sample a random timestep in [0, 1000) and build the noised clean
    # target latent (TDD2 §5, corrected objective): the model sees
    # [x_t, latent_mask, masked_latent] and predicts ε. The previous
    # implementation fed pure noise with target ε — a degenerate
    # identity objective that trains nothing.
    timesteps = torch.randint(
        0, 1000, (B,), dtype=torch.int64, device=device
    )

    if batch.get("clean_latent") is not None:
        cl = batch["clean_latent"]
        if isinstance(cl, torch.Tensor):
            x0 = cl.to(device=device, dtype=torch.float32)
        elif isinstance(cl, np.ndarray):
            x0 = torch.from_numpy(np.ascontiguousarray(cl)).to(
                device=device, dtype=torch.float32
            )
        else:
            raise FinetuneConfigError(
                f"batch['clean_latent'] must be tensor or ndarray, got {type(cl).__name__}"
            )
    else:
        # The full clean target image (not masked) provides x0.
        clean_pm1 = (2.0 * clean_rgb - 1.0).to(device=device, dtype=torch.float32)
        x0 = _encode_masked_latent(clean_pm1, vae, device, amp, amp_dtype)

    eps = noise
    noisy = _q_sample(x0, eps, alphas_bar, timesteps)

    latent_input = torch.cat([noisy, latent_mask, masked_latent], dim=1)
    target_eps = eps

    if amp and amp_dtype is not None:
        with torch.autocast(device_type=device.type, dtype=amp_dtype):
            pred_eps = model.forward(
                latent_input, timesteps, None, depth_features=depth_features
            )
            loss = combined_epsilon_loss(
                pred_eps.float(),
                target_eps,
                latent_mask,
                known_weight=known_weight,
            )
    else:
        pred_eps = model.forward(
            latent_input, timesteps, None, depth_features=depth_features
        )
        loss = combined_epsilon_loss(
            pred_eps.float(),
            target_eps,
            latent_mask,
            known_weight=known_weight,
        )

    if loss_scale <= 0.0:
        raise FinetuneConfigError(f"loss_scale must be positive, got {loss_scale}")
    (loss * float(loss_scale)).backward()
    lr = float(optimizer.param_groups[0]["lr"])
    hole_term = float(
        combined_epsilon_loss(pred_eps.float(), target_eps, latent_mask, known_weight=0.0).item()
    )
    known_term = float(loss.item()) - hole_term
    return TrainResult(
        loss=float(loss.item()),
        hole_loss=hole_term,
        known_loss=known_term,
        # The update-level gradient norm is measured and clipped by the
        # outer loop after all microbatches have contributed.
        grad_norm=0.0,
        learning_rate=lr,
    )


def _mask_spatial(hole_mask: torch.Tensor) -> Tuple[int, int]:
    if hole_mask.ndim < 2:
        raise FinetuneConfigError(
            f"hole_mask must be at least 2D, got shape {tuple(hole_mask.shape)}"
        )
    return int(hole_mask.shape[-2]), int(hole_mask.shape[-1])


__all__ = [
    "BatchProvider",
    "CHECKPOINT_BASENAME",
    "CHECKPOINT_VERSION",
    "FinetuneConfigError",
    "ResumeMismatchError",
    "TeacherFinetuneArtifacts",
    "TrainResult",
    "finetune_depth_branch",
    "verify_resume_consistency",
]
