"""Teacher evaluation rows over a holdout manifest (TDD2 §5).

For every run spec (label, checkpoint or ``initial``, depth on/off):

1. build the teacher from the initial pretrained weights, then apply
   the finetuned trainable subset (``depth_adapter.*`` +
   ``model.diff_model.conv_in.*``) when a checkpoint is given;
2. for each holdout case × seed, run the 20-step DDIM teacher loop via
   ``cache_teacher_outputs`` (same scheduler config / seed protocol as
   training-cache generation) with the **online** ZipDepth predictor;
3. composite via ``contracts.inpaint`` and score
   ``hole_l1`` / ``hole_lpips`` / ``global_l1``;
4. aggregate and write a markdown report (composite
   ``S = 1·lpips + 3·hole_l1 + 1·global_l1``).
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from ..contracts import ConditionBatch, inpaint as contracts_inpaint
from ..data.grt_dataset import (
    CaseSpec,
    ZipDepthOnline,
    build_case,
    load_manifest,
)
from ..teachers.loader import MOEBIUS_PINNED_COMMIT, sha256_of_file
from ..teachers.wrapper import _default_depth_features
from .lpips_masked import build_lpips_alex, masked_lpips
from .metrics import global_l1, hole_l1, known_max_error


__all__ = ["run_evaluation", "pick_cases", "aggregate_rows"]

_VAE_SCALE_FALLBACK = 0.13025


def pick_cases(
    cases: Sequence[CaseSpec], max_cases: int, seed: int = 0
) -> List[CaseSpec]:
    """Deterministic unbiased subset (seeded shuffle, sorted pick)."""
    if max_cases <= 0 or len(cases) <= max_cases:
        return list(cases)
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(cases), size=max_cases, replace=False)
    return [cases[int(i)] for i in sorted(idx)]


def aggregate_rows(rows: List[Dict[str, float]]) -> Dict[str, float]:
    """Mean metrics over case rows + the composite score S."""
    if not rows:
        return {
            "n": 0, "hole_l1": 0.0, "hole_lpips": 0.0, "global_l1": 0.0,
            "S": 0.0, "known_max_error": 0.0, "hole_ratio": 0.0,
        }
    mean = {
        k: float(np.mean([r[k] for r in rows]))
        for k in ("hole_l1", "hole_lpips", "global_l1", "hole_ratio", "known_max_error")
    }
    mean["n"] = len(rows)
    mean["S"] = mean["hole_lpips"] + 3.0 * mean["hole_l1"] + mean["global_l1"]
    return mean


def _build_model_for_run(
    run: Dict[str, Any],
    initial_weights: str,
    device: str,
):
    import torch

    from ..teachers import (
        DepthConditionAdapter,
        DepthConditionedRemoval,
        OriginalRemovalBaseline,
    )
    from ..teachers.loader import load_removal_model

    base = load_removal_model(initial_weights, strict=True)
    ckpt = run["checkpoint"]
    if not run["use_depth"]:
        # The nodepth ablation still reflects checkpoint-trained weights:
        # apply the conv_in subset to the bare RemovalModel (depth-adapter
        # keys have no counterpart and are irrelevant with the branch off).
        if ckpt != "initial":
            payload = torch.load(ckpt, map_location="cpu", weights_only=False)
            mapped = {
                k.removeprefix("model."): v
                for k, v in payload["model_state"].items()
                if k.startswith("model.diff_model.conv_in.")
            }
            if mapped:
                _, unexpected = base.load_state_dict(mapped, strict=False)
                if unexpected:
                    raise ValueError(
                        f"checkpoint {ckpt} unexpected baseline keys: "
                        f"{sorted(unexpected)[:5]}"
                    )
        return OriginalRemovalBaseline(base).eval().to(device)
    wrapped = DepthConditionedRemoval(base, DepthConditionAdapter())
    if ckpt != "initial":
        payload = torch.load(ckpt, map_location="cpu", weights_only=False)
        sd = payload["model_state"]
        missing, unexpected = wrapped.load_state_dict(sd, strict=False)
        if unexpected:
            raise ValueError(
                f"checkpoint {ckpt} has unexpected keys: {sorted(unexpected)[:5]}"
            )
    return wrapped.eval().to(device)


def _eval_split(
    model,
    cases: Sequence[CaseSpec],
    *,
    image_dir: str,
    predictor,
    vae,
    device: str,
    seeds: Sequence[int],
    lpips_model,
    max_cases: int,
    teacher_sha: str,
    size: int = 512,
    log_every: int = 8,
) -> Dict[str, float]:
    import torch

    from ..training.teacher.cache import (
        cache_teacher_outputs,
        default_scheduler_config,
    )

    picked = pick_cases(list(cases), max_cases)
    scheduler_cfg = default_scheduler_config()
    rows: List[Dict[str, float]] = []
    t0 = time.time()
    for n, case in enumerate(picked, 1):
        built = build_case(
            case, image_dir=image_dir, predictor=predictor, size=size
        )
        condition = ConditionBatch(
            rgb_hole=built["rgb_hole"][None],
            hole_mask=built["hole_mask"][None],
            depth_hole=built["depth_hole"][None],
            noise=np.zeros((1, 4, size // 8, size // 8), np.float32),
        )
        mask_t = torch.from_numpy(condition.hole_mask).to(device)
        depth_t = torch.from_numpy(condition.depth_hole).to(device)
        rgb_t = torch.from_numpy(condition.rgb_hole).to(device)
        masked_image = (2.0 * rgb_t - 1.0) * (1.0 - mask_t)
        vae_dtype = next(vae.parameters()).dtype
        with torch.no_grad():
            encoded = vae.encode(masked_image.to(dtype=vae_dtype))
            latent = (
                encoded.latent_dist.mode()
                if hasattr(encoded, "latent_dist")
                else encoded.mode()
            )
            scale = float(getattr(vae.config, "scaling_factor", _VAE_SCALE_FALLBACK))
        masked_latent = (latent * scale).to(dtype=torch.float32)
        depth_features = _default_depth_features(mask_t, depth_t)

        def builder(case_id: str, seed: int, _c=condition, _m=masked_latent, _d=depth_features):
            return _c, _m, _d

        target = built["target"][None]
        for seed in seeds:
            entries = cache_teacher_outputs(
                model,
                case_list=[case.case_id],
                seed_list=[int(seed)],
                scheduler_cfg=scheduler_cfg,
                data_version="eval_v1",
                teacher_checkpoint_sha256=teacher_sha,
                moebius_commit=MOEBIUS_PINNED_COMMIT,
                vae=vae,
                case_batch_builder=builder,
            )
            candidate = entries[0].teacher_rgb  # [1,3,H,W] in [0,1]
            composed = contracts_inpaint(condition, candidate)
            rows.append(
                {
                    "case_id": case.case_id,
                    "seed": int(seed),
                    "hole_ratio": float(built["hole_ratio"]),
                    "hole_l1": hole_l1(composed, target, condition.hole_mask),
                    "hole_lpips": masked_lpips(
                        composed, target, condition.hole_mask, model=lpips_model,
                        device=device,
                    ),
                    "global_l1": global_l1(composed, target),
                    "known_max_error": float(
                        known_max_error(composed, target, condition.hole_mask)
                    ),
                }
            )
        if log_every and n % log_every == 0:
            print(
                f"[eval] {n}/{len(picked)} cases elapsed={time.time() - t0:.0f}s",
                flush=True,
            )
    return aggregate_rows(rows)


def run_evaluation(
    *,
    runs: Sequence[Dict[str, Any]],
    initial_weights: str,
    vae_dir: str,
    manifest: str,
    image_dir: str,
    splits: Sequence[str],
    max_cases: int = 64,
    seeds: Sequence[int] = (0,),
    device: str = "cuda",
    zipdepth_checkpoint: Optional[str] = None,
    output: str,
) -> Dict[str, Any]:
    """Run all (run, split) cells and write the markdown + JSON report."""
    import torch

    from ..teachers.loader import load_vae

    all_cases = load_manifest(manifest)
    predictor = ZipDepthOnline(checkpoint=zipdepth_checkpoint, device=device)
    vae = load_vae(vae_dir).to(device)
    lpips_model = build_lpips_alex(device)
    teacher_sha = sha256_of_file(initial_weights)

    results: Dict[str, Any] = {"cells": {}, "rows": {}}
    lines = [
        "# Teacher evaluation (TDD2 §5)",
        "",
        "S = 1·hole_lpips + 3·hole_l1 + 1·global_l1  (lower is better)",
        "",
    ]
    for split in splits:
        cases = [c for c in all_cases if c.split == split]
        if not cases:
            continue
        lines += [f"## split: {split}", "", "| run | n | hole_lpips | hole_l1 | global_l1 | S | known_max_err |", "|---|---|---|---|---|---|---|"]
        for run in runs:
            model = _build_model_for_run(run, initial_weights, device)
            summary = _eval_split(
                model,
                cases,
                image_dir=image_dir,
                predictor=predictor,
                vae=vae,
                device=device,
                seeds=seeds,
                lpips_model=lpips_model,
                max_cases=max_cases,
                teacher_sha=teacher_sha,
            )
            cell_key = f"{split}::{run['label']}"
            results["cells"][cell_key] = summary
            lines.append(
                f"| {run['label']} | {summary['n']} | {summary['hole_lpips']:.4f} | "
                f"{summary['hole_l1']:.4f} | {summary['global_l1']:.4f} | "
                f"{summary['S']:.4f} | {summary['known_max_error']:.6f} |"
            )
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        lines.append("")

    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    results["rows"] = lines
    Path(str(out) + ".json").write_text(
        json.dumps(results["cells"], indent=2), encoding="utf-8"
    )
    print(f"[eval] report -> {out}")
    return results
