"""Console entry points for the moebius-finetune package (TDD2 §6).

Heavy ML imports (torch, diffusers, zipdepth) are deferred into the
command bodies so that ``import moebius_finetune.cli`` stays cheap and
CPU-safe. Each entry point is a thin argparse shell over the owning
subpackage's API:

* ``prepare-data``  — build the GRT split manifest (data/, Agent A)
* ``train-teacher`` — depth-conditioned fine-tune (training/teacher, B)
* ``evaluate``      — teacher eval rows over a holdout manifest (evaluation/, A)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Sequence


def prepare_data(argv: Optional[Sequence[str]] = None) -> int:
    """Build the GRT split manifest over a COCO image directory."""
    parser = argparse.ArgumentParser(prog="moebius-finetune-prepare-data")
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--output", required=True, help="manifest.json path")
    parser.add_argument("--num-train", type=int, required=True)
    parser.add_argument("--num-holdout-images", type=int, required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    from moebius_finetune.data.grt_dataset import build_manifest, save_manifest

    manifest = build_manifest(
        image_dir=args.image_dir,
        num_train=args.num_train,
        num_holdout_images=args.num_holdout_images,
        seed=args.seed,
    )
    save_manifest(manifest, args.output)
    counts: dict = {}
    for c in manifest["cases"]:
        counts[c["split"]] = counts.get(c["split"], 0) + 1
    print(json.dumps({"output": str(args.output), "splits": counts}))
    return 0


def train_teacher(argv: Optional[Sequence[str]] = None) -> int:
    """Fine-tune the depth-conditioned Moebius teacher (TDD2 §4/§5)."""
    parser = argparse.ArgumentParser(prog="moebius-finetune-train-teacher")
    parser.add_argument("--weights", required=True, help="Moebius .bin checkpoint")
    parser.add_argument("--vae-dir", required=True, help="VAE dir (config.json + bin)")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--kind",
        default="depth_only",
        choices=["depth_only", "unfreeze_conv_in", "local_smoke"],
    )
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--branch-lr", type=float, default=None)
    parser.add_argument("--conv-in-lr", type=float, default=None)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--zipdepth-checkpoint", default=None)
    parser.add_argument(
        "--init-adapter-from",
        default=None,
        help="warm-start the depth adapter from a stage-1 checkpoint",
    )
    args = parser.parse_args(argv)

    import torch

    from moebius_finetune.data.grt_dataset import (
        GrtTrainProvider,
        ZipDepthOnline,
        load_manifest,
    )
    from moebius_finetune.teachers import DepthConditionAdapter, DepthConditionedRemoval
    from moebius_finetune.teachers.loader import (
        load_removal_model,
        load_vae,
        sha256_of_file,
    )
    from moebius_finetune.training.teacher.finetune import finetune_depth_branch
    from moebius_finetune.training.teacher.recipe import recipe_from_yaml_dict

    recipe_kwargs: dict = {
        "kind": args.kind,
        "seed": args.seed,
        "grad_accum_steps": args.grad_accum,
        "save_every_steps": args.save_every,
        "log_every_steps": args.log_every,
        "amp": not args.no_amp,
    }
    if args.steps is not None:
        recipe_kwargs["steps"] = args.steps
    if args.branch_lr is not None:
        recipe_kwargs["branch_lr" if args.kind != "depth_only" else "lr"] = args.branch_lr
    if args.conv_in_lr is not None:
        recipe_kwargs["conv_in_lr"] = args.conv_in_lr
    recipe = recipe_from_yaml_dict(recipe_kwargs)

    model = load_removal_model(args.weights, strict=True)
    wrapped = DepthConditionedRemoval(model, DepthConditionAdapter())
    if args.init_adapter_from:
        import torch

        payload = torch.load(
            args.init_adapter_from, map_location="cpu", weights_only=False
        )
        sd = payload["model_state"]
        missing, unexpected = wrapped.load_state_dict(sd, strict=False)
        if unexpected:
            raise SystemExit(f"unexpected keys in {args.init_adapter_from}")
        adapter_keys = [k for k in sd if k.startswith("depth_adapter.")]
        print(
            f"[init] warm-started adapter from {args.init_adapter_from} "
            f"({len(adapter_keys)} tensors, stage={payload.get('stage', {}).get('name')})"
        )
    vae = load_vae(args.vae_dir)
    cases = load_manifest(args.manifest)
    predictor = ZipDepthOnline(checkpoint=args.zipdepth_checkpoint, device=args.device)
    provider = GrtTrainProvider(
        cases,
        image_dir=args.image_dir,
        predictor=predictor,
        split="train",
        seed=args.seed,
    )
    meta = {
        "path": str(Path(args.weights).absolute()),
        "sha256": sha256_of_file(args.weights),
    }
    artifacts = finetune_depth_branch(
        wrapped,
        recipe,
        batch_provider=provider,
        output_dir=args.output_dir,
        vae=vae,
        weight_metadata=meta,
        data_version="grt_v1",
        device=torch.device(args.device),
    )
    print(json.dumps(artifacts.to_dict(), indent=2, default=str))
    return 0


def evaluate(argv: Optional[Sequence[str]] = None) -> int:
    """Teacher evaluation rows over a holdout manifest (TDD2 §5)."""
    parser = argparse.ArgumentParser(prog="moebius-finetune-evaluate")
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        metavar="LABEL=CHECKPOINT[:nodepth]",
        help="evaluation row; CHECKPOINT 'initial' uses --weights",
    )
    parser.add_argument("--weights", required=True, help="initial Moebius .bin")
    parser.add_argument("--vae-dir", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--splits", default="holdout_images")
    parser.add_argument("--max-cases", type=int, default=64)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--zipdepth-checkpoint", default=None)
    parser.add_argument("--output", required=True, help="report markdown path")
    args = parser.parse_args(argv)

    from moebius_finetune.evaluation.teacher_eval import run_evaluation

    runs = []
    for spec in args.run:
        label, _, rest = spec.partition("=")
        nodepth = rest.endswith(":nodepth")
        ckpt = rest.removesuffix(":nodepth")
        runs.append({"label": label, "checkpoint": ckpt, "use_depth": not nodepth})
    run_evaluation(
        runs=runs,
        initial_weights=args.weights,
        vae_dir=args.vae_dir,
        manifest=args.manifest,
        image_dir=args.image_dir,
        splits=args.splits.split(","),
        max_cases=args.max_cases,
        seeds=args.seeds,
        device=args.device,
        zipdepth_checkpoint=args.zipdepth_checkpoint,
        output=args.output,
    )
    return 0


def cache_teacher(argv: Optional[Sequence[str]] = None) -> int:
    raise NotImplementedError("wired when the offline teacher cache is needed")


def train_codec(argv: Optional[Sequence[str]] = None) -> int:
    raise NotImplementedError("wired in the student distillation stage (old TDD §7)")


def train_student(argv: Optional[Sequence[str]] = None) -> int:
    raise NotImplementedError("wired in the student distillation stage (old TDD §7)")


def profile(argv: Optional[Sequence[str]] = None) -> int:
    raise NotImplementedError("wired in the deployment stage (old TDD §8)")


def export(argv: Optional[Sequence[str]] = None) -> int:
    raise NotImplementedError("wired in the deployment stage (old TDD §8)")
