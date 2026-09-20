"""Multi-granularity distillation of MoebiusSmallStudent from the fine-tuned teacher.

Plan: ``.omo/plans/moebius-small-distill.md`` (todo 7).

Pipeline per step (all latents live in the **TAESDXL** space, ``scaling_factor=1.0``):

1. Draw ``batch_size`` train cases that have a cached disparity and build
   them in a thread pool (image decode + warp are the CPU cost).
2. Encode ``rgb_hole`` and ``target`` with the frozen TAESDXL encoder, both
   fed in ``[0, 1]`` — no ``[-1, 1]`` conversion and no external scaling.
3. Build ``latent_mask`` with ``mode="nearest"`` (matching the script that
   produced the teacher checkpoint).
4. Sample ``t`` logit-normally (``1000 * sigmoid(randn)``) and keep both
   representations: the integer ``t_long`` drives ``add_noise``, the float
   ``t_float`` drives the student's sinusoidal embedding.
5. Run the frozen teacher under ``no_grad`` on the **conditional branch
   only** (``input_ids = [0..9]``): the KD target is the teacher's CFG-1.0
   output, measured to be at least as good as CFG 4.5 on the GRT holdout
   while costing a single (not doubled) forward.
6. Combine the four losses with the adaptive gradient-norm weights and step.
7. Keep an EMA (in RAM only — no extra disk artifacts) and checkpoint every
   ``--save-every`` steps as ``{step, model_state, ema_state, cfg}``.

Run::

    cd distill
    MOEBIUS_UPSTREAM_DIR=/home/dog/project/moebius_distill/Moebius \\
      .venv/bin/python -m moebius_finetune.training.student.train_small \\
      --steps 3000 --batch-size 32 --out-dir <dir>
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass

import numpy as np
import torch
import torch.nn.functional as F
from diffusers import DDPMScheduler

sys.path.insert(0, "/home/dog/project/HAMI/thirdParty/taesd")
sys.path.insert(0, "/home/dog/project/moebius_distill/Moebius")

from taesd import TAESD  # noqa: E402
from removal.v1_2 import build_removal_model  # noqa: E402

from moebius_finetune.data.grt_dataset import build_case, load_manifest  # noqa: E402
from moebius_finetune.students.moebius_small import MoebiusSmallStudent  # noqa: E402
from moebius_finetune.training.student.elatentlpips_setup import (  # noqa: E402
    load_elatentlpips,
)
from moebius_finetune.training.student.prompt import build_masked_prompt  # noqa: E402
from moebius_finetune.training.student.multigranular import (  # noqa: E402
    FeatureProjections,
    cal_adaptive_weights,
    cal_elatentlpips_loss,
    cal_kd_loss,
    cal_task_loss,
    total_loss,
)

IMG_DIR = "/home/dog/datasets/image/coco_train2017/train2017"
MANIFEST = "/home/dog/datasets/moebius_finetune/grt_manifest.json"
DISP_CACHE = "/home/dog/datasets/moebius_finetune/disparity"
FINAL_PT = "/home/dog/datasets/moebius_finetune/runs/taesdxl_moebius_ft/final.pt"
MOEBIUS_YAML = "/home/dog/project/moebius_distill/Moebius/config/model_cfg/moebius.yaml"
TAESDXL_ENC = "/home/dog/project/HAMI/thirdParty/taesd/taesdxl_encoder.pth"
TAESDXL_DEC = "/home/dog/project/HAMI/thirdParty/taesd/taesdxl_decoder.pth"
OUT_DIR = "/home/dog/datasets/moebius_finetune/runs/moebius_small_v1"

NUM_EMBEDDINGS = 20
LATENT_SIZE = 64
RGB_SIZE = 512


@dataclass
class TrainConfig:
    steps: int = 3000
    batch_size: int = 32
    lr: float = 1e-3
    weight_decay: float = 0.01
    warmup: int = 100
    grad_clip: float = 1.0
    ema_decay: float = 0.9999
    log_every: int = 50
    save_every: int = 500
    seed: int = 0
    workers: int = 8
    out_dir: str = OUT_DIR
    resume: str = ""


def _pick_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _train_cases() -> list:
    cases = [
        c
        for c in load_manifest(MANIFEST)
        if c.split == "train" and os.path.exists(f"{DISP_CACHE}/{c.source_id}.npy")
    ]
    if not cases:
        raise RuntimeError(f"no train cases with cached disparity in {MANIFEST}")
    return cases


def _build_one(case) -> dict:
    disp = np.load(f"{DISP_CACHE}/{case.source_id}.npy")
    return build_case(case, image_dir=IMG_DIR, disp=disp, size=RGB_SIZE)


def _build_batch(cases, pool: ThreadPoolExecutor) -> list[dict]:
    built = list(pool.map(_build_one, cases))
    return built


def _to_latents(built, taesd, device, dtype):
    """Stack the batch and encode with the frozen TAESDXL encoder."""
    target = torch.from_numpy(np.stack([b["target"] for b in built])).to(device, dtype)
    hole = torch.from_numpy(np.stack([b["rgb_hole"] for b in built])).to(device, dtype)
    mask = torch.from_numpy(np.stack([b["hole_mask"] for b in built])).to(device, dtype)

    prompt = build_masked_prompt(hole, mask)
    with torch.no_grad():
        x0 = taesd.encoder(target)
        masked = taesd.encoder(prompt)
    latent_mask = F.interpolate(mask, size=(LATENT_SIZE, LATENT_SIZE), mode="nearest")
    return x0, masked, latent_mask


def _cond_ids(batch: int, device) -> torch.Tensor:
    return torch.arange(NUM_EMBEDDINGS // 2, device=device).unsqueeze(0).expand(batch, -1)


_UNSET = object()


def train(
    cfg: TrainConfig,
    log,
    *,
    student=None,
    taesd=None,
    teacher=None,
    lpips=_UNSET,
    scheduler=None,
    cases=None,
    builder=None,
    device=None,
) -> dict:
    """Run the distillation loop.

    Every heavy component can be injected for tests; ``None`` means
    "build the real one" and ``lpips=_UNSET`` distinguishes "build" from
    an explicit ``None`` (which disables the perceptual term).
    """
    device = device or _pick_device()
    dtype = torch.float32
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    os.makedirs(cfg.out_dir, exist_ok=True)

    if student is None:
        student = MoebiusSmallStudent()
    student = student.to(device)
    projections = FeatureProjections(student_channels=student.channels).to(device)
    ema = copy.deepcopy(student).eval()
    for p in ema.parameters():
        p.requires_grad_(False)

    total_params = sum(p.numel() for p in student.parameters())
    log(
        f"student params={total_params} "
        f"projection params={sum(p.numel() for p in projections.parameters())}"
    )

    if taesd is None:
        taesd = TAESD(encoder_path=TAESDXL_ENC, decoder_path=TAESDXL_DEC)
    taesd = taesd.to(device, dtype).eval()
    for p in taesd.parameters():
        p.requires_grad_(False)

    if teacher is None:
        teacher = build_removal_model(MOEBIUS_YAML, NUM_EMBEDDINGS).to(device, dtype)
        payload = torch.load(FINAL_PT, map_location="cpu", weights_only=False)
        teacher.load_state_dict(payload["ema_state"])
    teacher = teacher.to(device, dtype)
    teacher.eval().requires_grad_(False)

    if scheduler is None:
        scheduler = DDPMScheduler(
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            num_train_timesteps=1000,
            clip_sample=False,
        )
        scheduler.alphas_cumprod = scheduler.alphas_cumprod.to(device)

    if lpips is _UNSET:
        lpips = load_elatentlpips(device=str(device))
    elif lpips is not None:
        lpips = lpips.to(device).eval().requires_grad_(False)

    trainable = list(student.parameters()) + list(projections.parameters())
    optimizer = torch.optim.AdamW(trainable, lr=cfg.lr, weight_decay=cfg.weight_decay)

    start_step = 0
    if cfg.resume:
        ckpt = torch.load(cfg.resume, map_location="cpu", weights_only=False)
        student.load_state_dict(ckpt["model_state"])
        ema.load_state_dict(ckpt["ema_state"])
        if "projection_state" in ckpt:
            projections.load_state_dict(ckpt["projection_state"])
        start_step = int(ckpt["step"])
        log(f"resumed from {cfg.resume} at step {start_step}")

    if cases is None:
        cases = _train_cases()
    rng = np.random.default_rng(cfg.seed + start_step)
    log(f"data: {len(cases)} train cases with cached disparity")

    anchors = (student.enc2[-2].pw2.weight, student.head[-1].weight)
    history: list[dict] = []
    t0 = time.perf_counter()
    pool = ThreadPoolExecutor(max_workers=cfg.workers) if builder is None else None
    try:
        for step in range(start_step, cfg.steps):
            idx = rng.integers(0, len(cases), size=cfg.batch_size)
            subset = [cases[int(i)] for i in idx]
            if builder is not None:
                built = builder(subset)
            else:
                built = _build_batch(subset, pool)
            x0, masked, latent_mask = _to_latents(built, taesd, device, dtype)

            noise = torch.randn_like(x0)
            t_float = 1000.0 * torch.sigmoid(torch.randn(cfg.batch_size, device=device))
            t_long = t_float.long()
            noisy = scheduler.add_noise(x0, noise, t_long)
            x9 = torch.cat([noisy, latent_mask, masked], dim=1)

            with torch.no_grad():
                teacher_out = teacher(x9, t_long, _cond_ids(cfg.batch_size, device))

            student_out = student(x9, t_float)
            featkd, outkd = cal_kd_loss(student_out, teacher_out, projections)
            task = cal_task_loss(student_out, noise)
            if lpips is None:
                lpips_loss = None
            else:
                lpips_loss = cal_elatentlpips_loss(
                    student_out, noise, lpips, scheduler, t_long, noisy
                )

            feat_w, out_w_outkd, out_w_lpips, diag = cal_adaptive_weights(
                featkd,
                task,
                outkd,
                lpips_loss,
                feat_anchor=anchors[0],
                out_anchor=anchors[1],
            )
            loss = total_loss(
                featkd,
                task,
                outkd,
                lpips_loss,
                feat_weight_task=feat_w,
                out_weight_outkd=out_w_outkd,
                out_weight_elatentlpips=out_w_lpips,
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, cfg.grad_clip)
            lr_scale = min(1.0, (step + 1 - start_step) / max(1, cfg.warmup))
            for group in optimizer.param_groups:
                group["lr"] = cfg.lr * lr_scale
            optimizer.step()

            with torch.no_grad():
                for p, ep in zip(student.parameters(), ema.parameters()):
                    ep.mul_(cfg.ema_decay).add_(p, alpha=1.0 - cfg.ema_decay)

            record = {
                "step": step + 1,
                "loss": float(loss.detach()),
                "loss_featkd": float(featkd.detach()),
                "loss_outkd": float(outkd.detach()),
                "loss_task": float(task.detach()),
                "loss_elatentlpips": (
                    float(lpips_loss.detach()) if lpips_loss is not None else 0.0
                ),
                "feat_weight_task": float(feat_w),
                "out_weight_outkd": float(out_w_outkd),
                "out_weight_elatentlpips": float(out_w_lpips),
                "lr": optimizer.param_groups[0]["lr"],
                **diag,
            }
            history.append(record)

            if (step + 1) % cfg.log_every == 0 or step == start_step:
                elapsed = time.perf_counter() - t0
                done = step + 1 - start_step
                log(
                    f"step={step + 1}/{cfg.steps} loss={record['loss']:.4f} "
                    f"featkd={record['loss_featkd']:.4f} outkd={record['loss_outkd']:.4f} "
                    f"task={record['loss_task']:.4f} lpips={record['loss_elatentlpips']:.4f} "
                    f"w_feat={record['feat_weight_task']:.4f} w_outkd={record['out_weight_outkd']:.4f} "
                    f"w_lpips={record['out_weight_elatentlpips']:.4f} "
                    f"lr={record['lr']:.2e} ({elapsed / done:.2f}s/step)"
                )

            if (step + 1) % cfg.save_every == 0:
                _save(cfg, student, ema, step + 1, projections)
                log(f"saved step_{step + 1:05d}.pt")
    finally:
        if pool is not None:
            pool.shutdown(wait=False)

    _save(cfg, student, ema, cfg.steps, projections)
    log(f"saved final.pt")
    elapsed = time.perf_counter() - t0
    done = cfg.steps - start_step
    return {
        "steps": cfg.steps,
        "elapsed_s": elapsed,
        "s_per_step": elapsed / max(1, done),
        "history": history,
    }

def _save(cfg: TrainConfig, student, ema, step: int, projections=None) -> None:
    payload = {
        "step": step,
        "model_state": student.state_dict(),
        "ema_state": ema.state_dict(),
        "cfg": asdict(cfg),
    }
    if projections is not None:
        payload["projection_state"] = projections.state_dict()
    torch.save(payload, f"{cfg.out_dir}/step_{step:05d}.pt")
    if step == cfg.steps:
        torch.save(payload, f"{cfg.out_dir}/final.pt")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=TrainConfig.steps)
    parser.add_argument("--batch-size", type=int, default=TrainConfig.batch_size)
    parser.add_argument("--lr", type=float, default=TrainConfig.lr)
    parser.add_argument("--warmup", type=int, default=TrainConfig.warmup)
    parser.add_argument("--seed", type=int, default=TrainConfig.seed)
    parser.add_argument("--workers", type=int, default=TrainConfig.workers)
    parser.add_argument("--log-every", type=int, default=TrainConfig.log_every)
    parser.add_argument("--save-every", type=int, default=TrainConfig.save_every)
    parser.add_argument("--out-dir", type=str, default=TrainConfig.out_dir)
    parser.add_argument("--resume", type=str, default="")
    args = parser.parse_args(argv)

    cfg = TrainConfig(
        steps=args.steps,
        batch_size=args.batch_size,
        lr=args.lr,
        warmup=args.warmup,
        seed=args.seed,
        workers=args.workers,
        log_every=args.log_every,
        save_every=args.save_every,
        out_dir=args.out_dir,
        resume=args.resume,
    )
    os.makedirs(cfg.out_dir, exist_ok=True)
    log_path = f"{cfg.out_dir}/train.log"

    def log(message: str) -> None:
        line = f"[{time.strftime('%H:%M:%S')}] {message}"
        print(line, flush=True)
        with open(log_path, "a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    log(f"START {json.dumps(asdict(cfg))}")
    result = train(cfg, log)
    log(
        f"DONE steps={result['steps']} elapsed={result['elapsed_s'] / 60:.1f}min "
        f"s/step={result['s_per_step']:.2f}"
    )
    with open(f"{cfg.out_dir}/history.json", "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
