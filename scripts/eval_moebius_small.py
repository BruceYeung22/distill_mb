"""GRT holdout evaluator for MoebiusSmallStudent (plan todo 9).

Replicates ``runs/taesdxl_moebius_ft/eval_trajectory.py`` step for step:

* 8 holdout images chosen by ``np.random.default_rng(7)`` over the sorted
  ``holdout_images`` source ids, all four (dmax, direction) combos → 32 cases;
* disparity computed **online** with ZipDepth (the holdout split has no
  cached disparity) and reused across a source image's four cases;
* 10-step DDIM from pure noise, TAESDXL encode/decode, hole-L1 over
  ``mask > 0.5``;
* the noise draw is pinned with ``--noise-seed`` (default ``12345``, the
  value used for the teacher's CFG measurement) so runs are comparable.
  The reference draw order is by ``case_id``.

``--teacher-check`` evaluates the fine-tuned teacher (CFG 1.0 — the
measured-equal-or-better target used for distillation) and is the
protocol's self-test: it should land within ±0.002 of 0.0698.

Run::

    cd distill
    MOEBIUS_UPSTREAM_DIR=/home/dog/project/moebius_distill/Moebius \\
      .venv/bin/python scripts/eval_moebius_small.py --teacher-check
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from diffusers import DDIMScheduler

sys.path.insert(0, "/home/dog/project/HAMI/thirdParty/taesd")
sys.path.insert(0, "/home/dog/project/moebius_distill/Moebius")
sys.path.insert(0, "/home/dog/project/moebius_distill/distill/src")

from taesd import TAESD  # noqa: E402
from removal.v1_2 import build_removal_model, load_removal_model  # noqa: E402
from utils_infer import predict_noise  # noqa: E402

from moebius_finetune.data.grt_dataset import (  # noqa: E402
    ZipDepthOnline,
    _load_source_rgb01,
    build_case,
    load_manifest,
)
from moebius_finetune.students.moebius_small import MoebiusSmallStudent  # noqa: E402
from moebius_finetune.training.student.prompt import build_masked_prompt  # noqa: E402

IMG_DIR = "/home/dog/datasets/image/coco_train2017/train2017"
MANIFEST = "/home/dog/datasets/moebius_finetune/grt_manifest.json"
BASE_WEIGHTS = "/home/dog/project/moebius_distill/Moebius/weights/moebius/pretrained/diffusion_pytorch_model.bin"
MOEBIUS_YAML = "/home/dog/project/moebius_distill/Moebius/config/model_cfg/moebius.yaml"
TAESDXL_ENC = "/home/dog/project/HAMI/thirdParty/taesd/taesdxl_encoder.pth"
TAESDXL_DEC = "/home/dog/project/HAMI/thirdParty/taesd/taesdxl_decoder.pth"
TEACHER_PT = "/home/dog/datasets/moebius_finetune/runs/taesdxl_moebius_ft/final.pt"
STUDENT_PT = "/home/dog/datasets/moebius_finetune/runs/moebius_small_v1/final.pt"
OUT_JSON = "/home/dog/datasets/moebius_finetune/runs/moebius_small_v1/eval.json"

NUM_EMBEDDINGS = 20
STEPS = 10
GUIDANCE = 1.0


def _holdout() -> list:
    cases_all = load_manifest(MANIFEST)
    cases_hold = [c for c in cases_all if c.split == "holdout_images"]
    hold_imgs = sorted({c.source_id for c in cases_hold})
    rng = np.random.default_rng(7)
    sel = [hold_imgs[i] for i in rng.permutation(len(hold_imgs))[:8]]
    return sorted(
        [c for c in cases_hold if c.source_id in sel], key=lambda c: c.case_id
    )


def _build_cases(cases, predictor, log) -> list[dict]:
    images = sorted({c.source_id for c in cases})
    log(f"computing disparity for {len(images)} holdout images via ZipDepth (online)")
    start = time.perf_counter()
    disp_cache = {}
    for image_id in images:
        rgb01 = _load_source_rgb01(IMG_DIR, image_id, 512)
        disp_cache[image_id] = np.asarray(
            predictor(
                np.clip(
                    rgb01.transpose(1, 2, 0)[..., ::-1] * 255.0, 0, 255
                ).astype(np.uint8)
            ),
            dtype=np.float32,
        )
    log(f"  disparity done ({time.perf_counter() - start:.0f}s)")

    built = []
    for case in cases:
        sample = build_case(case, image_dir=IMG_DIR, disp=disp_cache[case.source_id])
        built.append(
            {
                "case_id": case.case_id,
                "rgb_hole": sample["rgb_hole"],
                "hole_mask": sample["hole_mask"],
                "target": sample["target"],
            }
        )
    return built


def _hole_l1(pred_pm1: torch.Tensor, gt_pm1: torch.Tensor, mask: torch.Tensor) -> float:
    """Mean ``|pred - gt|`` over the hole.

    ``mask`` arrives as ``[1, H, W]`` (``build_case``'s layout), so the 2-D
    mask is ``mask[0]``. The upstream ``eval_trajectory.py`` writes
    ``mask_t[0, 0]``, which on this layout selects a single image **row** —
    scoring one scanline of the hole instead of the hole, and returning a
    fake ``0.0`` whenever that row misses the hole.
    """
    m = (mask[0] if mask.dim() == 3 else mask) > 0.5
    if not bool(m.any()):
        return 0.0
    return float((pred_pm1[0][:, m] - gt_pm1[0][:, m]).abs().mean())


@torch.no_grad()
def evaluate(
    cases: list[dict],
    *,
    student,
    teacher,
    taesd,
    scheduler,
    device,
    input_ids_eval,
    log,
) -> dict:
    h_l1: list[float] = []
    per_case: list[dict] = []
    start = time.perf_counter()
    for index, case in enumerate(cases):
        prompt01 = torch.from_numpy(case["rgb_hole"][None]).to(device).float()
        gt_pm1 = torch.from_numpy(case["target"][None] * 2 - 1).to(device).float()
        mask_t = torch.from_numpy(case["hole_mask"]).to(device).float()
        lat_mask = F.interpolate(mask_t[None], size=(64, 64), mode="nearest")
        prompt01 = build_masked_prompt(prompt01, mask_t[None])

        with torch.no_grad():
            masked_lat = taesd.encoder(prompt01)
            scheduler.set_timesteps(num_inference_steps=STEPS, device=device)
            noise = torch.randn_like(masked_lat)
            noisy = noise
            for timestep in scheduler.timesteps:
                t_step = timestep.to(device).unsqueeze(0)
                if teacher is not None:
                    eps = predict_noise(
                        diff_model=teacher,
                        noisy_latents=noisy,
                        resized_masks=lat_mask,
                        masked_latents=masked_lat,
                        timesteps=t_step,
                        input_ids=input_ids_eval,
                        guidance_scale=GUIDANCE,
                    )
                else:
                    x9 = torch.cat([noisy, lat_mask, masked_lat], dim=1)
                    eps = student(x9, t_step.float()).sample
                noisy = scheduler.step(eps, t_step, noisy, return_dict=False)[0]
            out_01 = taesd.decoder(noisy)
            pred_pm1 = (out_01 * 2 - 1).clamp(-1, 1)
            value = _hole_l1(pred_pm1, gt_pm1, mask_t)
        h_l1.append(value)
        per_case.append({"case_id": case["case_id"], "hole_l1": value})
        if (index + 1) % 8 == 0:
            log(
                f"  {index + 1}/{len(cases)} cases "
                f"running_mean={sum(h_l1) / len(h_l1):.4f} "
                f"({time.perf_counter() - start:.0f}s)"
            )
    return {
        "hole_l1": sum(h_l1) / len(h_l1),
        "per_case": per_case,
        "n_cases": len(h_l1),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher-check", action="store_true")
    parser.add_argument("--student", type=str, default=STUDENT_PT)
    parser.add_argument(
        "--state",
        choices=("ema_state", "model_state"),
        default="ema_state",
        help="which weights to evaluate from the checkpoint",
    )
    parser.add_argument("--out", type=str, default=OUT_JSON)
    parser.add_argument("--noise-seed", type=int, default=12345)
    args = parser.parse_args(argv)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32

    def log(message: str) -> None:
        print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)

    taesd = TAESD(encoder_path=TAESDXL_ENC, decoder_path=TAESDXL_DEC)
    taesd = taesd.to(device, dtype).eval()
    for p in taesd.parameters():
        p.requires_grad_(False)

    scheduler = DDIMScheduler(
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        num_train_timesteps=1000,
        clip_sample=False,
    )
    scheduler.alphas_cumprod = scheduler.alphas_cumprod.to(device)

    student = None
    teacher = None
    if args.teacher_check:
        teacher = build_removal_model(MOEBIUS_YAML, NUM_EMBEDDINGS).to(device, dtype)
        load_removal_model(teacher, BASE_WEIGHTS, device, dtype, strict=True)
        payload = torch.load(TEACHER_PT, map_location="cpu", weights_only=False)
        teacher.load_state_dict(payload["ema_state"])
        teacher.eval().requires_grad_(False)
        half = NUM_EMBEDDINGS // 2
        uncond = torch.tensor([list(range(half, NUM_EMBEDDINGS))], dtype=torch.int64, device=device)
        cond = torch.tensor([list(range(half))], dtype=torch.int64, device=device)
        input_ids_eval = torch.cat([uncond, cond]).contiguous()
        label = f"teacher (CFG {GUIDANCE}) from {TEACHER_PT}"
    else:
        student = MoebiusSmallStudent().to(device)
        payload = torch.load(args.student, map_location="cpu", weights_only=False)
        student.load_state_dict(payload[args.state])
        student.eval().requires_grad_(False)
        input_ids_eval = None
        label = f"student {args.state} from {args.student} (step {payload.get('step')})"

    cases = _holdout()
    log(f"{label}: {len(cases)} holdout cases")
    built = _build_cases(cases, ZipDepthOnline(device=str(device)), log)

    torch.manual_seed(args.noise_seed)
    np.random.seed(args.noise_seed)
    result = evaluate(
        built,
        student=student,
        teacher=teacher,
        taesd=taesd,
        scheduler=scheduler,
        device=device,
        input_ids_eval=input_ids_eval,
        log=log,
    )
    result.update(
        {
            "label": label,
            "noise_seed": args.noise_seed,
            "steps": STEPS,
            "guidance_scale": GUIDANCE if teacher is not None else None,
            "taesd": "taesdxl",
        }
    )
    log(f"hole_l1 = {result['hole_l1']:.6f} over {result['n_cases']} cases")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=1)
    log(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
