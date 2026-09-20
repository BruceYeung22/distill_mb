"""Side-by-side visual QA for MoebiusSmallStudent vs the teacher.

Reuses the evaluator's protocol so the pictures correspond to the same 32
holdout cases and the same initial noise: warp input (hole zeroed) /
student / teacher (CFG 1.0) / ground truth, on the full frame and on a
hole-centred crop.

Writes ``<out>/full_<source_id>.png``, ``<out>/crop_<source_id>.png`` and
``<out>/overview.png``.

Run::

    cd distill
    MOEBIUS_UPSTREAM_DIR=/home/dog/project/moebius_distill/Moebius \\
      .venv/bin/python scripts/viz_moebius_small.py
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import eval_moebius_small as ev  # noqa: E402

from moebius_finetune.students.moebius_small import MoebiusSmallStudent  # noqa: E402
from moebius_finetune.training.student.prompt import build_masked_prompt  # noqa: E402

OUT_DIR = "/home/dog/datasets/moebius_finetune/runs/moebius_small_v1/viz"
STUDENT_PT = "/home/dog/datasets/moebius_finetune/runs/moebius_small_v1/final.pt"
COLS = ("warp input (hole)", "student (10.2M)", "teacher (226M)", "ground truth")


def _to_np(tensor_pm1: torch.Tensor) -> np.ndarray:
    arr = ((tensor_pm1.detach().cpu() + 1) / 2).clamp(0, 1).permute(1, 2, 0).numpy()
    return arr


def _hole_l1(pred_pm1: torch.Tensor, gt_pm1: torch.Tensor, mask: torch.Tensor) -> float:
    m = mask > 0.5
    if not bool(m.any()):
        return 0.0
    return float((pred_pm1[:, m] - gt_pm1[:, m]).abs().mean())


@torch.no_grad()
def _predict(sample: dict, *, model, teacher, taesd, scheduler, device, input_ids_eval):
    hole_pm1 = torch.from_numpy(sample["rgb_hole"][None] * 2 - 1).to(device).float()
    mask_t = torch.from_numpy(sample["hole_mask"]).to(device).float()
    lat_mask = F.interpolate(mask_t[None], size=(64, 64), mode="nearest")
    prompt01 = build_masked_prompt(
        torch.from_numpy(sample["rgb_hole"][None]).to(device).float(), mask_t[None]
    )
    masked_lat = taesd.encoder(prompt01)

    scheduler.set_timesteps(num_inference_steps=ev.STEPS, device=device)
    noisy = torch.randn_like(masked_lat)
    for timestep in scheduler.timesteps:
        t_step = timestep.to(device).unsqueeze(0)
        if teacher is not None:
            eps = ev.predict_noise(
                diff_model=teacher,
                noisy_latents=noisy,
                resized_masks=lat_mask,
                masked_latents=masked_lat,
                timesteps=t_step,
                input_ids=input_ids_eval,
                guidance_scale=ev.GUIDANCE,
            )
        else:
            x9 = torch.cat([noisy, lat_mask, masked_lat], dim=1)
            eps = model(x9, t_step.float()).sample
        noisy = scheduler.step(eps, t_step, noisy, return_dict=False)[0]
    pred = (taesd.decoder(noisy) * 2 - 1).clamp(-1, 1)[0]
    return pred, mask_t[0], hole_pm1[0]


def _crop_box(mask: np.ndarray, size: int = 176, margin: int = 24) -> tuple[int, int, int, int]:
    ys, xs = np.nonzero(mask > 0.5)
    if len(ys) == 0:
        c = mask.shape[0] // 2
        return c - size // 2, c - size // 2, c + size // 2, c + size // 2
    y0, y1 = int(ys.min()) - margin, int(ys.max()) + margin
    x0, x1 = int(xs.min()) - margin, int(xs.max()) + margin
    cy, cx = (y0 + y1) // 2, (x0 + x1) // 2
    half = size // 2
    y0, x0 = max(0, cy - half), max(0, cx - half)
    y0, x0 = min(y0, mask.shape[0] - size), min(x0, mask.shape[1] - size)
    return y0, x0, y0 + size, x0 + size


def _figure(rows: list[dict], title: str, path: str, crop: bool) -> None:
    n = len(rows)
    fig, axes = plt.subplots(n, 4, figsize=(13, 3.3 * n), squeeze=False)
    for i, row in enumerate(rows):
        box = row["box"] if crop else None
        panels = [
            row["warp"],
            row["student"],
            row["teacher"],
            row["gt"],
        ]
        for j, panel in enumerate(panels):
            img = panel[box[0] : box[2], box[1] : box[3]] if box else panel
            axes[i][j].imshow(img)
            axes[i][j].set_xticks([])
            axes[i][j].set_yticks([])
            if i == 0:
                axes[i][j].set_title(COLS[j], fontsize=11)
        axes[i][0].set_ylabel(
            f"{row['case_id']}\nstudent {row['student_l1']:.3f} | teacher {row['teacher_l1']:.3f}",
            fontsize=8,
        )
    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--student", type=str, default=STUDENT_PT)
    parser.add_argument("--out", type=str, default=OUT_DIR)
    parser.add_argument("--noise-seed", type=int, default=ev.__dict__.get("NOISE_SEED", 12345))
    args = parser.parse_args(argv)

    os.makedirs(args.out, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def log(message: str) -> None:
        print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)

    taesd = ev.TAESD(encoder_path=ev.TAESDXL_ENC, decoder_path=ev.TAESDXL_DEC)
    taesd = taesd.to(device).eval()
    for p in taesd.parameters():
        p.requires_grad_(False)

    student = MoebiusSmallStudent().to(device)
    payload = torch.load(args.student, map_location="cpu", weights_only=False)
    student.load_state_dict(payload["model_state"])
    student.eval().requires_grad_(False)

    teacher = ev.build_removal_model(ev.MOEBIUS_YAML, ev.NUM_EMBEDDINGS).to(device)
    ev.load_removal_model(teacher, ev.BASE_WEIGHTS, device, torch.float32, strict=True)
    teacher.load_state_dict(
        torch.load(ev.TEACHER_PT, map_location="cpu", weights_only=False)["ema_state"]
    )
    teacher.eval().requires_grad_(False)
    half = ev.NUM_EMBEDDINGS // 2
    input_ids_eval = torch.cat(
        [
            torch.tensor([list(range(half, ev.NUM_EMBEDDINGS))], dtype=torch.int64, device=device),
            torch.tensor([list(range(half))], dtype=torch.int64, device=device),
        ]
    ).contiguous()

    scheduler = ev.DDIMScheduler(
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        num_train_timesteps=1000,
        clip_sample=False,
    )
    scheduler.alphas_cumprod = scheduler.alphas_cumprod.to(device)

    cases = ev._holdout()
    built = ev._build_cases(cases, ev.ZipDepthOnline(device=str(device)), log)

    torch.manual_seed(args.noise_seed)
    np.random.seed(args.noise_seed)

    by_image: dict[str, list[dict]] = {}
    for index, sample in enumerate(built):
        gt_pm1 = torch.from_numpy(sample["target"][None] * 2 - 1).to(device).float()[0]
        s_pred, mask_t, warp_pm1 = _predict(
            sample, model=student, teacher=None, taesd=taesd, scheduler=scheduler,
            device=device, input_ids_eval=None,
        )
        t_pred, _, _ = _predict(
            sample, model=None, teacher=teacher, taesd=taesd, scheduler=scheduler,
            device=device, input_ids_eval=input_ids_eval,
        )
        row = {
            "case_id": sample["case_id"],
            "warp": _to_np(warp_pm1),
            "student": _to_np(s_pred),
            "teacher": _to_np(t_pred),
            "gt": _to_np(gt_pm1),
            "student_l1": _hole_l1(s_pred, gt_pm1, mask_t),
            "teacher_l1": _hole_l1(t_pred, gt_pm1, mask_t),
            "box": _crop_box(mask_t.cpu().numpy()),
        }
        by_image.setdefault(sample["case_id"].split("__")[0], []).append(row)
        if (index + 1) % 8 == 0:
            log(f"  {index + 1}/{len(built)} cases rendered")

    for source_id, rows in sorted(by_image.items()):
        rows = sorted(rows, key=lambda r: r["case_id"])
        _figure(rows, f"{source_id} — GRT holdout, 10-step DDIM, seed {args.noise_seed}",
                f"{args.out}/full_{source_id}.png", crop=False)
        _figure(rows, f"{source_id} — hole crop", f"{args.out}/crop_{source_id}.png", crop=True)
    log(f"wrote {len(by_image)} full + {len(by_image)} crop figures to {args.out}")

    first = [sorted(rows, key=lambda r: r["case_id"])[0] for _, rows in sorted(by_image.items())]
    _figure(first, "overview — first combo per holdout image (hole crop)",
            f"{args.out}/overview.png", crop=True)
    log("wrote overview.png")

    s_all = [r["student_l1"] for rows in by_image.values() for r in rows]
    t_all = [r["teacher_l1"] for rows in by_image.values() for r in rows]
    log(f"student mean {sum(s_all)/len(s_all):.4f} | teacher mean {sum(t_all)/len(t_all):.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
