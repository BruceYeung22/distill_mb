"""Visual comparison of the teacher under the two hole-fill conventions.

The current pipeline (``full_ft_v1.py`` / ``eval_moebius_small.py``) builds
the prompt as ``rgb * (1 - mask)``, i.e. the hole is **black**. The
project's own ``training/teacher/cache.py`` uses
``(2.0 * rgb - 1.0) * (1.0 - mask)``, i.e. the hole is **0** in [-1, 1] —
mid-gray. This renders the teacher's DDIM output under both so the
difference is visible rather than argued.

Columns: warp input / teacher @ hole=black / teacher @ hole=midgray /
ground truth, on a hole-centred crop.

Writes ``<out>/<source_id>.png`` and ``<out>/overview.png``.

Run::

    cd distill
    MOEBIUS_UPSTREAM_DIR=/home/dog/project/moebius_distill/Moebius \\
      .venv/bin/python scripts/viz_teacher_convention.py
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
import viz_moebius_small as vz  # noqa: E402

from moebius_finetune.students.moebius_small import MoebiusSmallStudent  # noqa: E402, F401

OUT_DIR = "/home/dog/datasets/moebius_finetune/runs/moebius_small_v1/viz/teacher_convention"
COLS = (
    "warp input (hole)",
    "teacher @ hole=BLACK (current)",
    "teacher @ hole=MIDGRAY",
    "ground truth",
)


@torch.no_grad()
def denoise(masked01: torch.Tensor, mask_t: torch.Tensor, *, teacher, taesd, scheduler, device, ids):
    lat_mask = F.interpolate(mask_t[None], size=(64, 64), mode="nearest")
    masked_lat = taesd.encoder(masked01)
    scheduler.set_timesteps(num_inference_steps=ev.STEPS, device=device)
    noisy = torch.randn_like(masked_lat)
    for timestep in scheduler.timesteps:
        t_step = timestep.to(device).unsqueeze(0)
        eps = ev.predict_noise(
            diff_model=teacher,
            noisy_latents=noisy,
            resized_masks=lat_mask,
            masked_latents=masked_lat,
            timesteps=t_step,
            input_ids=ids,
            guidance_scale=ev.GUIDANCE,
        )
        noisy = scheduler.step(eps, t_step, noisy, return_dict=False)[0]
    return (taesd.decoder(noisy) * 2 - 1).clamp(-1, 1)[0]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=str, default=OUT_DIR)
    parser.add_argument("--noise-seed", type=int, default=12345)
    args = parser.parse_args(argv)
    os.makedirs(args.out, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def log(message: str) -> None:
        print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)

    taesd = ev.TAESD(encoder_path=ev.TAESDXL_ENC, decoder_path=ev.TAESDXL_DEC).to(device).eval()
    for p in taesd.parameters():
        p.requires_grad_(False)

    teacher = ev.build_removal_model(ev.MOEBIUS_YAML, ev.NUM_EMBEDDINGS).to(device)
    ev.load_removal_model(teacher, ev.BASE_WEIGHTS, device, torch.float32, strict=True)
    teacher.load_state_dict(
        torch.load(ev.TEACHER_PT, map_location="cpu", weights_only=False)["ema_state"]
    )
    teacher.eval().requires_grad_(False)
    half = ev.NUM_EMBEDDINGS // 2
    ids = torch.cat(
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

    built = ev._build_cases(ev._holdout(), ev.ZipDepthOnline(device=str(device)), log)
    torch.manual_seed(args.noise_seed)
    np.random.seed(args.noise_seed)

    by_image: dict[str, list[dict]] = {}
    for index, sample in enumerate(built):
        hole01 = torch.from_numpy(sample["rgb_hole"][None]).to(device).float()
        mask_t = torch.from_numpy(sample["hole_mask"]).to(device).float()
        gt_pm1 = torch.from_numpy(sample["target"][None] * 2 - 1).to(device).float()[0]
        midgray01 = hole01 + 0.5 * mask_t[None]

        black = denoise(hole01, mask_t, teacher=teacher, taesd=taesd,
                        scheduler=scheduler, device=device, ids=ids)
        gray = denoise(midgray01, mask_t, teacher=teacher, taesd=taesd,
                       scheduler=scheduler, device=device, ids=ids)

        m = mask_t[0] > 0.5
        row = {
            "case_id": sample["case_id"],
            "warp": vz._to_np(torch.from_numpy(sample["rgb_hole"] * 2 - 1).to(device).float()),
            "black": vz._to_np(black),
            "gray": vz._to_np(gray),
            "gt": vz._to_np(gt_pm1),
            "l1_black": float((black[:, m] - gt_pm1[:, m]).abs().mean()),
            "l1_gray": float((gray[:, m] - gt_pm1[:, m]).abs().mean()),
            "box": vz._crop_box(mask_t[0].cpu().numpy()),
        }
        by_image.setdefault(sample["case_id"].split("__")[0], []).append(row)
        if (index + 1) % 8 == 0:
            log(f"  {index + 1}/{len(built)} cases")

    def render(rows, title, path):
        n = len(rows)
        fig, axes = plt.subplots(n, 4, figsize=(13, 3.3 * n), squeeze=False)
        for i, r in enumerate(rows):
            box = r["box"]
            panels = [r["warp"], r["black"], r["gray"], r["gt"]]
            for j, panel in enumerate(panels):
                axes[i][j].imshow(panel[box[0]:box[2], box[1]:box[3]])
                axes[i][j].set_xticks([])
                axes[i][j].set_yticks([])
                if i == 0:
                    axes[i][j].set_title(COLS[j], fontsize=10)
            axes[i][0].set_ylabel(
                f"{r['case_id']}\nblack {r['l1_black']:.3f} | midgray {r['l1_gray']:.3f}",
                fontsize=8,
            )
        fig.suptitle(title, fontsize=13)
        fig.tight_layout()
        fig.savefig(path, dpi=110)
        plt.close(fig)

    for source_id, rows in sorted(by_image.items()):
        rows = sorted(rows, key=lambda r: r["case_id"])
        render(rows, f"{source_id} — teacher DDIM, hole-convention comparison",
               f"{args.out}/{source_id}.png")
    first = [sorted(rows, key=lambda r: r["case_id"])[0] for _, rows in sorted(by_image.items())]
    render(first, "overview — teacher under both hole conventions (hole crop)",
           f"{args.out}/overview.png")

    a = np.array([[r["l1_black"], r["l1_gray"]] for rows in by_image.values() for r in rows])
    log(f"32 cases: mean in-hole L1  black={a[:,0].mean():.3f}  midgray={a[:,1].mean():.3f}")
    log(f"wrote {len(by_image)} figures + overview.png to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
