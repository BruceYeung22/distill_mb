"""Real-COCO block-level profile of the Moebius teacher.

Validates the synthetic-GRT findings of scripts/profile_teacher_blocks.py
against the 32-case real COCO holdout (same selection as
scripts/eval_moebius_small.py: np.random.default_rng(7) permutation,
first 8 holdout_images, all 4 (dmax, direction) combos -> 32 cases).

Per-case pipeline:
  1. Load COCO RGB 512x512 from /home/dog/datasets/image/coco_train2017/train2017
  2. Compute disparity online via ZipDepth (the holdout split has no
     cached disparity; same as eval_moebius_small._build_cases).
  3. build_case -> rgb_hole, hole_mask, depth_hole, target
  4. Mid-gray prompt (HOLE_FILL_VALUE = 0.5 via training/student/prompt.py)
  5. TAESDXL encode target + prompt; nearest-interp mask to 64x64
  6. x9 = [q_sample(x0) | nearest(mask) | encoder(prompt)]
  7. Run baseline forward + 30 block-level zero-ablation forwards.
  8. Per-case in-hole L1 (mask > 0.5) of pred_noise vs noise.

Strictly inside distill/; no checkpoints touched.

Reuses the model/hook helpers from scripts/profile_teacher_blocks.py
(same directory, sys.path-injected) so the block enumeration and
zero-hook semantics are identical to the synthetic run.

Usage:
    cd distill && MOEBIUS_UPSTREAM_DIR=/home/dog/project/moebius_distill/Moebius \\
      .venv/bin/python scripts/profile_teacher_real.py
"""

from __future__ import annotations

import os
import sys
import time
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

# Re-use everything from the synthetic block profiler (same dir).
sys.path.insert(0, "/home/dog/project/HAMI/thirdParty/taesd")
sys.path.insert(0, "/home/dog/project/moebius_distill/Moebius")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import profile_teacher_blocks as ptb

from moebius_finetune.data.grt_dataset import (  # noqa: E402
    build_case,
    load_manifest,
)
from moebius_finetune.training.student.prompt import build_masked_prompt  # noqa: E402

LATENT_SIZE = 64
RGB_SIZE = 512
NUM_EMBEDDINGS = 20
NUM_HOLDOUT_IMAGES = 8
HOLDOUT_SEED = 7
NOISE_SEED = 12345
TAESDXL_NOISE_KEY = "taesd_global"


def _holdout_cases() -> List:
    """Replicate scripts/eval_moebius_small._holdout exactly."""
    cases_all = load_manifest("/home/dog/datasets/moebius_finetune/grt_manifest.json")
    cases_hold = [c for c in cases_all if c.split == "holdout_images"]
    hold_imgs = sorted({c.source_id for c in cases_hold})
    rng = np.random.default_rng(HOLDOUT_SEED)
    sel = [hold_imgs[i] for i in rng.permutation(len(hold_imgs))[:NUM_HOLDOUT_IMAGES]]
    return sorted([c for c in cases_hold if c.source_id in sel], key=lambda c: c.case_id)


def _zipdepth_predictor():
    """Lazily-loaded ZipDepth predictor. Mirrors eval_moebius_small._build_cases."""
    from moebius_finetune.data.grt_dataset import ZipDepthOnline
    predictor = ZipDepthOnline(device="cuda")
    return predictor


@torch.no_grad()
def build_real_batches(device, dtype, cases) -> List[Dict[str, torch.Tensor]]:
    """For each case, load COCO RGB, compute disparity online, build x9.

    Returns a list of batch dicts (B=1 each) ready to feed the teacher.
    """
    IMG_DIR = "/home/dog/datasets/image/coco_train2017/train2017"
    cases_by_img: Dict[str, List] = {}
    for c in cases:
        cases_by_img.setdefault(c.source_id, []).append(c)

    predictor = _zipdepth_predictor()
    disp_cache: Dict[str, np.ndarray] = {}
    print(f"  computing disparity online for {len(cases_by_img)} unique images ...")
    t0 = time.time()
    for image_id, cs in cases_by_img.items():
        from PIL import Image
        p = None
        for ext in (".jpg", ".jpeg", ".png"):
            cand = os.path.join(IMG_DIR, f"{image_id}{ext}")
            if os.path.exists(cand):
                p = cand
                break
        if p is None:
            raise FileNotFoundError(f"source image not found for {image_id}")
        with Image.open(p) as im:
            rgb = np.asarray(im.convert("RGB").resize((RGB_SIZE, RGB_SIZE), Image.Resampling.BILINEAR))
        rgb01 = (rgb.astype(np.float32) / 255.0).transpose(2, 0, 1)  # (3, 512, 512)
        bgr_uint8 = (np.clip(rgb01.transpose(1, 2, 0)[..., ::-1] * 255.0, 0, 255)).astype(np.uint8)
        disp = np.asarray(predictor(bgr_uint8), dtype=np.float32)
        disp_cache[image_id] = disp
    print(f"  ZipDepth done in {time.time() - t0:.1f}s")

    taesd = getattr(ptb, TAESDXL_NOISE_KEY)
    batches: List[Dict[str, torch.Tensor]] = []
    gen = torch.Generator(device="cpu").manual_seed(NOISE_SEED)

    # Per-case noise draws in case_id-sorted order (matches eval_moebius_small)
    for c in cases:
        sample = build_case(c, image_dir=IMG_DIR, disp=disp_cache[c.source_id])
        target = torch.from_numpy(sample["target"][None]).to(device, dtype)             # (1,3,512,512)
        rgb_hole = torch.from_numpy(sample["rgb_hole"][None]).to(device, dtype)         # (1,3,512,512)
        hole_mask = torch.from_numpy(sample["hole_mask"][None]).to(device, dtype)       # (1,1,512,512)

        prompt = build_masked_prompt(rgb_hole, hole_mask)                               # mid-gray hole
        target_01 = target.clamp(0, 1)
        prompt_01 = prompt.clamp(0, 1)
        x0_lat = taesd.encoder(target_01)
        masked_lat = taesd.encoder(prompt_01)
        lat_mask = F.interpolate(hole_mask, size=(LATENT_SIZE, LATENT_SIZE), mode="nearest")

        # q_sample with deterministic per-case noise seed (case-id-salted)
        case_salt = int.from_bytes(bytes.fromhex(c.case_id.encode("utf-8").hex()[:8]), "big") % (2**31)
        gen_case = torch.Generator(device="cpu").manual_seed(NOISE_SEED + case_salt)
        t = torch.randint(50, 950, (1,), generator=gen_case).to(device).long()
        noise = torch.randn(1, 4, LATENT_SIZE, LATENT_SIZE, generator=gen_case).to(device, dtype)
        a_bar = torch.sigmoid(-t.float() / 100.0).view(-1, 1, 1, 1).to(device, dtype)
        x_t = a_bar.sqrt() * x0_lat + (1.0 - a_bar).sqrt() * noise
        x9 = torch.cat([x_t, lat_mask, masked_lat], dim=1)
        input_ids = torch.arange(NUM_EMBEDDINGS // 2, device=device).unsqueeze(0).contiguous()

        batches.append({
            "case_id": c.case_id,
            "x9": x9,
            "t": t,
            "input_ids": input_ids,
            "x0_lat": x0_lat,
            "x_t": x_t,
            "lat_mask": lat_mask,
            "masked_lat": masked_lat,
            "noise": noise,
            "hole_ratio": float(sample["hole_ratio"]),
        })

    return batches


def main():
    global taesd_global  # noqa: PLW0603 — set in ptb module
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("FATAL: cuda required")
        sys.exit(2)

    torch.manual_seed(0)
    np.random.seed(0)

    print(f"=== block-level profile Moebius teacher (226M) — REAL COCO 32-case holdout ===")
    print(f"=== device: {torch.cuda.get_device_name(0)}, torch={torch.__version__} ===")
    print()

    print("[setup] loading TAESDXL ...")
    ptb.taesd_global = ptb.build_taesd(device)

    print("[setup] loading Moebius teacher (final.pt) ...")
    teacher = ptb.build_teacher(device)
    teacher_unet = teacher.diff_model
    teacher_unet.eval()
    print(f"        teacher (diff_model) params = {sum(p.numel() for p in teacher_unet.parameters()):,}")

    blocks = ptb.enumerate_block_modules(teacher_unet)
    print(f"        leaf modules enumerated: {len(blocks)}")
    print()

    cases = _holdout_cases()
    print(f"[data] holdout: {len(cases)} cases from {NUM_HOLDOUT_IMAGES} COCO images")
    batches = build_real_batches(device, torch.float32, cases)
    print(f"        done — {len(batches)} batches")
    print()

    # baseline
    print("[baseline] running 32 cases with no hooks ...")
    base_l1 = []
    for b in batches:
        out = ptb.run_one(teacher, b).to(device)
        # In-hole L1 (matches eval_moebius_small._hole_l1)
        m = b["lat_mask"][0, 0] > 0.5
        if not m.any():
            base_l1.append(float("nan"))
            continue
        diff = (out[0] - b["noise"][0]).abs().mean(dim=0)
        base_l1.append(float(diff[m].mean()))
    valid = [x for x in base_l1 if not (x != x)]   # filter nan
    base_mean = float(np.mean(valid))
    base_std = float(np.std(valid))
    print(f"  baseline in-hole ‖pred-noise‖₁ = {base_mean:.4f} ± {base_std:.4f}")
    print()

    # block-level ablation
    print(f"[ablation] zero-ablation per block ({len(blocks)} blocks × {len(batches)} cases) ...")
    block_results: List[Tuple[str, str, float, float]] = []
    t0 = time.time()
    for sname, label, mod in blocks:
        deltas = []
        for b in batches:
            out = ptb.run_ablation_one(teacher, b, mod).to(device)
            m = b["lat_mask"][0, 0] > 0.5
            if not m.any():
                continue
            diff = (out[0] - b["noise"][0]).abs().mean(dim=0)
            l1 = float(diff[m].mean())
            deltas.append(l1 - base_mean)
        block_results.append((sname, label, float(np.mean(deltas)), float(np.std(deltas))))
    print(f"  ablation done in {time.time() - t0:.1f}s")
    print()

    # report
    print("=== Block-level zero-ablation (sorted by Δ in-hole L1, ascending) ===")
    print(f"  baseline: {base_mean:.4f}  (32 real-COCO cases, mid-gray holdout)")
    threshold = 0.01 * max(abs(base_mean), 0.1)
    print(f"  threshold for 'live': ±{threshold:.4f} (1% of baseline)")
    print()
    print(f"  {'stage':8s}  {'block':10s}  {'Δ in-hole L1':>14s}  {'rel Δ':>10s}  {'verdict':<14s}  module-class")
    block_results_sorted = sorted(block_results, key=lambda r: r[2])
    for sname, label, dm, ds in block_results_sorted:
        rel = dm / base_mean if base_mean > 0 else 0.0
        verdict = "REDUNDANT" if abs(dm) < threshold else (
            "LIVE-rev" if dm < 0 else "LIVE")
        idx = next(i for i, (s, l, _) in enumerate(blocks) if s == sname and l == label)
        mod_cls = type(blocks[idx][2]).__name__
        print(f"  {sname:8s}  {label:10s}  {dm:+14.4f}  {rel:+10.3f}  {verdict:<14s}  {mod_cls}")

    print()
    print("=== Reverse-contributing blocks (the ones that HURT when active) ===")
    rev = sorted([r for r in block_results if r[2] < -0.005], key=lambda r: r[2])
    if not rev:
        print("  (none beyond threshold)")
    else:
        for sname, label, dm, ds in rev:
            idx = next(i for i, (s, l, _) in enumerate(blocks) if s == sname and l == label)
            mod_cls = type(blocks[idx][2]).__name__
            print(f"  {sname}/{label:8s}  ΔL1={dm:+.4f}  class={mod_cls}")

    print()
    print("=== Per-stage breakdown (sum of block-level Δ within each stage) ===")
    by_stage: Dict[str, float] = {}
    for sname, label, dm, ds in block_results:
        by_stage[sname] = by_stage.get(sname, 0.0) + dm
    for sname in sorted(by_stage.keys()):
        print(f"  {sname:8s}  Σ ΔL1 = {by_stage[sname]:+.4f}")

    print()
    print("=== End of real-COCO block-level profile ===")


if __name__ == "__main__":
    main()
