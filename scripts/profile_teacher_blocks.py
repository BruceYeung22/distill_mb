"""Block-level profile of the Moebius teacher: zero-ablation per leaf module.

Stage-level profile (scripts/profile_teacher.py) already showed that:
  - up2_out (64^2 x 320) is the dominant in-hole contributor (~95%)
  - up1_out (32^2 x 1280) shows REVERSE contribution at every hole width tested
  - down2_out (16^2 x 1280, bottleneck) is REDUNDANT across all widths

This script zooms in one level deeper: enumerate every DWResnetBlock2D and
MixTransformer2DModel inside each stage and zero-ablate each independently.

Architecture (resolved from moebius.yaml on this box):
  3 down x (2 DWResnetBlock2D + 2 MixTransformer2DModel) = 12 blocks
  0 mid
  3 up   x (3 DWResnetBlock2D + 3 MixTransformer2DModel) = 18 blocks
  Total: 30 leaf modules.

Strictly inside distill/; no checkpoints touched.

Usage:
    cd distill && MOEBIUS_UPSTREAM_DIR=/home/dog/project/moebius_distill/Moebius \\
      .venv/bin/python scripts/profile_teacher_blocks.py
"""

from __future__ import annotations

import sys
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, "/home/dog/project/HAMI/thirdParty/taesd")
sys.path.insert(0, "/home/dog/project/moebius_distill/Moebius")

from taesd import TAESD  # noqa: E402
from removal.v1_2 import build_removal_model  # noqa: E402

LATENT_SIZE = 64
RGB_SIZE = 512
NUM_EMBEDDINGS = 20
NUM_CASES = 10
HOLE_DMAX_PX = 32
CKA_DIM = 1024
SEED = 0
DEFAULT_HOLE_WIDTH_PX = 40


def linear_cka(x: np.ndarray, y: np.ndarray) -> float:
    x = x - x.mean(axis=0, keepdims=True)
    y = y - y.mean(axis=0, keepdims=True)
    ytx = y.T @ x
    norm_num = float(np.linalg.norm(ytx, ord="fro") ** 2)
    norm_xtx = float(np.linalg.norm(x.T @ x, ord="fro"))
    norm_yty = float(np.linalg.norm(y.T @ y, ord="fro"))
    denom = norm_xtx * norm_yty
    if denom <= 0:
        return 0.0
    return norm_num / denom


def features_per_case(feat_tensor: torch.Tensor, target_dim: int = CKA_DIM) -> np.ndarray:
    pooled = feat_tensor.mean(dim=(2, 3))  # (B, C)
    b, c = pooled.shape
    gen = torch.Generator(device="cpu").manual_seed(c)
    proj = torch.randn(c, target_dim, generator=gen)
    proj = proj / proj.norm(dim=0, keepdim=True)
    out = pooled.detach().to("cpu", torch.float32) @ proj
    return out.numpy()


def unwrap(out):
    if hasattr(out, "sample"):
        return out.sample
    if isinstance(out, tuple):
        return out[0]
    return out


def make_zero_hook():
    def hook(_module, _inp, out):
        if isinstance(out, tuple):
            return (torch.zeros_like(out[0]),) + out[1:]
        return torch.zeros_like(out)
    return hook


# ----------------------------- module enumeration -----------------------------


def enumerate_block_modules(teacher_unet) -> List[Tuple[str, str, torch.nn.Module]]:
    """Return list of (stage_name, block_label, module) for every leaf module.

    stage_name:  down0, down1, down2, up0, up1, up2
    block_label:  resnet0 / resnet1 / attn0 / attn1 (down blocks have 2 each;
                 up blocks have 3 each).
    """
    blocks: List[Tuple[str, str, torch.nn.Module]] = []
    for di, db in enumerate(teacher_unet.down_blocks):
        sname = f"down{di}"
        for i, m in enumerate(db.resnets):
            blocks.append((sname, f"resnet{i}", m))
        for i, m in enumerate(db.attentions):
            if m is not None:
                blocks.append((sname, f"attn{i}", m))
    if teacher_unet.mid_block is not None:
        for i, m in enumerate(teacher_unet.mid_block.resnets):
            blocks.append(("mid", f"resnet{i}", m))
        for i, m in enumerate(teacher_unet.mid_block.attentions):
            if m is not None:
                blocks.append(("mid", f"attn{i}", m))
    for ui, ub in enumerate(teacher_unet.up_blocks):
        sname = f"up{ui}"
        for i, m in enumerate(ub.resnets):
            blocks.append((sname, f"resnet{i}", m))
        for i, m in enumerate(ub.attentions):
            if m is not None:
                blocks.append((sname, f"attn{i}", m))
    return blocks


# ----------------------------- teacher / taesd load -----------------------------

taesd_global = None


def build_taesd(device):
    enc = "/home/dog/project/HAMI/thirdParty/taesd/taesdxl_encoder.pth"
    dec = "/home/dog/project/HAMI/thirdParty/taesd/taesdxl_decoder.pth"
    taesd = TAESD(encoder_path=enc, decoder_path=dec)
    taesd.to(device=device, dtype=torch.float32)
    taesd.eval()
    return taesd


def build_teacher(device):
    yaml_path = "/home/dog/project/moebius_distill/Moebius/config/model_cfg/moebius.yaml"
    weights_path = "/home/dog/datasets/moebius_finetune/runs/taesdxl_moebius_ft/final.pt"
    teacher = build_removal_model(config_path=yaml_path, num_embeddings=NUM_EMBEDDINGS)
    state_dict = torch.load(weights_path, map_location="cpu", weights_only=False)
    if "ema_state" in state_dict:
        state_dict = state_dict["ema_state"]
    teacher.load_state_dict(state_dict, strict=False)
    teacher.to(device=device, dtype=torch.float32)
    teacher.eval()
    return teacher


def make_grt_batch(device, dtype, batch_size: int, hole_width_px: int,
                   dmax_px: int = HOLE_DMAX_PX, seed: int = 0) -> Dict[str, torch.Tensor]:
    gen = torch.Generator(device="cpu").manual_seed(seed)
    targets = torch.randn(batch_size, 3, RGB_SIZE, RGB_SIZE, generator=gen, dtype=torch.float32).clamp_(-1, 1)
    mask = torch.zeros(batch_size, 1, RGB_SIZE, RGB_SIZE, dtype=torch.float32)
    cx = RGB_SIZE // 2
    x0 = max(0, cx - hole_width_px // 2)
    x1 = min(RGB_SIZE, cx + hole_width_px // 2)
    mask[:, :, :, x0:x1] = 1.0
    warped = targets * (1.0 - mask)
    warped_01 = warped.clamp(0, 1)
    target_01 = targets.clamp(0, 1)
    with torch.no_grad():
        x0_lat = taesd_global.encoder(target_01.to(device, dtype))
        masked_lat = taesd_global.encoder(warped_01.to(device, dtype))
    lat_mask = F.interpolate(mask.to(device, dtype), size=(LATENT_SIZE, LATENT_SIZE), mode="nearest")
    t = torch.randint(50, 950, (batch_size,), generator=gen).to(device).long()
    noise = torch.randn(batch_size, 4, LATENT_SIZE, LATENT_SIZE, generator=gen).to(device, dtype)
    a_bar = torch.sigmoid(-t.float() / 100.0).view(-1, 1, 1, 1).to(device, dtype)
    x_t = a_bar.sqrt() * x0_lat + (1.0 - a_bar).sqrt() * noise
    x9 = torch.cat([x_t, lat_mask, masked_lat], dim=1)
    input_ids = torch.arange(NUM_EMBEDDINGS // 2, device=device).unsqueeze(0).expand(batch_size, -1).contiguous()
    return {"x9": x9, "t": t, "input_ids": input_ids, "x0_lat": x0_lat,
            "x_t": x_t, "lat_mask": lat_mask, "masked_lat": masked_lat, "noise": noise}


def hole_l1_vs_noise(pred_noise: torch.Tensor, noise: torch.Tensor, lat_mask: torch.Tensor) -> float:
    return float((pred_noise[0] - noise[0]).abs().mean())


@torch.no_grad()
def run_one(teacher_full, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
    out = teacher_full(batch["x9"], batch["t"], batch["input_ids"])
    return unwrap(out)


@torch.no_grad()
def run_ablation_one(teacher_full, batch: Dict[str, torch.Tensor],
                     target_module: torch.nn.Module) -> torch.Tensor:
    h = target_module.register_forward_hook(make_zero_hook())
    try:
        out = teacher_full(batch["x9"], batch["t"], batch["input_ids"])
        out = unwrap(out)
    finally:
        h.remove()
    return out


# ----------------------------- main -----------------------------


def main():
    global taesd_global
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--hole-width-px", type=int, default=DEFAULT_HOLE_WIDTH_PX)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("FATAL: cuda required")
        sys.exit(2)

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    print(f"=== block-level profile Moebius teacher (226M) — width={args.hole_width_px} / length=512 ===")
    print(f"=== device: {torch.cuda.get_device_name(0)}, torch={torch.__version__} ===")
    print()

    print("[setup] loading TAESDXL ...")
    taesd_global = build_taesd(device)

    print("[setup] loading Moebius teacher (final.pt) ...")
    teacher = build_teacher(device)
    teacher_unet = teacher.diff_model
    teacher_unet.eval()
    n_params = sum(p.numel() for p in teacher_unet.parameters())
    print(f"        teacher (diff_model) params = {n_params:,}")

    blocks = enumerate_block_modules(teacher_unet)
    print(f"        leaf modules enumerated: {len(blocks)}")
    for sname in sorted({b[0] for b in blocks}):
        cnt = sum(1 for b in blocks if b[0] == sname)
        kinds = {}
        for _, label, m in [b for b in blocks if b[0] == sname]:
            kinds[label] = kinds.get(label, 0) + 1
        print(f"          {sname}: {cnt} blocks ({kinds})")
    print()

    print(f"[data] building {NUM_CASES} GRT cases (hole_width≈{args.hole_width_px}px @ {RGB_SIZE})")
    batches = [make_grt_batch(device, torch.float32, 1, args.hole_width_px, HOLE_DMAX_PX, SEED + i)
               for i in range(NUM_CASES)]
    print(f"        done")
    print()

    # baseline
    print("[baseline] running 10 cases with no hooks ...")
    base_noise_l1 = []
    for b in batches:
        out = run_one(teacher, b).to(device)
        base_noise_l1.append(hole_l1_vs_noise(out, b["noise"], b["lat_mask"]))
    base_mean = float(np.mean(base_noise_l1))
    base_std = float(np.std(base_noise_l1))
    print(f"  baseline in-hole ‖pred-noise‖₁ = {base_mean:.4f} ± {base_std:.4f}")
    print()

    # block-level ablation
    print(f"[ablation] zero-ablation per block ({len(blocks)} blocks × {NUM_CASES} cases) ...")
    block_results: List[Tuple[str, str, float, float]] = []  # (stage, label, delta_mean, delta_std)
    for sname, label, mod in blocks:
        deltas = []
        for b in batches:
            out = run_ablation_one(teacher, b, mod).to(device)
            n = hole_l1_vs_noise(out, b["noise"], b["lat_mask"])
            deltas.append(n - base_mean)
        block_results.append((sname, label, float(np.mean(deltas)), float(np.std(deltas))))

    # report
    print()
    print("=== Block-level zero-ablation (sorted by Δ in-hole L1, ascending) ===")
    print(f"  baseline: {base_mean:.4f}")
    print(f"  threshold for 'live': ±{0.01 * max(abs(base_mean), 0.1):.4f} (1% of baseline)")
    print()
    print(f"  {'stage':8s}  {'block':10s}  {'Δ in-hole L1':>14s}  {'rel Δ':>10s}  {'verdict':<14s}  module-class")
    block_results_sorted = sorted(block_results, key=lambda r: r[2])
    for sname, label, dm, ds in block_results_sorted:
        rel = dm / base_mean if base_mean > 0 else 0.0
        verdict = "REDUNDANT" if abs(dm) < 0.01 * max(abs(base_mean), 0.1) else (
            "LIVE-rev" if dm < 0 else "LIVE")
        # module class name
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
        print(f"  {sname:8s}  Σ ΔL1 = {by_stage[sname]:+.4f}   (note: blocks compound non-linearly; "
              f"this is a sanity sum, not a causal claim)")

    print()
    print("=== End of block-level profile ===")


if __name__ == "__main__":
    main()
