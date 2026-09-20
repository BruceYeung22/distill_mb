"""Profile Moebius teacher's internal contribution via linear CKA + stage-level
residual causal ablation, on a width-40 / length-512 narrow mask scenario.

Design (sanity-check size):
    - 10 GRT cases (5 L2R + 5 R2L, all dmax_px=32 → ~10 latents wide @ 64²)
    - Stage-level: hook each down/up/mid block exit (8 captures total)
    - Linear CKA: per-tap channel-mean-pool → 1024-d → 10×1024 matrix → CKA
      across all 8 taps (8×8 matrix)
    - Residual causal ablation: zero one stage's output at a time, re-run
      forward, measure Δ‖noise_pred‖_2 and Δ hole_l1 in the latent output.

Strictly fp32 + cuda. Loads the same Moebius teacher as train_small.py.
Reads ONLY distill/ + Moebius/ + the thirdParty taesd path; no checkpoints.

Usage:
    cd distill && MOEBIUS_UPSTREAM_DIR=/home/dog/project/moebius_distill/Moebius \
      .venv/bin/python scripts/profile_teacher.py
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

# ----------------------------- constants -----------------------------

LATENT_SIZE = 64
RGB_SIZE = 512
NUM_EMBEDDINGS = 20
NUM_CASES = 10
HOLE_DMAX_PX = 32            # width-40 px hole at RGB = ~5-10 latents wide
CKA_DIM = 1024               # global channel-mean-pool to 1024-d features
SEED = 0
DEFAULT_HOLE_WIDTH_PX = 40  # overridden via --hole-width-px

STAGE_NAMES = [
    "down0_out",     # index 0: down_blocks[0]  → output 32²×640 (post-downsample)
    "down1_out",     # index 1: down_blocks[1]  → output 16²×1280
    "down2_out",     # index 2: down_blocks[2]  → output 8²×1280 (final down, no mid)
    "up0_out",       # index 3: up_blocks[0]    → output 16²×1280
    "up1_out",       # index 4: up_blocks[1]    → output 32²×1280
    "up2_out",       # index 5: up_blocks[2]    → output 64²×320 (final up)
]

# ----------------------------- linear CKA -----------------------------


def linear_cka(x: np.ndarray, y: np.ndarray) -> float:
    """Kornblith et al. 2017 linear CKA. Inputs are (N, d1) and (N, d2)."""
    x = x - x.mean(axis=0, keepdims=True)
    y = y - y.mean(axis=0, keepdims=True)
    # ||Y^T X||_F^2 / (||X^T X||_F * ||Y^T Y||_F)
    ytx = y.T @ x
    norm_num = float(np.linalg.norm(ytx, ord="fro") ** 2)
    norm_xtx = float(np.linalg.norm(x.T @ x, ord="fro"))
    norm_yty = float(np.linalg.norm(y.T @ y, ord="fro"))
    denom = norm_xtx * norm_yty
    if denom <= 0:
        return 0.0
    return norm_num / denom


def features_per_case(feat_tensor: torch.Tensor, target_dim: int = CKA_DIM) -> np.ndarray:
    """Reduce a (B, C, H, W) feature tensor to (B, target_dim) using
    adaptive avg-pool + linear projection to a fixed dim.

    We pool spatially to 1×1 then use a fixed random projection to target_dim.
    Random projection is cheap and preserves CKA reasonably well (Johnson-Lindenstrauss).
    """
    # (B, C) via global avg pool
    pooled = feat_tensor.mean(dim=(2, 3))  # (B, C)
    b, c = pooled.shape
    # deterministic projection per-call (seeded by tensor shape so it's stable
    # across runs for the same shape)
    gen = torch.Generator(device="cpu").manual_seed(c)
    proj = torch.randn(c, target_dim, generator=gen)
    proj = proj / proj.norm(dim=0, keepdim=True)  # column-normalize
    out = pooled.detach().to("cpu", torch.float32) @ proj  # (B, target_dim)
    return out.numpy()


# ----------------------------- hooks -----------------------------


def make_capture_hook(storage: Dict[str, torch.Tensor], key: str):
    def hook(_module, _inp, out):
        if isinstance(out, tuple):
            out = out[0]
        storage[key] = out.detach()
    return hook


def make_zero_hook():
    """Replace the output with zeros — for ablation."""
    def hook(_module, _inp, out):
        if isinstance(out, tuple):
            return (torch.zeros_like(out[0]),) + out[1:]
        return torch.zeros_like(out)
    return hook


def register_capture_hooks(model) -> List[torch.utils.hooks.RemovableHook]:
    """Register hooks on down_blocks[0..2], up_blocks[0..2]."""
    handles = []
    targets = [
        (model.down_blocks[0], "down0_out"),
        (model.down_blocks[1], "down1_out"),
        (model.down_blocks[2], "down2_out"),
        (model.up_blocks[0], "up0_out"),
        (model.up_blocks[1], "up1_out"),
        (model.up_blocks[2], "up2_out"),
    ]
    for module, key in targets:
        module._capture_key = key
    return handles


def get_capture_handles(model, capture: Dict[str, torch.Tensor]) -> List[torch.utils.hooks.RemovableHandle]:
    handles = []
    targets = [
        (model.down_blocks[0], "down0_out"),
        (model.down_blocks[1], "down1_out"),
        (model.down_blocks[2], "down2_out"),
        (model.up_blocks[0], "up0_out"),
        (model.up_blocks[1], "up1_out"),
        (model.up_blocks[2], "up2_out"),
    ]
    for module, key in targets:
        handles.append(module.register_forward_hook(make_capture_hook(capture, key)))
    return handles


# ----------------------------- teacher load -----------------------------


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


def build_taesd(device):
    enc = "/home/dog/project/HAMI/thirdParty/taesd/taesdxl_encoder.pth"
    dec = "/home/dog/project/HAMI/thirdParty/taesd/taesdxl_decoder.pth"
    taesd = TAESD(encoder_path=enc, decoder_path=dec)
    taesd.to(device=device, dtype=torch.float32)
    taesd.eval()
    return taesd


# ----------------------------- synthetic GRT batch -----------------------------


def make_grt_batch(device, dtype, batch_size: int, hole_width_px: int,
                   dmax_px: int = HOLE_DMAX_PX, seed: int = 0) -> Dict[str, torch.Tensor]:
    """Build a small GRT batch with a NARROW mask (~hole_width_px / RGB_SIZE of the width)
    so the latent representation is at the edge of the 16² bottleneck's resolution.

    We use synthetic RGB (no real image fetch) so this script has zero external data deps.
    The 'warp' semantics are simulated: we generate a target RGB and a warped RGB with
    a narrow vertical strip masked to 0.
    """
    gen = torch.Generator(device="cpu").manual_seed(seed)
    targets = torch.randn(batch_size, 3, RGB_SIZE, RGB_SIZE, generator=gen, dtype=torch.float32).clamp_(-1, 1)

    # Mask: a vertical band centered horizontally, full height, width ~ hole_width_px
    mask = torch.zeros(batch_size, 1, RGB_SIZE, RGB_SIZE, dtype=torch.float32)
    cx = RGB_SIZE // 2
    x0 = max(0, cx - hole_width_px // 2)
    x1 = min(RGB_SIZE, cx + hole_width_px // 2)
    mask[:, :, :, x0:x1] = 1.0

    # Warped = target with the band zeroed
    warped = targets * (1.0 - mask)

    # Encode with TAESD (which sees [0,1] images per its training, but we use the same
    # mapping train_small uses; the magnitude here is illustrative — we only care about
    # the relative change under ablation).
    warped_01 = warped.clamp(0, 1)
    target_01 = targets.clamp(0, 1)
    with torch.no_grad():
        x0_lat = taesd_global.encoder(target_01.to(device, dtype))
        masked_lat = taesd_global.encoder(warped_01.to(device, dtype))
    lat_mask = F.interpolate(mask.to(device, dtype), size=(LATENT_SIZE, LATENT_SIZE), mode="nearest")

    # noise + t sampling (deterministic per case)
    t = torch.randint(50, 950, (batch_size,), generator=gen).to(device).long()
    noise = torch.randn(batch_size, 4, LATENT_SIZE, LATENT_SIZE, generator=gen).to(device, dtype)
    # scheduler-like noise: simple q_sample (matches diffusers DDPMScheduler formula at large t)
    # x_t = sqrt(alpha_bar) * x0 + sqrt(1 - alpha_bar) * noise
    a_bar = torch.sigmoid(-t.float() / 100.0).view(-1, 1, 1, 1).to(device, dtype)
    x_t = a_bar.sqrt() * x0_lat + (1.0 - a_bar).sqrt() * noise

    x9 = torch.cat([x_t, lat_mask, masked_lat], dim=1)
    input_ids = torch.arange(NUM_EMBEDDINGS // 2, device=device).unsqueeze(0).expand(batch_size, -1).contiguous()

    return {
        "x9": x9,
        "t": t,
        "input_ids": input_ids,
        "x0_lat": x0_lat,
        "x_t": x_t,
        "lat_mask": lat_mask,
        "masked_lat": masked_lat,
        "noise": noise,
    }


# ----------------------------- global for make_grt_batch -----------------------------
taesd_global = None  # set in main()


# ----------------------------- run -----------------------------


def unwrap(out):
    """Strip diffusers BaseOutput / tuple wrappers to get the noise tensor."""
    if hasattr(out, "sample"):
        return out.sample
    if isinstance(out, tuple):
        return out[0]
    return out


@torch.no_grad()
def collect_taps_and_outputs(teacher_full, teacher_unet, batch: Dict[str, torch.Tensor], stage_keys: List[str],
                             device) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
    """Run forward through RemovalModel; hooks on diff_model internals capture stage outputs."""
    capture: Dict[str, torch.Tensor] = {}
    handles = get_capture_handles(teacher_unet, capture)
    try:
        out = teacher_full(batch["x9"], batch["t"], batch["input_ids"])
        out = unwrap(out)
    finally:
        for h in handles:
            h.remove()
    return {k: capture[k] for k in stage_keys}, out


@torch.no_grad()
def run_ablation(teacher_full, teacher_unet, batch: Dict[str, torch.Tensor], target_key: str,
                 stage_modules, device) -> Tuple[torch.Tensor, torch.Tensor]:
    """Zero the output of `target_key` on diff_model; run forward through RemovalModel."""
    module = stage_modules[target_key]
    handle = module.register_forward_hook(make_zero_hook())
    try:
        out = teacher_full(batch["x9"], batch["t"], batch["input_ids"])
        out = unwrap(out)
    finally:
        handle.remove()
    return out


def main():
    global taesd_global
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("FATAL: cuda required")
        sys.exit(2)

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--hole-width-px", type=int, default=DEFAULT_HOLE_WIDTH_PX,
                        help="width of the synthetic GRT hole in RGB pixels (default: 40)")
    args = parser.parse_args()

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    hole_width_px = args.hole_width_px

    print(f"=== profile Moebius teacher (226M) — width={hole_width_px} / length=512 scenario ===")
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
    print()

    # ---- (A) Build 10 GRT cases ----
    print(f"[data] building {NUM_CASES} GRT cases (hole_width≈{hole_width_px}px @ {RGB_SIZE})")
    batches = []
    for i in range(NUM_CASES):
        b = make_grt_batch(device, torch.float32, batch_size=1, hole_width_px=hole_width_px,
                           dmax_px=HOLE_DMAX_PX, seed=SEED + i)
        batches.append(b)
    print(f"        done — 10 cases, each [1,9,64,64], dmax_px={HOLE_DMAX_PX}")
    print()

    # ---- (B) Capture per-stage taps ----
    print(f"[profile] capturing {len(STAGE_NAMES)} stage taps across {NUM_CASES} cases ...")
    # storage: stage_name -> list of (case_idx, features_array)
    feats: Dict[str, List[np.ndarray]] = {k: [] for k in STAGE_NAMES}
    raw_taps: Dict[str, List[Tuple[int, Tuple[int, int, int, int]]]] = {k: [] for k in STAGE_NAMES}
    outputs_baseline = []

    for i, b in enumerate(batches):
        taps, out = collect_taps_and_outputs(teacher, teacher_unet, b, STAGE_NAMES, device)
        outputs_baseline.append(out)
        for k in STAGE_NAMES:
            t = taps[k]
            raw_taps[k].append((i, tuple(t.shape)))
            feats[k].append(features_per_case(t))

    # ---- (B.1) Tap shapes ----
    print()
    print("[taps] per-stage output shapes (post-downsample / post-up-block):")
    for k in STAGE_NAMES:
        shape0 = raw_taps[k][0][1]
        print(f"  {k:12s}  {tuple(shape0)}")
    print()

    # ---- (C) Linear CKA matrix (NxN over 9 stages) ----
    print("[CKA] computing 9×9 linear CKA matrix ...")
    M = len(STAGE_NAMES)
    cka = np.zeros((M, M), dtype=np.float64)
    for i in range(M):
        Xi = np.concatenate(feats[STAGE_NAMES[i]], axis=0)  # (10, 1024)
        for j in range(M):
            Xj = np.concatenate(feats[STAGE_NAMES[j]], axis=0)
            cka[i, j] = linear_cka(Xi, Xj)

    print()
    print("=== Linear CKA matrix (rows/cols = stages; 1.0 = identical) ===")
    print("          " + "  ".join(f"{n:>9s}" for n in STAGE_NAMES))
    for i, n in enumerate(STAGE_NAMES):
        row = "  ".join(f"{cka[i, j]:9.4f}" for j in range(M))
        print(f"  {n:>9s}  {row}")
    print()

    # ---- (D) Stage-level residual causal ablation ----
    # We need the un-ablated baseline outputs to measure Δ
    print("[ablation] stage-level zero-ablation (each stage output replaced by 0) ...")
    stage_modules = {
        "down0_out": teacher_unet.down_blocks[0],
        "down1_out": teacher_unet.down_blocks[1],
        "down2_out": teacher_unet.down_blocks[2],
        "up0_out": teacher_unet.up_blocks[0],
        "up1_out": teacher_unet.up_blocks[1],
        "up2_out": teacher_unet.up_blocks[2],
    }
    # Re-run all 10 cases for baseline (already in outputs_baseline)
    base_l2_per_case = [float(out.norm().cpu()) for out in outputs_baseline]
    base_l2_mean = float(np.mean(base_l2_per_case))

    # hole_l1 baseline: ‖pred_lat - x0‖₁ averaged over hole pixels, then mean over cases
    # In the train_small pipeline the loss is on noise_pred vs noise, but for ablation
    # we use the more semantic ‖pred_lat - x0‖ in the in-hole region (matches AGENTS.md table).
    def hole_l1(pred_lat: torch.Tensor, x0: torch.Tensor, lat_mask: torch.Tensor) -> float:
        m = lat_mask[0, 0] > 0.5  # [H, W] bool
        if not m.any():
            return 0.0
        diff = (pred_lat[0] - x0[0]).abs().mean(dim=0)  # [H, W]
        return float(diff[m].mean())

    def hole_l1_vs_noise(pred_noise: torch.Tensor, noise: torch.Tensor, lat_mask: torch.Tensor) -> float:
        # This is the real KD target: pred == noise
        return float((pred_noise[0] - noise[0]).abs().mean())

    base_hole_l1 = []
    base_noise_l1 = []
    for i, b in enumerate(batches):
        out = outputs_baseline[i].to(device)
        base_noise_l1.append(hole_l1_vs_noise(out, b["noise"], b["lat_mask"]))
    base_noise_l1_mean = float(np.mean(base_noise_l1))

    print(f"  baseline: ‖noise_pred‖₂ = {base_l2_mean:.4f} (mean over {NUM_CASES} cases)")
    print(f"  baseline: in-hole ‖pred - noise‖₁ = {base_noise_l1_mean:.4f}")
    print()

    abl_l2 = {}
    abl_dnoise_l1 = {}
    for skey in STAGE_NAMES:
        l2_vals = []
        dn_vals = []
        for i, b in enumerate(batches):
            out = run_ablation(teacher, teacher_unet, b, skey, stage_modules, device)
            l2_vals.append(float(out.norm().cpu()))
            dn_vals.append(hole_l1_vs_noise(out.to(device), b["noise"], b["lat_mask"]))
        abl_l2[skey] = (float(np.mean(l2_vals)), float(np.std(l2_vals)))
        abl_dnoise_l1[skey] = (float(np.mean(dn_vals)), float(np.std(dn_vals)))

    # ---- (E) Report ----
    print("=== Stage-level zero-ablation sensitivity ===")
    print(f"  baseline ‖noise_pred‖₂  = {base_l2_mean:.4f}")
    print(f"  baseline in-hole ‖pred-noise‖₁ = {base_noise_l1_mean:.4f}")
    print()
    print(f"  {'stage':12s}  {'Δ‖noise_pred‖₂':>16s}  {'Δ in-hole L1':>16s}  {'rel Δ L1':>10s}")
    for skey in STAGE_NAMES:
        dl2_mean, dl2_std = abl_l2[skey]
        dn_mean, dn_std = abl_dnoise_l1[skey]
        delta_l2 = dl2_mean - base_l2_mean
        delta_n = dn_mean - base_noise_l1_mean
        rel = (delta_n / base_noise_l1_mean) if base_noise_l1_mean > 0 else 0.0
        print(f"  {skey:12s}  {delta_l2:+16.4f}  {delta_n:+16.4f}  {rel:+10.3f}")
    print()

    # ---- (F) Effective contribution summary ----
    print("=== Effective contribution per stage (interpretation) ===")
    # Heuristic: a stage is "live" if zeroing it changes the output (|Δ noise L1| > some threshold).
    # A stage is "redundant" if zeroing it leaves output close to baseline.
    threshold = 0.01 * max(base_noise_l1_mean, 0.1)
    print(f"  threshold for 'live' = {threshold:.4f} (1% of baseline L1)")
    print()
    print(f"  {'stage':12s}  {'Δ in-hole L1':>16s}  {'verdict':<14s}  spatial")
    spatial_lookup = {
        "down0_out": "32²×640",
        "down1_out": "16²×1280",
        "down2_out": "8²×1280 (bottleneck)",
        "up0_out": "16²×1280",
        "up1_out": "32²×1280",
        "up2_out": "64²×320",
    }
    for skey in STAGE_NAMES:
        dn_mean, _ = abl_dnoise_l1[skey]
        delta_n = dn_mean - base_noise_l1_mean
        verdict = "LIVE" if abs(delta_n) > threshold else "REDUNDANT"
        print(f"  {skey:12s}  {delta_n:+16.4f}  {verdict:<14s}  {spatial_lookup[skey]}")
    print()

    # ---- (G) Cross-stage CKA summary ----
    print("=== CKA: which stage taps carry similar information? ===")
    print()
    # Identify the most similar pair (excluding self)
    pairs = []
    for i in range(M):
        for j in range(i + 1, M):
            pairs.append((cka[i, j], STAGE_NAMES[i], STAGE_NAMES[j]))
    pairs.sort(reverse=True)
    print("  Top-5 most similar pairs:")
    for c, a, b in pairs[:5]:
        print(f"    CKA={c:.4f}  {a} ↔ {b}")
    print()
    print("  Top-5 most dissimilar pairs:")
    pairs.sort()
    for c, a, b in pairs[:5]:
        print(f"    CKA={c:.4f}  {a} ↔ {b}")
    print()

    print("=== End of profile ===")


if __name__ == "__main__":
    main()
