"""Block-level linear CKA on real COCO 32-case holdout, plus
cross-reference against the block-level zero-ablation deltas.

Closes the open question in docs/teacher-profile-v1.md sec 7
(block-level CKA was pending). Specifically tests:

  - Do the 6 reverse-contributing up1 blocks have HIGH CKA to the
    load-bearing up2/resnet2 — implying they re-encode the same
    feature poorly (redundancy-in-noise)?
  - Do the down1 / down2 / up0 blocks (newly reverse on real data)
    have HIGH CKA to up2 too — same mechanism, different stage?
  - Which blocks form a tight cluster by CKA — the
    "encoding-decoding equivalence" pattern of residual streams?

Strictly inside distill/; no checkpoints touched.

Reuses:
  - scripts/profile_teacher_blocks.py for enumerate_block_modules
    (same sys.path-injection pattern as profile_teacher_real.py)
  - scripts/profile_teacher_real.py for _holdout_cases + build_real_batches

Usage:
    cd distill && MOEBIUS_UPSTREAM_DIR=/home/dog/project/moebius_distill/Moebius \\
      .venv/bin/python scripts/profile_teacher_real_cka.py
"""

from __future__ import annotations

import os
import sys
from typing import Dict, List

import numpy as np
import torch

sys.path.insert(0, "/home/dog/project/HAMI/thirdParty/taesd")
sys.path.insert(0, "/home/dog/project/moebius_distill/Moebius")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import profile_teacher_blocks as ptb
import profile_teacher_real as ptr

# Block-level ablation deltas from scripts/profile_teacher_real.py
# (real COCO 32-case holdout, baseline in-hole L1 = 0.2514).
# Ordered to match enumerate_block_modules' output (down first, then up).
BLOCK_LABELS_ORDERED: List[str] = [
    # down0 (2 resnet then 2 attn, per enumerate_block_modules order)
    "down0/resnet0", "down0/resnet1",
    "down0/attn0",   "down0/attn1",
    # down1
    "down1/resnet0", "down1/resnet1",
    "down1/attn0",   "down1/attn1",
    # down2
    "down2/resnet0", "down2/resnet1",
    "down2/attn0",   "down2/attn1",
    # up0 (3 resnet then 3 attn)
    "up0/resnet0", "up0/resnet1", "up0/resnet2",
    "up0/attn0",   "up0/attn1",   "up0/attn2",
    # up1
    "up1/resnet0", "up1/resnet1", "up1/resnet2",
    "up1/attn0",   "up1/attn1",   "up1/attn2",
    # up2
    "up2/resnet0", "up2/resnet1", "up2/resnet2",
    "up2/attn0",   "up2/attn1",   "up2/attn2",
]

ABLATION_DELTAS = {
    "up1/resnet1": -0.0944, "up1/resnet2": -0.0633, "up1/attn2": -0.0533,
    "up0/resnet2": -0.0427, "up1/attn0": -0.0370, "down1/attn1": -0.0352,
    "down2/resnet0": -0.0330, "down2/attn0": -0.0293, "up1/resnet0": -0.0274,
    "up0/attn1": -0.0254, "up1/attn1": -0.0236, "down1/attn0": -0.0183,
    "up2/resnet0": -0.0180, "up2/attn0": -0.0119, "up0/resnet1": -0.0103,
    "down0/resnet1": -0.0047, "down1/resnet1": -0.0035,
    "up0/attn0": -0.0025, "up0/resnet0": -0.0019, "down2/resnet1": -0.0015,
    "down2/attn1": -0.0011, "up0/attn2": +0.0015,
    "down1/resnet0": +0.0170, "down0/resnet0": +0.0359,
    "down0/attn1": +0.0423, "down0/attn0": +0.0654,
    "up2/attn1": +0.1538, "up2/resnet1": +0.3786,
    "up2/attn2": +0.5565, "up2/resnet2": +0.6331,
}

CKA_DIM = 1024


def features_per_case(feat_tensor: torch.Tensor, target_dim: int = CKA_DIM) -> np.ndarray:
    """Global avg-pool + random projection to fixed dim (stable per C)."""
    pooled = feat_tensor.mean(dim=(2, 3))  # (B, C)
    b, c = pooled.shape
    gen = torch.Generator(device="cpu").manual_seed(c)
    proj = torch.randn(c, target_dim, generator=gen)
    proj = proj / proj.norm(dim=0, keepdim=True)
    out = pooled.detach().to("cpu", torch.float32) @ proj
    return out.numpy()


def linear_cka(x: np.ndarray, y: np.ndarray) -> float:
    x = x - x.mean(axis=0, keepdims=True)
    y = y - y.mean(axis=0, keepdims=True)
    ytx = y.T @ x
    norm_num = float(np.linalg.norm(ytx, ord="fro") ** 2)
    norm_xtx = float(np.linalg.norm(x.T @ x, ord="fro"))
    norm_yty = float(np.linalg.norm(y.T @ y, ord="fro"))
    denom = norm_xtx * norm_yty
    return norm_num / denom if denom > 0 else 0.0


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("FATAL: cuda required")
        sys.exit(2)

    torch.manual_seed(0)
    np.random.seed(0)

    print(f"=== block-level linear CKA on real COCO 32-case holdout ===")
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

    # Sanity: ordered labels must match
    actual_labels = [f"{s}/{l}" for s, l, _ in blocks]
    assert actual_labels == BLOCK_LABELS_ORDERED, (
        f"block order mismatch.\nexpected: {BLOCK_LABELS_ORDERED[:4]}\n"
        f"actual:   {actual_labels[:4]}"
    )

    cases = ptr._holdout_cases()
    print(f"[data] holdout: {len(cases)} cases")
    batches = ptr.build_real_batches(device, torch.float32, cases)
    print(f"        done — {len(batches)} batches")
    print()

    # Capture: per case, per block -> (C, H, W) feature tensor
    print(f"[CKA] capturing block features across {len(batches)} cases x {len(blocks)} blocks ...")
    storage: Dict[str, List[torch.Tensor]] = {label: [] for label in actual_labels}
    handles = []
    for (_, label, mod), key in zip(blocks, actual_labels):
        handles.append(mod.register_forward_hook(_capture_cb(storage, key)))
    try:
        for b in batches:
            _ = ptr.ptb.run_one(teacher, b)
    finally:
        for h in handles:
            h.remove()

    # Build per-block (N_cases, CKA_DIM) feature matrix
    block_feats: Dict[str, np.ndarray] = {}
    for label in actual_labels:
        feats_per_case = [features_per_case(t) for t in storage[label]]
        # each is (1, 1024); concatenate to (N_cases, 1024)
        block_feats[label] = np.concatenate(feats_per_case, axis=0)
    print(f"        done")
    print()

    # 30x30 CKA matrix
    M = len(actual_labels)
    cka = np.zeros((M, M), dtype=np.float64)
    for i, a in enumerate(actual_labels):
        for j, b in enumerate(actual_labels):
            cka[i, j] = linear_cka(block_feats[a], block_feats[b])

    # Anchor: up2/resnet2 (the load-bearing block, +0.6331 in ablation)
    anchor = "up2/resnet2"
    anchor_idx = actual_labels.index(anchor)
    cka_to_anchor = cka[:, anchor_idx]

    # Print CKA matrix
    short_names = [_short(label) for label in actual_labels]
    print("=== Block-level linear CKA matrix (real COCO, 32 cases) ===")
    print()
    # Header
    print("          " + "  ".join(f"{n:>6s}" for n in short_names))
    for i, n in enumerate(short_names):
        row = "  ".join(f"{cka[i, j]:6.3f}" for j in range(M))
        delta = ABLATION_DELTAS[actual_labels[i]]
        delta_pct = delta / 0.2514 * 100
        print(f"  {n:>9s}  {row}  ΔL1={delta:+.4f} ({delta_pct:+.1f}%)")
    print()

    # Correlate CKA to anchor with ablation delta
    print(f"=== CKA to load-bearing anchor '{anchor}' (ΔL1=+0.6331) vs ablation ΔL1 ===")
    deltas = np.array([ABLATION_DELTAS[l] for l in actual_labels])
    cka_arr = cka_to_anchor
    # rank correlation (Spearman) - more robust than Pearson
    from scipy.stats import spearmanr
    rho, p = spearmanr(cka_arr, deltas)
    print(f"  Spearman ρ(CKA_to_anchor, ΔL1) = {rho:+.4f}  (p={p:.4f})")
    print(f"  Higher CKA to {anchor} -> blocks are encoding similar info as the load-bearing block.")
    print()
    print(f"  {'block':16s}  {'CKA to ' + anchor:>20s}  {'ΔL1':>10s}  {'verdict':<20s}")
    sorted_by_cka = sorted(zip(actual_labels, cka_arr, deltas), key=lambda t: -t[1])
    for label, c, d in sorted_by_cka[:10]:
        verdict = "REDUNDANT with anchor?" if (c > 0.95 and d < 0) else (
            "co-load-bearing" if (c > 0.95 and d > 0) else "distinct")
        print(f"  {label:16s}  {c:>20.4f}  {d:>+10.4f}  {verdict}")
    print()
    print(f"  --- top 10 LOWEST CKA to {anchor} ---")
    for label, c, d in sorted_by_cka[-10:]:
        verdict = "highly distinct"
        print(f"  {label:16s}  {c:>20.4f}  {d:>+10.4f}  {verdict}")
    print()

    # Specific test: do up1 blocks cluster with up2/resnet2?
    print(f"=== up1 cluster test: do all 6 up1 blocks encode similar info to up2/resnet2? ===")
    up1_labels = [l for l in actual_labels if l.startswith("up1/")]
    print(f"  up1 labels: {up1_labels}")
    print()
    print(f"  {'up1 block':16s}  {'CKA to up2/resnet2':>20s}  {'ΔL1':>10s}")
    for label in up1_labels:
        c = cka[actual_labels.index(label), anchor_idx]
        d = ABLATION_DELTAS[label]
        print(f"  {label:16s}  {c:>20.4f}  {d:>+10.4f}")
    up1_mean = float(np.mean([cka[actual_labels.index(l), anchor_idx] for l in up1_labels]))
    print(f"\n  up1 mean CKA to {anchor}: {up1_mean:.4f}")
    print(f"  (if high, up1 blocks re-encode up2's feature poorly -> dropping them is justified)")
    print()

    # Find the most clustered block groups (off-diagonal CKA >= 0.95)
    print("=== Tight block clusters (off-diagonal CKA >= 0.95) ===")
    edges = []
    for i in range(M):
        for j in range(i + 1, M):
            if cka[i, j] >= 0.95:
                edges.append((cka[i, j], actual_labels[i], actual_labels[j]))
    edges.sort(reverse=True)
    if not edges:
        print("  (none)")
    else:
        for c, a, b in edges[:15]:
            print(f"  CKA={c:.4f}  {a} <-> {b}")
    print()

    print("=== End of block-level CKA + cross-ablation ===")


def _capture_cb(storage, key):
    def hook(_module, _inp, out):
        if isinstance(out, tuple):
            out = out[0]
        storage[key].append(out.detach())
    return hook


def _short(label: str) -> str:
    """Compact label for matrix display, e.g. 'down0/r0' instead of 'down0/resnet0'."""
    stage, blk = label.split("/")
    short_stage = stage.replace("down", "d").replace("up", "u")
    short_blk = blk.replace("resnet", "r").replace("attn", "a")
    return f"{short_stage}/{short_blk}"


if __name__ == "__main__":
    main()
