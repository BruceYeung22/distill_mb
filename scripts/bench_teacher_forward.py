"""Benchmark TAESDXL + Moebius teacher forward latency on the DGX Spark (GB10).

Goal: measure the per-step forward cost of the training loop's teacher side.
We benchmark four pieces in isolation:

    1. TAESDXL encoder (one call on a [B, 3, 512, 512] RGB prompt)
    2. TAESDXL encoder + mask interpolation (the _to_latents step)
    3. Moebius teacher forward (CFG 1.0, single cond batch [B, 10])
       - 3 input channels for the prompt 9ch input actually NOT used by
         teacher; teacher takes [B, 4, 64, 64] latent + timesteps + input_ids
    4. Combined: encode (target + prompt) + nearest-mask + teacher forward

We compare eager mode against torch.compile(mode="reduce-overhead") which
uses CUDA graph capture.

Strictly fp32 + cuda. The script does NOT touch training or checkpoints.

Usage:
    cd distill && MOEBIUS_UPSTREAM_DIR=/home/dog/project/moebius_distill/Moebius \
      .venv/bin/python scripts/bench_teacher_forward.py
"""

from __future__ import annotations

import argparse
import os
import statistics
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, "/home/dog/project/HAMI/thirdParty/taesd")
sys.path.insert(0, "/home/dog/project/moebius_distill/Moebius")

from taesd import TAESD  # noqa: E402
from removal.v1_2 import build_removal_model  # noqa: E402

LATENT_SIZE = 64
RGB_SIZE = 512
NUM_EMBEDDINGS = 20  # teacher CFG=1.0 uses input_ids = torch.arange(10).unsqueeze(0)


def cuda_time(fn, warmup: int = 10, iters: int = 30):
    """Run `fn` `warmup+iters` times; return per-iter ms (median, mean, std)."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        starts[i].record()
        fn()
        ends[i].record()
    torch.cuda.synchronize()
    times_ms = [s.elapsed_time(e) for s, e in zip(starts, ends)]
    return times_ms


def summarise(label, times_ms):
    times_ms = sorted(times_ms)
    med = times_ms[len(times_ms) // 2]
    mean = statistics.fmean(times_ms)
    std = statistics.pstdev(times_ms)
    p10 = times_ms[int(len(times_ms) * 0.10)]
    p90 = times_ms[int(len(times_ms) * 0.90)]
    print(
        f"  {label:40s}  median={med:7.2f} ms  mean={mean:7.2f}  "
        f"std={std:5.2f}  p10={p10:7.2f}  p90={p90:7.2f}"
    )
    return med


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


@torch.no_grad()
def bench_taesd_encoder(taesd, device, batch):
    rgb = torch.randn(batch, 3, RGB_SIZE, RGB_SIZE, device=device, dtype=torch.float32)
    fn = lambda: taesd.encoder(rgb)  # noqa: E731
    return fn


@torch.no_grad()
def bench_to_latents(taesd, device, batch):
    """Mirrors train_small._to_latents (encode target + encode prompt + nearest-mask)."""
    target = torch.randn(batch, 3, RGB_SIZE, RGB_SIZE, device=device, dtype=torch.float32)
    prompt = torch.randn(batch, 3, RGB_SIZE, RGB_SIZE, device=device, dtype=torch.float32)
    mask = torch.randn(batch, 1, RGB_SIZE, RGB_SIZE, device=device, dtype=torch.float32).clamp_(0, 1)

    def fn():
        x0 = taesd.encoder(target)
        masked = taesd.encoder(prompt)
        lm = F.interpolate(mask, size=(LATENT_SIZE, LATENT_SIZE), mode="nearest")
        return x0, masked, lm

    return fn


@torch.no_grad()
def bench_teacher_forward(teacher, device, batch):
    """CFG 1.0: single cond batch. teacher takes 9ch prompt input + timesteps + input_ids."""
    x9 = torch.randn(batch, 9, LATENT_SIZE, LATENT_SIZE, device=device, dtype=torch.float32)
    t = torch.randint(0, 1000, (batch,), device=device, dtype=torch.long)
    input_ids = torch.arange(NUM_EMBEDDINGS // 2, device=device).unsqueeze(0).expand(batch, -1).contiguous()

    def fn():
        return teacher(x9, t, input_ids)

    return fn


@torch.no_grad()
def bench_combined(taesd, teacher, device, batch):
    """Encode + mask interp + teacher forward — the real per-step cost."""
    target = torch.randn(batch, 3, RGB_SIZE, RGB_SIZE, device=device, dtype=torch.float32)
    prompt = torch.randn(batch, 3, RGB_SIZE, RGB_SIZE, device=device, dtype=torch.float32)
    mask = torch.randn(batch, 1, RGB_SIZE, RGB_SIZE, device=device, dtype=torch.float32).clamp_(0, 1)
    t = torch.randint(0, 1000, (batch,), device=device, dtype=torch.long)
    input_ids = torch.arange(NUM_EMBEDDINGS // 2, device=device).unsqueeze(0).expand(batch, -1).contiguous()

    def fn():
        x0 = taesd.encoder(target)
        masked = taesd.encoder(prompt)
        lm = F.interpolate(mask, size=(LATENT_SIZE, LATENT_SIZE), mode="nearest")
        x_t = x0
        x9 = torch.cat([x_t, lm, masked], dim=1)
        return teacher(x9, t, input_ids)

    return fn


def run(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("FATAL: cuda not available on this box. Bench aborts.")
        sys.exit(2)

    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")  # matches Moebius teacher init

    print(f"=== Device: {torch.cuda.get_device_name(0)} ===")
    print(f"=== torch={torch.__version__}  cuda={torch.version.cuda} ===")
    print(f"=== dtype=fp32  cudnn.benchmark={torch.backends.cudnn.benchmark} ===")
    print()

    print("[setup] loading TAESDXL ...")
    taesd = build_taesd(device)
    print(f"        taesd.encoder params={sum(p.numel() for p in taesd.encoder.parameters()):,}")
    print()

    print("[setup] loading Moebius teacher (final.pt) ...")
    t0 = time.time()
    teacher = build_teacher(device)
    print(f"        teacher params={sum(p.numel() for p in teacher.parameters()):,}")
    print(f"        teacher load: {time.time()-t0:.1f}s")
    print()

    batch_sizes = [int(b) for b in args.batch_sizes.split(",")]
    skipped = [b for b in batch_sizes if b > args.max_batch]
    batch_sizes = [b for b in batch_sizes if b <= args.max_batch]
    if skipped:
        print(f"[guard] skipping batches above max-batch={args.max_batch}: {skipped}")
        print(f"        (combined TAESD x2 + 226M Moebius OOM at batch=16 on the 121 GB GB10)")
    if not batch_sizes:
        print("FATAL: no batch sizes to run after applying --max-batch")
        sys.exit(2)
    warmup = args.warmup
    iters = args.iters

    results = {}
    for batch in batch_sizes:
        print(f"--- batch={batch} ---")

        fn_enc = bench_taesd_encoder(taesd, device, batch)
        fn_lat = bench_to_latents(taesd, device, batch)
        fn_t = bench_teacher_forward(teacher, device, batch)
        fn_all = bench_combined(taesd, teacher, device, batch)

        print("  [eager]")
        e_enc = summarise("taesd.encoder (single)", cuda_time(fn_enc, warmup, iters))
        e_lat = summarise("taesd x2 + mask interp", cuda_time(fn_lat, warmup, iters))
        e_t = summarise("teacher fwd (CFG 1.0)", cuda_time(fn_t, warmup, iters))
        e_all = summarise("combined per-step cost", cuda_time(fn_all, warmup, iters))

        compiled_results = {}
        if args.compile:
            print("  [torch.compile reduce-overhead]")
            try:
                taesd_enc_c = torch.compile(taesd.encoder, mode="reduce-overhead")
                teacher_c = torch.compile(teacher, mode="reduce-overhead")

                # Wrap the compiled modules into small forwarders for fair timing
                @torch.no_grad()
                def fn_enc_c(rgb):
                    return taesd_enc_c(rgb)

                rgb_warm = torch.randn(batch, 3, RGB_SIZE, RGB_SIZE, device=device, dtype=torch.float32)
                # warm-up the compile (also triggers the actual graph capture)
                fn_enc_c_warm = lambda: fn_enc_c(rgb_warm)  # noqa: E731
                for _ in range(3):
                    fn_enc_c_warm()
                torch.cuda.synchronize()
                c_enc = summarise(
                    "taesd.encoder (single)",
                    cuda_time(fn_enc_c_warm, warmup, iters),
                )

                @torch.no_grad()
                def fn_t_c(x9, t, ids):
                    return teacher_c(x9, t, ids)

                x9_warm = torch.randn(batch, 9, LATENT_SIZE, LATENT_SIZE, device=device, dtype=torch.float32)
                t_warm = torch.randint(0, 1000, (batch,), device=device, dtype=torch.long)
                ids_warm = torch.arange(NUM_EMBEDDINGS // 2, device=device).unsqueeze(0).expand(batch, -1).contiguous()
                fn_t_c_warm = lambda: fn_t_c(x9_warm, t_warm, ids_warm)  # noqa: E731
                for _ in range(3):
                    fn_t_c_warm()
                torch.cuda.synchronize()
                c_t = summarise(
                    "teacher fwd (CFG 1.0)",
                    cuda_time(fn_t_c_warm, warmup, iters),
                )

                compiled_results = {"enc": c_enc, "teacher": c_t}
            except Exception as exc:
                print(f"  compile failed: {exc}")

        results[batch] = {
            "eager": {"enc": e_enc, "lat": e_lat, "teacher": e_t, "combined": e_all},
            "compiled": compiled_results,
        }
        print()

    # Summary table
    print("=== Summary (median ms / iter) ===")
    print(f"{'batch':>6} | {'enc eager':>10} {'enc compiled':>12} | "
          f"{'teacher eager':>14} {'teacher compiled':>17} | {'combined eager':>14}")
    print("-" * 90)
    for batch in batch_sizes:
        r = results[batch]
        e_enc = r["eager"]["enc"]
        c_enc = r["compiled"].get("enc", float("nan"))
        e_t = r["eager"]["teacher"]
        c_t = r["compiled"].get("teacher", float("nan"))
        e_all = r["eager"]["combined"]
        print(
            f"{batch:>6} | {e_enc:>10.2f} {c_enc:>12.2f} | {e_t:>14.2f} {c_t:>17.2f} | {e_all:>14.2f}"
        )
    print()
    print("Notes:")
    print("  - enc: TAESDXL encoder on [B,3,512,512] -> [B,4,64,64]")
    print("  - teacher fwd: CFG 1.0, single cond batch [B, 10] input_ids")
    print("  - combined = 2x taesd encoder + nearest mask + teacher forward")
    print("    (the per-step teacher-side cost in train_small.train)")
    print("  - compiled mode: torch.compile(mode='reduce-overhead'), CUDA graphs")
    print("  - benchmark was run on", torch.cuda.get_device_name(0))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--batch-sizes", type=str, default="1,4,8,16,32",
                   help="comma-separated batch sizes")
    p.add_argument("--max-batch", type=int, default=4,
                   help="hard cap: skip any batch above this to avoid OOM on "
                        "the 121 GB GB10 (combined forward of TAESD + 226M Moebius)")
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--iters", type=int, default=30)
    p.add_argument("--no-compile", dest="compile", action="store_false")
    p.set_defaults(compile=True)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args)
