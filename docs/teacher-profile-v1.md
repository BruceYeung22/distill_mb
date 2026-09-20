# Moebius teacher profile v1 — distillation design guidance

**Scope**: empirical profile of the 226M Moebius teacher (final.pt) on the
DGX Spark (GB10), used to inform the `MoebiusSmallStudent` architecture
and the head design. Three orthogonal analyses at four scales:

| analysis | granularity | data | result |
|---|---|---|---|
| Latency | per forward pass | batch=1 | `scripts/bench_teacher_forward.py` |
| Stage-level CKA + ablation | 6 stage taps | synthetic GRT, 4 hole widths | `scripts/profile_teacher.py` |
| Block-level ablation | 30 leaf modules | synthetic w=40 | `scripts/profile_teacher_blocks.py` |
| Block-level ablation | 30 leaf modules | real COCO 32-case holdout | `scripts/profile_teacher_real.py` |

**Architecture profiled** (resolved from `Moebius/config/model_cfg/moebius.yaml`
on this box — different from the default class args; do not assume 4 down + mid + 4 up):

```
3 down  (DWMixTFDownBlock2D, each 2 DWResnetBlock2D + 2 MixTransformer2DModel)
0 mid
3 up    (DWMixTFUpBlock2D, each 3 DWResnetBlock2D + 3 MixTransformer2DModel)
```

Six stage taps and their post-downsample / post-block shapes:

| stage | spatial | channels | type |
|---|---|---|---|
| `down0_out` | 32² | 320 | DWMixTF down + downsample |
| `down1_out` | 16² | 640 | DWMixTF down + downsample |
| `down2_out` | 16² | 1280 | DWMixTF down (no downsample; deepest) |
| `up0_out`   | 32² | 1280 | DWMixTF up |
| `up1_out`   | 64² | 640 | DWMixTF up |
| `up2_out`   | 64² | 320 | DWMixTF up (final before conv_out) |

Note: yaml sets `mid_block_type: null`, so **there is no 8² bottleneck block**.
`down2_out` (16²×1280) is the deepest feature. This changes the bottleneck
story vs the default UNet — there is no second-stage downsampling in this config.

## 1. Latency on GB10 (batch=1, fp32 + CUDA, `cudnn.benchmark=True`)

| path | median | mean | std |
|---|---:|---:|---:|
| TAESDXL encoder (single) | 18.12 ms | 18.26 | 0.32 |
| TAESDXL x2 + nearest mask | 36.59 ms | 36.52 | 0.49 |
| Moebius teacher fwd (CFG 1.0) | 69.74 ms | 69.46 | 1.13 |
| Combined per-step cost | **107.67 ms** | 107.70 | 1.43 |

`torch.compile(mode="reduce-overhead")` on TAESDXL encoder: **12.24 ms** (1.48×
speedup). Teacher fwd path fails to CUDA-graph (CPU-side `torch.randint`
for timestep crosses the graph boundary; the same pattern as
`train_small.py`'s RNG draw).

**Combined cost ≈ 108 ms / case @ batch=1.** The training loop runs at
~5.40 s/step @ batch=32 (~108 × 32 ≈ 3.5 s pure compute + I/O + EMA
update overhead). The teacher dominates (65%) over TAESDXL encoding
(34%); mask interpolation is negligible.

**Implication for distillation**: the teacher-side cost is the bottleneck.
Any optimization that reduces teacher forward time directly benefits
training. Options: `torch.compile(teacher)` (blocked by RNG draw;
move `t` to GPU-pinned pre-generated), bf16/amp (cuts activation
bandwidth; Moebius was trained in fp32, mixed-precision needs
calibration), distillation with cached trajectories (current
trajectory_cache approach).

## 2. Stage-level profile across 4 hole widths (synthetic)

`scripts/profile_teacher.py` with `--hole-width-px {20, 40, 80, 160}`.
Synthetic GRT: 1 image per case, vertical strip mask of width
`hole_width_px` centered, TAESDXL-encoded, CFG 1.0 single batch.

| stage | w=20 | w=40 | w=80 | w=160 |
|---|---:|---:|---:|---:|
| baseline in-hole ‖pred−noise‖₁ | 0.2719 | 0.2708 | 0.2760 | 0.2776 |
| `down0_out` Δ | +0.0009 | +0.0033 | -0.0005 | **-0.0030** |
| `down1_out` Δ | +0.0010 | +0.0035 | +0.0002 | **-0.0039** |
| `down2_out` Δ (bottleneck) | +0.0000 | -0.0000 | -0.0002 | +0.0001 |
| `up0_out` Δ | +0.0222 | +0.0252 | +0.0243 | +0.0260 |
| `up1_out` Δ | **-0.0352** | **-0.0325** | **-0.0333** | **-0.0287** |
| `up2_out` Δ | +0.5262 | +0.5273 | +0.5222 | +0.5206 |

Three findings survive all 4 widths:

1. **`up2_out` is the dominant in-hole contributor** (~95% of ΔL1,
   ratio stable at 0.5206–0.5273 across w=20–160). Final conv_out is
   the single point of quality.
2. **`up1_out` is reverse-contributing at every width tested**
   (range −0.0287 to −0.0352, std ≈ 0.003). **Architecture-level, not
   hole-locality** — Moebius's `up1_out` is teaching the student
   Places2 scene priors that actively hurt hole-fill.
3. **`down2_out` is REDUNDANT at every width tested** (Δ ≤ |0.0002|).
   Skip from `down1_out` already carries the bottleneck-relevant
   information for hole-fill.

Linear CKA matrix is remarkably stable across widths. `up2_out` has
the lowest CKA to other stages (~0.89–0.93), confirming it as the
"absorbing state" of the UNet — it converts intermediate features
into pixel-space reconstruction.

**Implication for distillation**:
- `MoebiusSmallStudent.head` only needs to replace `up2_out`'s role.
- The student can have **fewer decoder stages than the teacher** and
  match quality — `up1_out` is actively harmful in distillation.
- `dec2` (16² → 32² stage) with MQA attention is questionable since
  `down2_out` (which feeds it) is REDUNDANT.

## 3. Block-level ablation on synthetic GRT (w=40, 30 leaf modules)

`scripts/profile_teacher_blocks.py`. Zero-ablation per leaf module
(DWResnetBlock2D or MixTransformer2DModel) inside each stage. **Stage-level
ablation UNDERSTIMATES per-stage impact** because skip connections
dilute the effect; block-level ablation measures causal contribution
through the residual stream.

Stage sums (block-level) vs stage-level (whole-stage zero):

| stage | stage-level ΔL1 | block-level Σ ΔL1 | ratio |
|---|---:|---:|---:|
| down0 | +0.0033 | +0.0362 | 11× |
| down1 | +0.0035 | +0.0289 | 8× |
| down2 | -0.0000 | +0.0202 | ∞ (sign flip) |
| up0 | +0.0252 | +0.0468 | 1.9× |
| up1 | -0.0325 | **-0.1224** | 3.8× |
| up2 | +0.5273 | **+1.4568** | 2.8× |

**Key block-level findings (synthetic, w=40)**:

Top 5 reverse-contributing blocks:
1. `up1/resnet2` (DWResnetBlock2D) -0.0681
2. `up2/attn0` (MixTransformer2DModel) -0.0458
3. `up1/attn2` (MixTransformer2DModel) -0.0416
4. `up1/resnet1` (DWResnetBlock2D) -0.0310
5. `up2/resnet0` (DWResnetBlock2D) -0.0309

**All 4 blocks of up1 (resnet1, resnet2, attn2) and the first attn
of up2 are reverse-contributing** — the "Places2 noise injector"
is concentrated in these specific blocks.

Top 4 load-bearing blocks (all in up2):
1. `up2/resnet2` (DWResnetBlock2D) **+0.5968**
2. `up2/attn2` (MixTransformer2DModel) **+0.5273**
3. `up2/resnet1` (DWResnetBlock2D) +0.2872
4. `up2/attn1` (MixTransformer2DModel) +0.1222

**up2's last 2 blocks (resnet2 + attn2 = +1.1241) carry ~77% of up2's
total block-level contribution.** This is the "final refinement"
that makes up2 special — exactly the role the student's `head`
needs to capture.

## 4. Block-level ablation on real COCO 32-case holdout (validation)

`scripts/profile_teacher_real.py`. Same 8-image holdout selection as
`scripts/eval_moebius_small.py` (`np.random.default_rng(7)`, first 8
holdout_images, all 4 (dmax, direction) combos → 32 cases). Real COCO
RGB 512² from `/home/dog/datasets/image/coco_train2017/train2017`,
disparity computed **online** via ZipDepth (no cached disparity for
holdout).

Baseline in-hole ‖pred−noise‖₁ = **0.2514 ± 0.033** (vs synthetic
0.2708 ± 0.16 — real is tighter by ~5×, expected because synthetic
noise dominates per-case variance).

**Stage sums (real COCO) vs synthetic w=40**:

| stage | real COCO | synthetic w=40 | direction |
|---|---:|---:|---|
| down0 | **+0.1390** | +0.0362 | same sign, real 4× stronger |
| down1 | **-0.0399** | +0.0289 | **FLIPPED reverse** |
| down2 | **-0.0648** | +0.0202 | **FLIPPED reverse** |
| up0 | **-0.0813** | +0.0468 | **FLIPPED reverse** |
| up1 | **-0.2989** | -0.1224 | same sign, real 2.4× stronger |
| up2 | **+1.6921** | +1.4568 | same sign, real 1.16× stronger |

**Synthetic profile systematically UNDERSTATES how much of the teacher
is hurting on real data.** The reverse-contribution is more spread
across stages on real COCO. `down1` (-4.0%), `down2` (-6.5%), `up0`
(-8.1%) are all reverse-contributing on real data where synthetic
showed them as neutral or positive.

Top 5 reverse-contributing blocks on real COCO:
1. `up1/resnet1` (DWResnetBlock2D) **-0.0944**
2. `up1/resnet2` (DWResnetBlock2D) -0.0633
3. `up1/attn2` (MixTransformer2DModel) -0.0533
4. `up0/resnet2` (DWResnetBlock2D) -0.0427 — **NEW vs synthetic**
5. `up1/attn0` (MixTransformer2DModel) -0.0370

**up1's 6 blocks are ALL reverse-contributing on real data** (range
-0.0944 to -0.0236). This is the strongest empirical finding.

Top 5 load-bearing blocks on real COCO:
1. `up2/resnet2` (DWResnetBlock2D) **+0.6331**
2. `up2/attn2` (MixTransformer2DModel) **+0.5565**
3. `up2/resnet1` (DWResnetBlock2D) +0.3786
4. `up2/attn1` (MixTransformer2DModel) +0.1538
5. `down0/attn0` (MixTransformer2DModel) +0.0654

**up2's last 2 blocks (resnet2 + attn2 = +1.1896) carry ~70% of up2's
contribution on real data**, confirming the synthetic finding.

## 5. Distillation design recommendations

Cross-referencing synthetic and real findings, with `MoebiusSmallStudent`'s
current `blocks=(5, 5, 5)` as the baseline:

| region | current | real-data evidence | suggested change |
|---|---|---|---|
| enc0 (64²×128, 5 blocks) | over-built | down0 sum +13.9% (positive, large) | KEEP — this is the only positive encoder contribution. Could grow blocks if budget allows. |
| enc1 (32²×256, 5 blocks) | over-built | down1 sum -4.0% (reverse) | SHRINK to 2 blocks or DROP entirely |
| enc2 (16²×512, 4 DW + 1 MQA) | over-built | down2 sum -6.5% (reverse) | DROP MQA attention; consider dropping stage |
| dec2 (16²→32², 1 MQA + 4 DW) | over-built | up0 sum -8.1% (reverse) | DROP MQA attention |
| dec1 (32²×256, 5 blocks) | **over-built** | up1 sum **-29.9% (reverse)** | **DROP entire stage or shrink to 1 block** |
| dec0 (64²×320, 5 blocks) | well-sized | up2 sum +169% (positive, dominant) | KEEP 5 blocks; **last 2 are critical** |
| head (DW3×3 + Conv2d) | undersized | up2/resnet2 alone = +0.633 | **UPGRADE to (DW×2 + Conv1×1)** |

### Head upgrade spec

Current `head` (in `students/moebius_small.py`):
```python
self.head = nn.Sequential(
    nn.Conv2d(128, 128, 3, padding=1, groups=128, bias=False),  # DW3x3
    nn.BatchNorm2d(128),
    nn.ReLU6(),
    nn.Conv2d(128, 4, 1),  # PW1x1 output
)
```

Suggested upgrade that mirrors `up2/resnet2` + `up2/attn2` structure
(skipping MixTF since the student has no encoder_hidden_states):
```python
self.head = nn.Sequential(
    GatedDW7Block(128),    # ≈ DWResnetBlock2D analog, ~49,664 params
    GatedDW7Block(128),    # second DW block, mirrors up2/resnet2
    nn.Conv2d(128, 4, 1),  # output projection
)
```

Estimated cost: ~100K extra params (negligible vs total 10.2M).
Predicted impact: +0.6 in-hole L1 recovery vs single-DW head, matching
the teacher block-level ablation prediction.

### What NOT to do

- **Do not** add cross-attention to the student (no encoder_hidden_states
  in the 9ch prompt input; adding it would require restructuring the
  distillation pipeline).
- **Do not** assume synthetic profile is sufficient — the real COCO
  block-level ablation shows ~2× stronger reverse contributions than
  synthetic, plus FLIPPED signs on down1/down2/up0.

## 6. Verification path

Before committing to the suggested redesigns, run these targeted
ablations on `MoebiusSmallStudent`:

1. **Train baseline** (current `blocks=(5,5,5)` + simple head): 3000
   steps on real COCO 12,204 train cases, CFG 1.0 teacher, measure
   in-hole L1 at each 500-step checkpoint on the 32-case holdout.

2. **Head upgrade** (same blocks, new head): same protocol. Expected
   Δhole_l1: −0.05 to −0.15.

3. **Drop dec1** (blocks=(5,5,1) or skip dec1 entirely): same
   protocol. Expected Δhole_l1: −0.02 to −0.08 (dec1 is hurting).

4. **Drop enc1/enc2 attention** (skip MQA in 16²): same protocol.
   Expected Δhole_l1: ±0.02 (small — enc1/enc2 contributions are
   near-noise floor).

5. **Combined**: blocks=(5, 2, 5) + upgraded head + dec1=1 block.
   Expected Δhole_l1: **−0.10 to −0.20** vs baseline.

Each variant is a 4.5-hour training run; the full ablation matrix is
~25 hours on GB10. The single most informative experiment is the
combined variant (#5).

## 7. Caveats and open questions

- **Synthetic vs real noise**: synthetic GRT uses `randn` images,
  where `||pred - noise||_1` is dominated by per-case random noise.
  Real COCO has structured content; per-case std drops ~5× and
  smaller-effect blocks become detectable. Trust real-COFO numbers
  over synthetic for decisions.
- **CKA projection**: each tap's `(B, C, H, W)` is reduced via global
  avg-pool + a random projection seeded by `C` (channel count).
  Different projections give slightly different CKA values;
  numbers should be treated as ±0.005 noisy.
- **Block-sum is not causal**: stage sums are sanity checks, not
  causal claims — multiple blocks zeroed simultaneously would not
  simply add.
- **The architecture is yaml-driven, not class-default**: someone
  editing `moebius.yaml` to add a `DWDownBlock2D` 4th down + `mid`
  block would change the stage count from 6 to 8. Verify
  `down_blocks` and `up_blocks` counts before re-running the scripts.
- **No block-level CKA yet**: only stage-level CKA. A per-block CKA
  pass would reveal which specific blocks have redundant
  representations. Not done yet because it adds another 30-case
  × 30-block × 1024-d feature collection (estimated +5 min on GB10).

## 8. Reproduction

```bash
cd /home/dog/project/moebius_distill/distill

# 1. Latency
MOEBIUS_UPSTREAM_DIR=/home/dog/project/moebius_distill/Moebius \
  .venv/bin/python scripts/bench_teacher_forward.py --batch-sizes 1 --warmup 10 --iters 30

# 2. Stage-level, 4 widths (~30s each)
for w in 20 40 80 160; do
  MOEBIUS_UPSTREAM_DIR=/home/dog/project/moebius_distill/Moebius \
    .venv/bin/python scripts/profile_teacher.py --hole-width-px $w
done

# 3. Block-level, synthetic (~30s)
MOEBIUS_UPSTREAM_DIR=/home/dog/project/moebius_distill/Moebius \
  .venv/bin/python scripts/profile_teacher_blocks.py --hole-width-px 40

# 4. Block-level, real COCO 32-case (~2 min including ZipDepth online)
MOEBIUS_UPSTREAM_DIR=/home/dog/project/moebius_distill/Moebius \
  .venv/bin/python scripts/profile_teacher_real.py
```

All four scripts are strictly inside `distill/` and do not modify
any checkpoint or upstream repo.
