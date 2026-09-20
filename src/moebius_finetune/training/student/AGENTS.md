# AGENTS.md — `src/moebius_finetune/training/student/`

Distillation loop and supporting components for the GRT hole-fill
student. Inherits the conventions in
[`../../../../AGENTS.md`](../../../../AGENTS.md) — mid-gray hole-fill,
9-channel input, EMA unusable, KD weight bug, ELATENTLPIPS scale.

## Files

| File | Role |
|------|------|
| `distill.py` | Earlier `DistillTrainer` (legacy; uses `distill_student` + `validate_cache_keys`) |
| `losses.py` | `distillation_loss`, `boundary_gradient_l1`, `hole_l1` — paired-distill losses + boundary term |
| `mobile_distill.py` | `train_mobile_student`, `train_mobile_student_traj`, `train_mobile_student_onpolicy`; `TrainStats` — 10-step schedule, teacher-forced, on-policy variants |
| `trajectory_cache.py` | RAM-resident `TrajectoryRamCache` for teacher-free distillation; student-grid rollout mode |
| `prompt.py` | `HOLE_FILL_VALUE = 0.5`, `build_masked_prompt` — **single source of truth for hole-fill** |
| `multigranular.py` | `FeatureProjections`, `cal_kd_loss`, `cal_task_loss`, `cal_elatentlpips_loss`, `cal_adaptive_weights`, `total_loss` — four-term loss book |
| `train_small.py` | `train(cfg, log, *, student, taesd, teacher, lpips, …)`, `TrainConfig`, `main` — `MoebiusSmallStudent` distillation driver |
| `elatentlpips_setup.py` | `load_elatentlpips`, `ensure_ckpt_dir`, `fetch_tuned_checkpoint` — mirror fetch + hardened loader |

## Re-exports (`__init__.py`)

Only `distill.py` + `losses.py`. Everything else must be imported
directly by module path — same anti-pattern as `students/__init__.py`.

```python
# Re-exported
from moebius_finetune.training.student import (
    DistillTrainer, SyntheticTeacherCache, TeacherCacheEntry,
    distill_student, validate_cache_keys,
    boundary_gradient_l1, distillation_loss, hole_l1,
)

# Direct imports (NOT in __init__)
from moebius_finetune.training.student.prompt import HOLE_FILL_VALUE, build_masked_prompt
from moebius_finetune.training.student.multigranular import (
    FeatureProjections, cal_kd_loss, cal_task_loss,
    cal_elatentlpips_loss, cal_adaptive_weights, total_loss,
    ELATENTLPIPS_INPUT_SCALE, ELATENTLPIPS_SDXL_FACTOR,
)
from moebius_finetune.training.student.train_small import train, TrainConfig, main
from moebius_finetune.training.student.elatentlpips_setup import load_elatentlpips
```

## `prompt.py` — mid-gray hole fill (single source of truth)

```python
HOLE_FILL_VALUE = 0.5  # = 0 in [-1,1] = what the pretrained teacher expects
build_masked_prompt(rgb01, mask01)  # rgb * (1-mask) + 0.5 * mask
```

- **Do not change this file's value**. Other code paths call `build_masked_prompt` and pass the result through `train_small._to_latents`, `scripts/eval_moebius_small.py`, `scripts/viz_moebius_small.py`.
- Measured full-hole L1 on 32-case holdout: black 0.562 vs mid-gray 0.217.
- `grt_dataset.build_case` deliberately returns a **black** hole (`rgb*(1-mask)`); do not "fix" it. The fill is applied here, at prompt-assembly time, so the warp product layout remains stable for other consumers.

## `multigranular.py` — four-term loss book

Constants (locked):
- `FEAT_LOSS_WEIGHT = 1.0`
- `KD_LOSS_WEIGHT = 0.01`
- `TASK_LOSS_WEIGHT = 0.5`
- `ELATENTLPIPS_LOSS_WEIGHT = 0.5`
- `ELATENTLPIPS_SDXL_FACTOR = 0.13025`
- **`ELATENTLPIPS_INPUT_SCALE = 1.0`** — do NOT change; the library is called with `normalize=False` and the scale keeps the post-BN std at the calibrated 0.730 (default would give 0.309).
- `TAP_PAIRS = ((0, 5), (1, 3), (2, 2))` — measured student→teacher tap indices by shape; not symmetric.

Functions:
- `FeatureProjections(student_channels=(128, 256, 512), teacher_channels=(320, 1280, 1280))` — 1×1 projections for KD; training-only, discarded at deploy.
- `cal_kd_loss(pred_S, pred_T, projections, *, tap_pairs=TAP_PAIRS, feat_loss_weight=FEAT_LOSS_WEIGHT) → (featkd, outkd)`.
- `cal_task_loss(pred_S, target_eps) → Tensor` — MSE(ε, noise).
- `cal_elatentlpips_loss(pred_S, target, lpips_model, scheduler, timesteps, noisy_latents) → Tensor`.
- `cal_adaptive_weights(featkd, task, outkd, lpips, *, feat_anchor, out_anchor) → (feat_w, out_w_outkd, out_w_lpips, diag)` — gradient-norm balancing with `clamp(..., 0.0, 1e4)` / `1e6`. **Currently missing the `KD_loss_weight` factor in `feat_w`'s numerator** (see top-level AGENTS §Gotchas #3); weight pins to the `1e4` clamp by ~step 150.

## `train_small.py` — distillation loop

Path constants (hard-coded at module top — same convention as upstream `full_ft_v1.py`):

```python
IMG_DIR      = "/home/dog/datasets/image/coco_train2017/train2017"
MANIFEST     = "/home/dog/datasets/moebius_finetune/grt_manifest.json"
DISP_CACHE   = "/home/dog/datasets/moebius_finetune/disparity"
FINAL_PT     = "/home/dog/datasets/moebius_finetune/runs/taesdxl_moebius_ft/final.pt"
MOEBIUS_YAML = "/home/dog/project/moebius_distill/Moebius/config/model_cfg/moebius.yaml"
TAESDXL_ENC  = "/home/dog/project/HAMI/thirdParty/taesd/taesdxl_encoder.pth"
TAESDXL_DEC  = "/home/dog/project/HAMI/thirdParty/taesd/taesdxl_decoder.pth"
EVAL_SEED    = 12345
```

Loop mechanics (locked):
- **Latent mask** uses `F.interpolate(mask, size=(64, 64), mode="nearest")` — never `bilinear` (that's PixelHacker).
- **VAE encoding**: TAESDXL takes `[0,1]` input with `scaling_factor=1.0` — do **not** do `*2-1`, do **not** multiply by `0.13025`.
- **t sampling**: `t_float = 1000 * sigmoid(randn(B))` (logit-normal). Must do **both** conversions: `t_long = t_float.long()` for `scheduler.add_noise`; `t_float` for sinusoidal embedding. They are not interchangeable.
- **9ch**: `x9 = torch.cat([scheduler.add_noise(x0_lat, noise, t_long), latent_mask, masked_lat], dim=1)`.
- **Teacher target**: `teacher.eval()`, `no_grad()`, single batch at **CFG 1.0** (i.e. cond branch only — `input_ids = torch.arange(10).unsqueeze(0)`, no double-batch).
- **EMA(0.9999)** kept in RAM only — never persisted (deploy uses `model_state`).
- **Checkpoints** every `--save-every` steps; `final.pt` at the end.

CLI:
```bash
cd distill && MOEBIUS_UPSTREAM_DIR=/home/dog/project/moebius_distill/Moebius \
  .venv/bin/python -m moebius_finetune.training.student.train_small \
  --steps 20 --batch-size 4 --out-dir /tmp/msmoke
```

## `elatentlpips_setup.py` — hardened LPIPS staging

- Mirror: `https://hf-mirror.com/Mingguksky/elatentlpips/resolve/main/elatentlpips_ckpt` (no internet to `huggingface.co`).
- `load_elatentlpips(*, cwd=None, device="cpu")`:
  - Disables `upfirdn2d._init` → skip ninja/CUDA JIT (ninja not installed; PyTorch path equivalent).
  - Temporarily relaxes `LatentVGG16BN.load_state_dict` to `strict=False` for the trunk shim only.
  - The authoritative `strict=True` load over the tuned checkpoint runs unchanged afterwards.
  - `pnet_rand=True` discards the 4-channel first conv (unusable with 4-channel latent input).
  - The 1.1 GB `sdxl_latent_vgg16.pth.tar` is only an init source — the tuned checkpoint fully overwrites its weights.

## `trajectory_cache.py` — RAM-resident teacher-trajectory cache

- `TrajectoryRamCache(capacity=N)` pins the full bf16 buffer up front (~54 GB for 160k trajectories) to avoid doubling the peak at finalize time.
- `grid="student"` integrates the teacher field along the student's own 10-step schedule (half the forwards AND matches the student's inference-time state distribution). `grid="teacher"` keeps the original schedule-subsample behavior.
- `state_loss_weights` upweights the low-noise end (t=151/51) where the eps-MSE is 10-15× the high-noise end.
- `x0_endpoint_weight`: the implied clean latent from the student's eps is pulled toward the teacher chain's endpoint.
- **On-policy** mode (`train_mobile_student_onpolicy`): the student rolls its own 10-step trajectory (no grad through DDIM), the teacher corrects at each visited state, and one batched grad forward regresses the student's eps onto those corrections.

## Test conventions

- `tests/training/student/test_train_small.py` — 7 tests; **no** real Moebius teacher; stubs TAESDXL/teacher/LPIPS via fixture. Spies `student.time_embed.forward` to assert no `nn.Embedding` lookup.
- `tests/training/student/test_multigranular.py` — 10 tests; 2 require `elatentlpips` + tuned checkpoint (`@pytest.mark.skipif` gates).
- `tests/training/student/test_trajectory_cache.py` — 13 tests; uses RAM-only synthetic cache.
- All require torch at module top level. Run with `pytest tests/training/student -q` (~40 s for the focused loop).
