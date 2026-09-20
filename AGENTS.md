# AGENTS.md — `distill/`

DIBR hole-filling student distillation. 226M depth-conditioned Moebius
teacher → 10.2M one-step student, RK3588 INT8 target.

Sibling trees (`Moebius/`, `PixelHacker/`, `ZipDepth/`, `m2svid/`,
`Hybrid-SD/`, `tdd/`) are pinned upstream references — **read-only**.
Workspace map: [`../AGENTS.md`](../AGENTS.md).

## Environment — DGX Spark (GB10), aarch64

| Concern  | Value                                                    |
|----------|----------------------------------------------------------|
| Arch/OS  | `aarch64`, Ubuntu 24.04, CUDA 13.0, `sm_121`             |
| Venv     | `.venv` (uv, Python 3.12.3, torch 2.13.0+cu130)          |
| RAM      | 121 GB — budget runs against an **80 GB peak**           |
| GPU      | GB10, driver 580.159                                     |

Import-time traps (not optional, even when unused):
- `flash-linear-attention` — `Moebius/model_lib/__init__.py` imports it; without it the teacher will not load.
- `onnxscript` — required by `torch.onnx.export` on torch 2.13.

`huggingface.co` is **unreachable** from this box. Use `hf-mirror.com`
(wired into `src/moebius_finetune/training/student/elatentlpips_setup.py`).

## Structure

```
distill/
├── AGENTS.md                 # this file
├── pyproject.toml            # moebius-finetune v0.0.1; 8 console scripts
├── .github/workflows/ci.yml  # 2 jobs: cpu-contract-tests, ml-cpu-tests
├── configs/                  # YAML configs; paths.local.yaml git-ignored
├── ckpt/                     # symlinks → /home/dog/datasets/.../elatentlpips_ckpt/
├── scripts/                  # driver scripts (NOT installable); see scripts/AGENTS.md
├── docs/                     # architecture.md, development.md
└── src/moebius_finetune/
    ├── __init__.py           # flat re-export of contracts
    ├── contracts.py          # public API (ConditionBatch, SampleManifest, …)
    ├── cli.py                # 8 entry points; 3 implemented, 5 stubs
    ├── data/                 # GRT dataset, ZipDepth, manifest IO
    ├── deployment/           # budget, ONNX/RKNN export, LPDDR4X bandwidth
    ├── evaluation/           # hole/global L1, masked LPIPS, evaluator
    ├── students/             # student archs; see students/AGENTS.md
    ├── teachers/             # Moebius wrapper, depth adapter, loader
    └── training/
        ├── codec/
        ├── student/          # distillation loop; see training/student/AGENTS.md
        └── teacher/          # finetune, cache, recipe
```

## Where to look

| Task | Location | Notes |
|------|----------|-------|
| Cross-module API | `src/moebius_finetune/contracts.py` | `ConditionBatch`, `SampleManifest`, `Split`, `Direction`, `CoordFrame`, `DepthNormalization`, `to_torch`, `inpaint`, `validate_condition`, `ConditionContractError` |
| GRT case building | `src/moebius_finetune/data/grt_dataset.py` → `build_case(case, *, image_dir, predictor=None, size=512, disp=None)` | Returns `rgb_hole`, `hole_mask`, `depth_hole`, `target`, `hole_ratio`. Disparity is never persisted. |
| Student model | `src/moebius_finetune/students/moebius_small.py` → `MoebiusSmallStudent` (NOT in `students/__init__.py`) | 10.16M params / 6.58 GMac at `[1,9,64,64]` |
| Distillation loop | `src/moebius_finetune/training/student/train_small.py` → `train(cfg, log, *, student, taesd, teacher, lpips, ...)` | Inject all heavy components. EMA in RAM only. |
| Loss book | `src/moebius_finetune/training/student/multigranular.py` | Four-term loss + adaptive grad-norm weights |
| Hole-fill prompt | `src/moebius_finetune/training/student/prompt.py` → `HOLE_FILL_VALUE = 0.5`, `build_masked_prompt` | **Single source of truth** — change here, nowhere else |
| LPIPS staging | `src/moebius_finetune/training/student/elatentlpips_setup.py` → `load_elatentlpips()` | Mirror fetch from hf-mirror.com; ninja/JIT disabled |
| Budget / MACs | `src/moebius_finetune/deployment/budget.py` → `compute_algorithm_macs(model, spec, *, adapter=None)` | Fails closed on unknown ops. Pass `adapter` for non-default forward signatures |
| Eval driver | `scripts/eval_moebius_small.py` | 32-case GRT holdout, 10-step DDIM, seed 12345; fixes the row-0 mask bug |
| Quick test loop | `pytest tests/test_contracts.py tests/test_config.py tests/data/ -q` | 157 tests, ~1.2s, no torch |
| Student loop | `pytest tests/students tests/training/student -q` | 133 tests, ~40s |

## Public API surface (`moebius_finetune`)

Re-exported from root: the 10 names in `contracts.py` plus (via
`students/__init__.py`) `InvertedResidualBlock`, `ContextBlock`,
`DownsampleBlock`, `UpsampleBlock`, `TailRefineBlock`, `fuse_bn`,
`LatentStudentV0`, `LightweightCodec`, `OneStepLatentNet`,
`MobileMoebius`, `PixelStudentV0`.

**Not re-exported (must import directly):**
- `MoebiusSmallStudent`, `StudentOutput`, `SinusoidalTimeEmbedding` ← `students.moebius_small`
- `GatedDW7Block` ← `students.gated`
- `train`, `TrainConfig`, `main` ← `training.student.train_small`
- `FeatureProjections`, `cal_*`, `total_loss`, `ELATENTLPIPS_INPUT_SCALE` ← `training.student.multigranular`
- `HOLE_FILL_VALUE`, `build_masked_prompt` ← `training.student.prompt`
- `load_elatentlpips`, `ensure_ckpt_dir`, `fetch_tuned_checkpoint` ← `training.student.elatentlpips_setup`

## Env vars

| Variable | Value | Fallback |
|----------|-------|----------|
| `MOEBIUS_UPSTREAM_DIR` | `/home/dog/project/moebius_distill/Moebius` | probed via `_candidate_roots()` (local first, then WSL/Windows forms) |
| `MOEBIUS_WEIGHTS_PATH` | `.../Moebius/weights/moebius/pretrained/diffusion_pytorch_model.bin` | conftest skip if missing |
| `ELATENTLPIPS_CKPT_DIR` | `/home/dog/datasets/moebius_finetune/elatentlpips_ckpt` | default in `elatentlpips_setup.py` |
| `ZIPDEPTH_UPSTREAM_DIR` | `/home/dog/project/moebius_distill/ZipDepth` | candidates in `grt_dataset.resolve_zipdepth_dir()` |
| `ELATENTLPIPS_CKPT_DIR` | see above | — |

`configs/paths.local.yaml` (git-ignored) holds this box's real paths.

## Data & runs

```
/home/dog/datasets/moebius_finetune/
├── disparity/          # per-image ZipDepth .npy cache
├── grt_manifest.json   # 12,204 GRT cases (train/eval split)
├── elatentlpips_ckpt/  # 1.1 GB shim + tuned checkpoint
└── runs/<run_name>/    # final.pt, step_*.pt, train.log, eval*.json, viz/, REPORT.md
```

`build_512_sample` / `build_case` read `<data_root>/disparity/<id>.npy`
and raise `FileNotFoundError` on miss. **Eval holdout runs ZipDepth
online** — no second cache (user directive).

`runs/` is outside the repo and holds hand-written teacher drivers
(`taesdxl_moebius_ft/full_ft_v1.py`, `eval_trajectory.py`) that contain
two real bugs. Do not copy; the corrected forms live in `scripts/`.

## Critical gotchas — read before touching training or eval

### 1. Hole-fill is mid-gray, NOT black

- In `[-1,1]` the hole must be at **0**; in `[0,1]` that is **0.5**.
- `grt_dataset.build_case` deliberately returns a black hole (`rgb*(1-mask)`); other consumers rely on this. Do not "fix" it.
- Fill is applied where the **prompt** is assembled. **Single source of truth**: `src/moebius_finetune/training/student/prompt.py` (`HOLE_FILL_VALUE = 0.5`, `build_masked_prompt`). Wired into `train_small._to_latents`, `scripts/eval_moebius_small.py`, `scripts/viz_moebius_small.py`.
- Measured full-hole L1, 32 GRT holdout, same teacher/noise: **black 0.562 vs mid-gray 0.217**.

### 2. EMA is unusable at these step counts

`ema_decay=0.9999` over ~3000 steps leaves ~74 % weight on the random
init. **Always evaluate and deploy `model_state`, never `ema_state`**
(pass `--state model_state`).

### 3. KD is effectively inert in the current weighting

`multigranular.py:243` — `cal_adaptive_weights` sets `feat_weight_task
= ‖∇featkd‖ / (‖∇task‖ + 1e-4)` but the numerator is missing the
`KD_loss_weight` factor upstream `cal_adaptive_weights_type8` has. The
weight pins to the `1e4` clamp ceiling by ~step 150; `outkd`/`featkd`
get ~1-4 % of gradient vs ~50 % each for task and LPIPS. Treat
"distillation" in a run's name as a claim to verify, not a fact.

### 4. E-LatentLPIPS input scale is pinned

`ELATENTLPIPS_INPUT_SCALE = 1.0` in `multigranular.py` with the library
called via `normalize=False`. The library's default double-normalises
our TAESDXL latents (post-BN std **0.309** vs correct **0.730**). Do
not "simplify" this away.

### 5. Latent mask uses `mode="nearest"` to 64×64

The `bilinear` in `train_distillation.py:317` belongs to the PixelHacker
pipeline and does **not** apply here.

### 6. Metric convention

`mask_t[0, 0] > 0.5` (legacy `eval_trajectory.py:97`) on `[1,H,W]`
selects **image row 0**, not the mask; it returns fake `0.0` whenever
the hole misses the top row (8 of 32 cases). **Correct form** is
`mask_t[0]` — implemented in `scripts/eval_moebius_small.py::_hole_l1`.

Reference values under the **correct** metric (full-hole L1, `[-1,1]`
units, 32 holdout cases, 10-step DDIM, noise seed 12345):

| Object                         | Value |
|--------------------------------|-------|
| Teacher @ mid-gray (correct)   | 0.217 |
| Oracle in-hole mean-fill       | 0.322 |
| Warp left as-is                | 0.365 |
| Teacher @ black                | 0.562 |
| Best student so far (v1 @6000) | 0.372 |

A student at 0.372 has **not** beaten constant mean-fill. Judge new
numbers against this table, not against the old `≤0.105` bar.

## Conventions

- **CPU tests must not import torch, diffusers, transformers, onnx, or
  rknn at module top level.** Lazy-import inside the test or skip.
- `tests/` mirrors `src/moebius_finetune/` subpackage-by-subpackage.
- Conftest exists only in `tests/teachers/`,
  `tests/training/student/`, `tests/training/teacher/` (none at top).
- `contracts.py` is the cross-module public API. Changing it means
  updating dependants **and** `tests/test_contracts.py` in the same
  change.
- `pyproject.toml` extras groups `data`, `teacher`, `student`, `deploy`
  are placeholders; only `dev` and `eval` carry deps today.
- `scripts/` are **not** registered as console scripts. They use
  `sys.path.insert` to find upstream Moebius + TAESDXL — they're driver
  scripts in the tradition of the upstream `runs/` folder.

## Anti-patterns — explicitly forbidden

- **DO NOT modify** `Moebius/`, `PixelHacker/`, `ZipDepth/`, `m2svid/`, `Hybrid-SD/`, `tdd/` (upstream).
- **DO NOT modify** the existing tests/ files in this repo unless
  tracked; new tests only under the existing subdirs.
- **DO NOT** persist disparity, latents, EMA cache, or any training
  intermediate state to disk.
- **DO NOT** change `HOLE_FILL_VALUE` outside `training/student/prompt.py`.
- **DO NOT** evaluate/deploy `ema_state` at these step counts.
- **DO NOT** fall back to `mask_t[0, 0]` for in-hole metrics.
- **DO NOT** introduce new dependencies outside `[project.optional-dependencies]` without a plan-doc entry.

## Commands

```bash
cd /home/dog/project/moebius_distill/distill

# Fast subset, no torch (157 tests, ~1.2s)
pytest tests/test_contracts.py tests/test_config.py tests/data/ -q

# Student-focused loop (133 tests, ~40s)
pytest tests/students tests/training/student -q

# Full suite (482 tests, slow; loads 226M teacher)
MOEBIUS_UPSTREAM_DIR=/home/dog/project/moebius_distill/Moebius \
  .venv/bin/python -m pytest tests/ -q

# Eval a student checkpoint (uses corrected row-0 metric)
cd distill && MOEBIUS_UPSTREAM_DIR=/home/dog/project/moebius_distill/Moebius \
  .venv/bin/python scripts/eval_moebius_small.py --teacher-check
```

## Notes

- `README.md` quick-start is **stale** (still shows `wsl.exe` and
  `/mnt/d/...`). It also claims the repo has no remote — there is a
  remote (`origin git@github.com:BruceYeung22/distill_mb.git`).
- `docs/development.md` retains WSL/Windows language. Use the actual
  paths here, not the docs.
- `src/moebius_finetune.egg-info/` is at the project root, not under `src/` — harmless.
- `logs/` (inside the repo) has stray PNG comparison grids; `.gitignore`
  excludes `/logs/` but those were committed before the rule.

## Subpackage AGENTS.md

- `src/moebius_finetune/students/AGENTS.md` — student architectures (pixel / latent / mobile / `moebius_small` / `gated` / common).
- `src/moebius_finetune/training/student/AGENTS.md` — distillation loop, losses, prompt builder, LPIPS staging.
- `scripts/AGENTS.md` — eval / viz drivers.
