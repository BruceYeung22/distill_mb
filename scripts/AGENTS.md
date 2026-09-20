# AGENTS.md — `scripts/`

Driver scripts for the GRT hole-fill student. **Not registered as
console scripts** in `pyproject.toml`. They use `sys.path.insert` to
locate upstream `Moebius/` and `thirdParty/taesd` rather than relying on
an install.

Inherits the conventions in [`../AGENTS.md`](../AGENTS.md) — mid-gray
hole-fill, the row-0 mask metric bug, the EMA-unusable rule, etc.

## Files

| File | Role | Args |
|------|------|------|
| `eval_moebius_small.py` | 32-case GRT holdout evaluator for `MoebiusSmallStudent`. Reproduces the `runs/taesdxl_moebius_ft/eval_trajectory.py` protocol with **correct** in-hole L1 metric. | `--ckpt PATH` (default `final.pt` in `OUT_DIR`), `--state {model_state, ema_state}` (default `model_state`), `--steps 10`, `--noise-seed 12345`, `--teacher-check` |
| `viz_moebius_small.py` | Side-by-side visual QA: warp / student / teacher (CFG 1.0) / GT, full-frame + hole-cropped. | `--ckpt PATH`, `--state {model_state, ema_state}`, `--n-cases 5`, `--noise-seed 12345` |
| `viz_teacher_convention.py` | Renders teacher output under **black-hole vs mid-gray-hole** to make the convention difference visible. | `--cases N`, `--noise-seed 12345` |

## Shared protocol (locked — these are why the three scripts coexist)

- **Holdout selection**: 8 images × 4 cases = 32.
  ```python
  rng = np.random.default_rng(7)
  hold_imgs = sorted({c.source_id for c in hold_cases})
  sel = [hold_imgs[i] for i in rng.permutation(len(hold_imgs))[:8]]
  ```
  Do **not** substitute "sorted first 32" — that's a different 8-image set.
- **Noise draw**: `torch.manual_seed(12345)` + `np.random.seed(12345)` set **once** before the case loop; per-case noise is sampled in case-id order. Pinned draw order is what makes the teacher-check reproducible.
- **Sampler**: 10-step DDIM (`scheduler.set_timesteps(10)`).
- **Student** runs **without CFG** (guidance is baked in).
- **Teacher-check** uses **CFG 1.0** (single batch, cond branch only — `input_ids = torch.arange(10).unsqueeze(0)`); `eps = cond`.

## Metric — `_hole_l1`

```python
def _hole_l1(pred_rgb: np.ndarray, target_rgb: np.ndarray,
             hole_mask: np.ndarray) -> float:
    """Full-hole L1 in [-1, 1] units."""
    m = hole_mask[0]                  # NOT hole_mask[0, 0]!
    pred_hole = pred_rgb[:, m > 0.5]
    target_hole = target_rgb[:, m > 0.5]
    return float(np.abs(pred_hole - target_hole).mean())
```

The legacy form `m = hole_mask[0, 0] > 0.5` selects **image row 0**,
not the mask — it returns fake `0.0` whenever the hole misses the top
row (8 of 32 cases). Every number from that script is void. Use
`hole_mask[0]` (this implementation) or `mask_t[0]` in torch.

Reference values (correct metric, 32 holdout, 10-step DDIM, seed 12345):

| Object                         | Value |
|--------------------------------|-------|
| Teacher @ mid-gray (correct)   | 0.217 |
| Oracle in-hole mean-fill       | 0.322 |
| Warp left as-is                | 0.365 |
| Teacher @ black                | 0.562 |
| Best student so far (v1 @6000) | 0.372 |

A student at 0.372 has **not** beaten constant mean-fill. Judge new
numbers against this table, not against the old `≤0.105` criterion.

## Hole-fill wiring

All three scripts use `build_masked_prompt` from
`moebius_finetune.training.student.prompt` (mid-gray). The TAESDXL
encoder consumes `[0, 1]` images with `scaling_factor=1.0` — do **not**
`*2-1` and do **not** multiply by `0.13025`.

## Path conventions

`viz_teacher_convention.py` and `viz_moebius_small.py` both
`sys.path.insert(0, os.path.dirname(__file__))` to import each other as
`sibling_module`. They expect `OUT_DIR` to already exist (created by
`train_small.py`); default is `moebius_small_v1` under
`/home/dog/datasets/moebius_finetune/runs/`.

## Usage

```bash
cd /home/dog/project/moebius_distill/distill
export MOEBIUS_UPSTREAM_DIR=/home/dog/project/moebius_distill/Moebius
.venv/bin/python scripts/eval_moebius_small.py --teacher-check     # reproduce 0.217
.venv/bin/python scripts/eval_moebius_small.py \
  --ckpt /home/dog/datasets/moebius_finetune/runs/moebius_small_v1/final.pt
.venv/bin/python scripts/viz_moebius_small.py --n-cases 5
.venv/bin/python scripts/viz_teacher_convention.py --cases 3
```

## Anti-patterns

- **Do not** copy these into a console-script entry point without
  also wiring their `sys.path.insert` away — they depend on
  `MOEBIUS_UPSTREAM_DIR` and the third-party TAESDXL weights directory
  existing on the host.
- **Do not** swap to `bilinear` latent-mask interpolation
  (PixelHacker convention) — the GRT path is `nearest`.
- **Do not** add CFG to the student forward — guidance is baked in.
- **Do not** evaluate `ema_state` at the current step counts;
  pass `--state model_state`.
