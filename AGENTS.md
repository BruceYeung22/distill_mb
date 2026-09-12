# AGENTS.md — work split inside `finetune/`

This file is the contract between the orchestrator (stage 0) and the
three development agents (A, B, C) called out in TDD §10. It exists so
that parallel work does not collide on the same file.

## Local environment (DGX Spark, aarch64) — read this first

The package is developed on a **DGX Spark (NVIDIA GB10, `aarch64`,
CUDA 13.0)**. This was previously a **Windows + WSL2 box with an RTX
5060 (`x86_64`)**, so the old docs and defaults describe a machine that
is no longer the target. Do not copy the WSL commands verbatim.

| Concern            | WSL + RTX 5060 (old)                    | DGX Spark GB10 (now)                              |
|--------------------|-----------------------------------------|---------------------------------------------------|
| Arch / OS          | `x86_64`, Windows + WSL2                | `aarch64`, Ubuntu 24.04 (`6.17` kernel)           |
| GPU                | RTX 5060                                | GB10 (Grace Blackwell, `sm_121`), driver `580.159`|
| CUDA               | 12.x                                    | 13.0 (`nvcc` V13.0.88)                            |
| Venv               | `~/ml-venv` under WSL                   | `distill/.venv` (uv, Python 3.12)                 |
| Repo mount path    | `/mnt/d/project/moebius_distill`        | `/home/dog/project/moebius_distill`               |
| Data root          | `D:/datasets/...` (Windows junction)    | `/home/dog/datasets`                              |
| Inference venv     | HAMI `~/ml-venv`                        | HAMI `.venv` (separate; do not mix)               |

There is no `/mnt/d`, no `D:/`, and no `~/ml-venv` on this box. Anything
that still references those paths is stale; the `_candidate_roots()`
helpers in the test conftests and `teachers/loader.py` list the local
path **first** and keep the WSL/Windows forms only as fallbacks.

### Install (one-time)

```bash
cd /home/dog/project/moebius_distill/distill
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -e .          # numpy, pyyaml, Pillow
uv pip install --python .venv/bin/python "pytest>=7.0" "pytest-cov>=4.0"
uv pip install --python .venv/bin/python \
    "torch==2.13.0" "diffusers==0.39.0" transformers accelerate safetensors \
    omegaconf timm tqdm einops opencv-python-headless scipy orjson toml pandas \
    "flash-linear-attention==0.3.2" \
    matplotlib onnx onnxruntime onnxscript
```

Notes that cost time to rediscover:

* `torch==2.13.0` from the configured mirror resolves to the
  `+cu130` aarch64 build on this box (this is what HAMI's venv also
  has). `torch.cuda.is_available()` is `True` and `sm_121` matmuls run.
* `flash-linear-attention` is required at **import** time by
  `Moebius/model_lib/__init__.py`, even though the 9-channel teacher
  path never calls the GLA branch. Without it the teacher will not load.
* `Pillow`, `matplotlib`, `onnx`, `onnxruntime`, and `onnxscript` are
  imported at module top level by the data pipeline / evaluation /
  deployment subpackages. `onnxscript` in particular is required by
  `torch.onnx.export` on torch 2.13; without it the ONNX exporter
  raises `ModuleNotFoundError`.
* `uv` is configured against the Tsinghua mirror
  (`UV_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple`); no manual
  index flags are needed.

### Run the tests

```bash
cd /home/dog/project/moebius_distill/distill
MOEBIUS_UPSTREAM_DIR=/home/dog/project/moebius_distill/Moebius \
  .venv/bin/python -m pytest tests/ -q
```

The environment variables the code reads, and their local values:

| Variable                     | Value on this box                                                  |
|------------------------------|--------------------------------------------------------------------|
| `MOEBIUS_UPSTREAM_DIR`       | `/home/dog/project/moebius_distill/Moebius`                         |
| `MOEBIUS_WEIGHTS_PATH`       | `.../Moebius/weights/moebius/pretrained/diffusion_pytorch_model.bin`|

Both have working fallbacks (the loader and conftests probe the local
path first), so exporting them is optional — but explicit is safer.

The full suite loads the 226M-parameter teacher several times and takes
**~6 minutes** on this box (it was skipped entirely in CI-less runs
before the weight path was fixed). The CPU-only contract subset
(`tests/test_contracts.py tests/test_config.py tests/data/`) still runs
in under a second and needs only numpy + pyyaml + Pillow + pytest.

### Real paths

`configs/paths.local.yaml` (git-ignored) holds this box's real paths:
teacher checkpoint, ZipDepth checkpoint, COCO directories, and the
disparity cache root. `configs/paths.example.yaml` stays a placeholder.

### Real-data pipeline prerequisite: the disparity cache

`data.pipeline_512.build_512_sample` reads disparity from
`<data_root>/disparity/<image_id>.npy` (or `.pt`); it does **not** run
ZipDepth itself. Without that cache every call raises
`FileNotFoundError` pointing at the cache step. On this box the cache
was produced by running `ZipDepth/checkpoints/zipdepth_base.pth` over
the 512-resized COCO images, e.g. for one image:

```python
import sys, numpy as np
from PIL import Image
sys.path.insert(0, "/home/dog/project/moebius_distill/ZipDepth")
from zipdepth.inference.predictor import DepthInference
inf = DepthInference(
    checkpoint_path="/home/dog/project/moebius_distill/ZipDepth/checkpoints/zipdepth_base.pth",
    device="cuda",
)
arr = np.asarray(Image.open("<coco>/<image_id>.jpg").convert("RGB").resize((512, 512)))
d = inf.infer_image(arr[:, :, ::-1].astype(np.uint8))   # predictor wants BGR uint8
np.save("<data_root>/disparity/<image_id>.npy", d.astype(np.float32))
```

A 6-image sample already exists under
`/home/dog/datasets/coco32_grt_local/` (disparity cache + 12 built
cases + `manifest.json`); it is a smoke fixture, not a training set.
The full upstream dataset build lives in the sibling **HAMI** project
(`scripts/prepare_coco32_grt.py`, `scripts/build_grt_train_set.py`),
whose `ZipDepthDisparity` adapter wraps the same checkpoint.

## File ownership

| Owner       | Path inside `finetune/src/moebius_finetune/`                                  | Notes                                |
|-------------|------------------------------------------------------------------------------|--------------------------------------|
| Stage 0     | `__init__.py`, `contracts.py`, `cli.py`                                       | Public surface. Change via PR only.  |
| Stage 0     | `data/__init__.py`, `teachers/__init__.py`, `students/__init__.py`           | Empty placeholders.                  |
| Stage 0     | `evaluation/__init__.py`, `deployment/__init__.py`, `training/__init__.py`   | Empty placeholders.                  |
| Agent A     | everything else under `data/` and `evaluation/`                              | Manifest, GRT, datasets, metrics.    |
| Agent B     | everything else under `teachers/` and `training/teacher/`                     | Moebius adapter, fine-tune, cache.   |
| Agent C     | everything else under `students/`, `training/codec/`, `training/student/`, `deployment/` | Codec, students, ONNX/RKNN. |

Configuration files under `configs/` are owned by stage 0; B and C may
extend them, but the loader and the path-registration helpers stay
under stage 0's control.

## Public-contract change flow

Anything exported from `moebius_finetune.contracts` (`ConditionBatch`,
`SampleManifest`, `Split`, `Direction`, `CoordFrame`,
`DepthNormalization`, `to_torch`, `inpaint`, `validate_condition`,
`ConditionContractError`) is part of the cross-agent API. To change
it:

1. Open a PR describing the breaking change, the impacted subpackages
   and the migration plan.
2. Wait for the orchestrator (stage 0 owner) to confirm.
3. Update the contract, the dependent agent's tests, and the
   `tests/test_contracts.py` synthetic fixtures in the same PR.
4. Re-run the CPU test suite locally and in CI.

The same rule applies to any new shared dataclass, enum or helper that
will be consumed by more than one of the A/B/C subpackages.

## Tests

* CPU tests live in `tests/` and must not `import torch`, `diffusers`,
  `transformers`, `onnx`, or `rknn` at module top level. Heavy
  dependencies are imported inside the test that needs them and are
  skipped if unavailable.
* Subpackage-specific tests belong inside the owning subpackage's
  `tests/` directory once the package owns a `tests/` folder.
* The reference plan in `tdd/moebius-depth-finetune-distill-2026-09-12-05-38-54.md`
  is the source of truth. Section numbers in commit messages and
  comments help reviewers locate context quickly.

## Reference

* Architecture overview: `docs/architecture.md`
* Local dev / how to add a module: `docs/development.md`
* Frozen development plan (in the parent workspace, **not** this
  repo): `tdd/moebius-depth-finetune-distill-2026-09-12-05-38-54.md`
