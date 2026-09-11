# moebius-finetune

> DIBR hole-filling under the **source-coordinate GRT protocol**, with a
> depth-conditioned Moebius teacher and a one-step distilled student
> targeting **RK3588 INT8**.

## What this is

The first goal of the project is to fill the holes left behind by DIBR
warping on a 512×512 image. The conditioning comes from the source
view: a hole mask, the hole-zeroed RGB, and a hole-zeroed L-neg / R-pos
signed inverse depth. The teacher is a depth-conditioned Moebius; the
student is a one-step distilled network that fits the 8 GMACs / 300 MB
DRAM budget on RK3588. Object removal is **not** the task definition.

The implementation follows the frozen development plan at
`tdd/moebius-depth-finetune-distill-2026-09-12-05-38-54.md` in the
parent workspace. Do not edit that file from this package.

## Repository layout

```text
finetune/
├── pyproject.toml
├── README.md
├── AGENTS.md
├── src/moebius_finetune/
│   ├── contracts.py          # public condition-batch + sample manifest API
│   ├── data/                 # Agent A
│   ├── teachers/             # Agent B
│   ├── students/             # Agent C
│   ├── training/             # Agent B + C (split into submodules later)
│   ├── evaluation/           # Agent A
│   └── deployment/           # Agent C
├── configs/                  # pyyaml, paths + training/eval/export
├── tests/                    # CPU-only synthetic tests, no torch by default
├── docs/                     # architecture.md, development.md
└── .github/workflows/        # CI for the public contract + configs
```

Ownership of each subpackage is recorded in `AGENTS.md`. Changes to
`contracts.py` and the public surface must be coordinated through the
orchestrator (stage 0).

## Quick start (local, CPU only)

The contract tests and config loader are pure numpy + pyyaml. Running
them does not need GPU, teacher weights, or any private data.

```bash
# from the workspace root, via WSL
wsl.exe -- bash -lc \
  "cd /mnt/d/project/moebius_distill/finetune && \
   ~/ml-venv/bin/python -m pytest tests/ -x"
```

The CI workflow under `.github/workflows/ci.yml` runs the same command
on a Python 3.12 runner without torch, diffusers, transformers, onnx
or rknn-toolkit.

## Path conventions

All filesystem paths come from a single YAML config
(`configs/paths.example.yaml`) which is committed as a **placeholder**:
real paths are local or per-server. The package never reads unlisted
weights; the loader rejects unregistered checkpoint paths and the
tests never reference them.

The pinned upstream commits referenced from the configs are:

* Moebius: `b88d462bacb9af6e7128a3b4cc4a07418bedfd61`
  (TDD §2.1)
* ZipDepth: the `base` checkpoint under `ZipDepth/checkpoints/`
* Hami data: the junction at `D:\project\Hami\data\images` (TDD §2.2)

These pins are recorded so that a future bootstrap script can fetch
exactly the right tree and verify SHA256.

## Known things this repository does **not** do

* It does not push to a remote. Local Git only.
* It does not modify the upstream Hami / Moebius / ZipDepth / `.reasonix`
  trees.
* It does not auto-download weights or calibration data.
* It does not bypass the public contract to import torch at module
  top-level. Heavy dependencies are deferred to the functions that
  actually need them.
* It does not claim RK3588 device results. There is no hardware under
  test yet; budget numbers are modelled and clearly labelled.
