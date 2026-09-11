# Local development guide

This document covers the local development loop for the
`moebius-finetune` package on Windows + WSL. Real training, the ONNX →
RKNN pipeline, and the server-side runs are out of scope for stage 0.

## 1. Layout

```text
finetune/
├── pyproject.toml
├── README.md
├── AGENTS.md
├── .github/workflows/ci.yml
├── src/moebius_finetune/
│   ├── contracts.py
│   ├── cli.py
│   ├── data/      ← Agent A
│   ├── teachers/  ← Agent B
│   ├── students/  ← Agent C
│   ├── training/  ← B and C
│   ├── evaluation/← Agent A
│   └── deployment/← Agent C
├── configs/
├── tests/
└── docs/
```

See `AGENTS.md` for who owns what.

## 2. Local Python environment

The package itself only needs `numpy>=1.24` and `pyyaml>=6.0`. The
optional extras (`teacher`, `student`, `deploy`) are populated by
Agents B and C as they take ownership of their subpackages.

For day-to-day work, install the package editable so the test
collector and console entry points resolve:

```bash
# from the workspace root
wsl.exe -- bash -lc "\
  cd /mnt/d/project/moebius_distill/finetune && \
  ~/ml-venv/bin/python -m pip install -e ."
```

The test runtime also needs `pytest>=7.0`, which is already present in
`~/ml-venv`.

## 3. Run the CPU contract tests

```bash
wsl.exe -- bash -lc "\
  cd /mnt/d/project/moebius_distill/finetune && \
  ~/ml-venv/bin/python -m pytest tests/ -x"
```

The CI workflow at `.github/workflows/ci.yml` runs the same command
on a Python 3.12 runner without torch / diffusers / onnx / rknn, so
this loop is the source of truth for the "does it still work" check.

## 4. Adding a new module

Pick the owner first. Anything that touches a subpackage must come
from its owner (A, B, or C) or from the orchestrator (stage 0). Once
you know the owner:

1. Place the new file inside the owning subpackage, e.g.
   `src/moebius_finetune/data/grt_mask.py` for Agent A.
2. If the new code introduces a public symbol that other agents will
   need, export it from `contracts.py` (orchestrator PR only) and
   mirror the change in `tests/test_contracts.py` so the synthetic
   test stays in sync.
3. Add a subpackage-level `tests/` if the new code needs its own CPU
   tests. Keep the heavy ML imports inside the test body and
   `pytest.importorskip` them so the CPU-only test set still passes.
4. Run the test suite locally before opening a PR.
5. Mention the TDD section that the new code implements in the commit
   message.

## 5. Adding / editing a config

1. Edit or add the YAML under `configs/`. Use forward slashes and
   keep paths as **placeholders** (real paths go into
   `paths.local.yaml`, which is git-ignored).
2. The top-level keys for training-style configs are
   `experiment`, `runtime`, `data`, `model`, `training`,
   `evaluation`. The `rknn_rk3588.yaml` export recipe uses a
   different shape (`target`, `export`, `quantization`, `budget`).
3. Update `tests/test_config.py` if a new config or new required
   field is added.

## 6. Path registry

* `configs/paths.example.yaml` is committed as a **placeholder**.
  Do not put real host paths there.
* The intended copy-and-edit workflow is:
  ```bash
  cp configs/paths.example.yaml configs/paths.local.yaml
  ```
  `paths.local.yaml` is git-ignored and can hold real paths.
* If you need a new path class (e.g. a calibration manifest
  location), add it to `paths.example.yaml` first, then mirror the
  change in your local copy.

## 7. Lint / type-check

Stage 0 does not pin a linter. CI only runs the CPU test set. If you
want to run a quick sanity check, `python -c "import ast;
ast.parse(open('path/to/file.py').read())"` is enough to catch
syntax errors.

## 8. Pre-commit checklist

Before pushing a PR:

* [ ] `python -m pytest tests/ -x` passes locally.
* [ ] No new top-level `import torch` outside `to_torch` (and any
      future deferred-import helper) inside `contracts.py`.
* [ ] No real host paths, weights, or `.env*` files were committed.
* [ ] Commit message references the relevant TDD section number.
* [ ] If the change touches the public contract, the orchestrator
      has been pinged.

## 9. References

* Frozen development plan: `tdd/moebius-depth-finetune-distill-2026-09-12-05-38-54.md`
  (in the parent workspace, not this repo).
* Architecture: `docs/architecture.md`.
* Agent ownership: `AGENTS.md`.
