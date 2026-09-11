# AGENTS.md — work split inside `finetune/`

This file is the contract between the orchestrator (stage 0) and the
three development agents (A, B, C) called out in TDD §10. It exists so
that parallel work does not collide on the same file.

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
