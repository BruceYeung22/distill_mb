# Architecture overview

The package implements the depth-conditioned Moebius teacher + one-step
distilled RK3588 INT8 student pipeline called out in the frozen
development plan at
`tdd/moebius-depth-finetune-distill-2026-09-12-05-38-54.md` (in the
parent workspace; not tracked by this repo).

## High-level data flow

```text
COCO source RGB ──► GRT depth (ZipDepth) ──► per-case sample manifest
                                              │
                                              ▼
                                  Moebius teacher (depth branch)
                                              │
                                              ▼
                              teacher cache: (case, seed) →
                              noise, final latent, decoded RGB,
                              scheduler config, checkpoint hash
                                              │
                                              ▼
                              codec + pixel/latent student
                                              │
                                              ▼
                              ONNX → RKNN INT8 (RK3588)
```

## Subpackage layout

```text
src/moebius_finetune/
├── contracts.py    # ConditionBatch, SampleManifest, validate_condition,
│                   # inpaint, to_torch.  Public API.  Stage 0 owns.
├── data/           # Agent A: GRT mask, ZipDepth cache, 512 dataset,
│                   # source-frame manifest.  TDD §4.2, §4.3.
├── teachers/       # Agent B: depth-conditioned Moebius adapter,
│                   # fine-tune loop, cache generator.  TDD §5.
├── students/       # Agent C: PixelStudent-v0, LatentStudent-v0,
│                   # lightweight codec.  TDD §6.
├── training/       # Stage 0 ships the placeholder.  B and C will
│                   # create training/{teacher,codec,student}/ as
│                   # they own the respective loops.
├── evaluation/     # Agent A: unified evaluator, per-case + group
│                   # metrics, comparison figures.  TDD §9.3.
└── deployment/     # Agent C: ONNX export, RKNN conversion,
                    # quantization, budget profiling.  TDD §8.
```

## Stage 0 deliverables

* `contracts.py` — typed condition batch, sample manifest, validator,
  deterministic `inpaint` composite, lazy `to_torch` adapter.
* Config skeletons for paths, smoke, teacher, students, RKNN export.
* `cli.py` console entry stubs that raise `NotImplementedError` until
  Agents A/B/C wire them.
* CPU-only synthetic tests for the contract and the configs.
* GitHub Actions CI for the CPU test set.
* Local Git repository (no remote, no push).
* This documentation.

## Cross-agent public surface

The cross-agent API is the union of the exports in
`moebius_finetune.contracts`. Changes to that surface go through the
contract-change flow described in `AGENTS.md` §"Public-contract change
flow".

## Reference

* TDD §3 — scaffolding plan (this document is the realised shape of
  that plan, not a redesign).
* TDD §4 — data + public interface contract (source of truth for
  shapes, dtypes, value conventions, manifest fields).
* TDD §5 — Moebius depth adapter.
* TDD §6 — student architectures.
* TDD §7 — offline one-step distillation.
* TDD §8 — budget modelling and RKNN deployment.
* TDD §9 — testing + evaluation protocol.
* TDD §10 — agent ownership and order of integration.
