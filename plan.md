# Image-Humanizer — Master Plan

Index only. Detailed specs live in `phases/phase-N.md`.

## Goal

Local-only dewatermarking: canonical source images → cached VAE reconstruction
pairs → one trained residual model → evaluated, then run on real images to
produce a PNG. One model, one trainer, no network at runtime.

## Principles

- Local checkpoints only; a missing checkpoint is an error, never a download.
- No automatic downloads.
- No adapter/plugin framework, no handler registry, no factory.
- Simplest fixed model and straight-line trainer.
- Add anything only after a measured need, one change at a time.

## Status

| Phase | Status | What it delivers |
|---|---|---|
| [0](phases/phase-0.md) Pin Dataset and Checkpoint Decisions | complete | Canonical image form, source identity, grouping, checkpoint provenance. |
| [1](phases/phase-1.md) Minimal Repository Scaffold | complete | Package, TOML config, `uv` project, checkpoint directories. |
| [2](phases/phase-2.md) Canonical Image Pipeline and Dataset Splits | complete | Deterministic preprocessing and frozen leakage-safe splits. |
| [3](phases/phase-3.md) VAE Registry and Checkpoint Discovery | complete | Local checkpoint and config resolution, returning findings, allocating no model. |
| [4](phases/phase-4.md) Simplified Reconstruction Boundary | complete (boundary) | Reconstruction contract fixed; the adapter framework was deleted on purpose, reconstruction is not implemented. |
| [5](phases/phase-5.md) One Direct Reconstruction Path | complete | One `reconstruct()` function for `sd15` from an already-resolved checkpoint. |
| [6](phases/phase-6.md) Reconstruction Pairs for Every Enabled Family | complete | Cached per-family pair images and an append-only manifest. |
| [7](phases/phase-7.md) Pair Dataset | complete | `PairDataset` yielding balanced aligned pairs, importing no VAE. |
| [8](phases/phase-8.md) Residual Pixel-Space U-Net | complete | One residual 4-level U-Net, identity at initialization. |
| [9](phases/phase-9.md) Baseline Training Run | complete | One AdamW run, one Charbonnier objective, best and last weights. |
| [10](phases/phase-10.md) Evaluate and Use the Trained Model | planned | Per-family metrics on the test split, and PNG inference. |

## Definition of done

Not yet reached. Phase 10 is planned. Every bullet below is a future
completion criterion, not a statement of what works today.

Precondition: the operator places every required local checkpoint — weights
and config, per the paths from phase 3 — on disk before phase 5 may run.
Nothing here downloads them, and a missing checkpoint is an error, not a
fallback.

The project is done only when all of the following hold:

- Preprocessing runs locally and writes canonical sources.
- Training pairs are generated locally from resolved checkpoints.
- One training run produces a model.
- Evaluation reports per-family results and clean-image drift.
- Inference writes an output PNG from the trained weights alone.
