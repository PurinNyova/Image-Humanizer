# Image Humanizer

Learn to invert the reconstruction artifacts ("watermarks") that image and
video VAEs leave in pixel space, so a plain U-Net can undo them without
touching a VAE at inference time.

## Setup

```sh
uv sync
```

All commands are local-only. VAE weights are not downloaded automatically.

## 1. Add the original images

Put the clean originals anywhere under `data/originals/`. Nested directories
are allowed. Supported image extensions are discovered recursively.

```text
data/originals/
├── photo-001.png
├── photo-002.jpg
└── another-set/
    └── photo-003.webp
```

`data/` is ignored by Git, so originals and generated pairs are not committed.

## 2. Add and validate the VAE checkpoints

Put each local checkpoint and its config in its named directory under
`checkpoints/vae/`. The exact filenames, revisions, and hashes are documented
in `configs/vae_registry.toml`, `docs/phase0_checkpoints.md`, and each
checkpoint directory's README.

```sh
uv run image-humanizer validate-vaes
```

Do not continue until every enabled family reports `OK`. Disabled families are
skipped. Validation only checks local files; it does not load model weights.

## 3. Preprocess originals and generate VAE pairs

```sh
uv run image-humanizer preprocess --input data/originals
uv run image-humanizer generate-pairs
```

`preprocess` canonicalizes each original to 512×512, writes clean targets to
`data/processed/groups/<source_id>/original.png`, and freezes train,
validation, and test splits. `generate-pairs` then writes one reconstruction
next to each original for every enabled, valid VAE and appends those pairs to
`data/processed/manifest.jsonl`.

Pair generation is resumable: existing manifest pairs and reconstruction PNGs
are skipped. It can be rerun after adding a missing checkpoint.

## 4. Train

Review `configs/train.toml`, especially `batch_size` and `steps`, then run:

```sh
uv run image-humanizer train \
  --data data/processed \
  --config configs/train.toml \
  --out checkpoints/dewatermarker/baseline
```

Training uses CUDA when available, otherwise CPU. It writes exactly:

```text
checkpoints/dewatermarker/baseline/
├── best.safetensors
├── last.safetensors
└── model_config.json
```

The baseline trains on canonical 512×512 pairs in fp32. On a 24 GB GPU, lower
`batch_size` if memory is insufficient. The U-Net itself accepts arbitrary
geometry, including images above 1 MP; full-resolution inference is scheduled
for Phase 10 and is not yet a CLI command.

## Layout

| Path | Purpose |
|---|---|
| `configs/` | Plain TOML: VAE registry, preprocessing, training. |
| `checkpoints/vae/<family>/` | One directory per VAE family; see its README. |
| `data/originals/` | Your clean source images; ignored by Git. |
| `data/processed/` | Generated clean targets, splits, manifest, and VAE reconstructions. |
| `checkpoints/dewatermarker/<run>/` | Generated training weights; ignored by Git. |
| `src/image_humanizer/` | Package. |
| `tests/` | Tests. |

## Plan

See `plan.md` and `phases/`. Implementation follows the phase order defined
there.

## Phase 0 records

| File | Contents |
|---|---|
| `docs/phase0_checkpoints.md` | Checkpoint inventory: repo, revision, hashes, class, licensing for all ten families. |
| `docs/phase0_decisions.md` | Decision record: resolution, canonicalization, temporal policy, precision, drift limits. |

`validate-vaes` is offline and local-only: it reads this disk and never
downloads. `hf_repo` and `hf_revision` are provenance only, citing the pinned
upstream commit the local bytes should match.

Two families (`flux`, `sd3`) are gated, so nothing fetches their weights for
you — retrieve them manually into their `checkpoints/vae/<family>/`
directories at the declared revision. `minimax_h3_image` is registered and
named but `enabled = false`: its `clip_length = 17` puts `T=1` outside the
documented regime, so it fails loudly rather than guessing.
