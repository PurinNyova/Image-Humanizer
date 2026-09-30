# Image Humanizer

Learn to invert the reconstruction artifacts ("watermarks") that image and
video VAEs leave in pixel space, so a plain U-Net can undo them without
touching a VAE at inference time.

## Setup

```sh
uv sync
```

## Usage

```sh
uv run image-humanizer --help
uv run image-humanizer validate-vaes
uv run image-humanizer preprocess --input path/to/sources
```

`preprocess` canonicalizes every source (EXIF → sRGB → alpha matte → one Lanczos
resize → one center crop at 512²), writes the clean target to
`data/processed/groups/<source_id>/original.png`, and freezes
`data/processed/splits.json`. It loads no VAE.

## Layout

| Path | Purpose |
|---|---|
| `configs/` | Plain TOML: VAE registry, preprocessing, training. |
| `checkpoints/vae/<family>/` | One directory per VAE family; see its README. |
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

Two families (`flux`, `sd3`) are gated and need `huggingface_hub login` before
their weights can be hashed. `minimax_h3_image` is registered and named but
`enabled = false`: its `clip_length = 17` puts `T=1` outside the documented
regime, so it fails loudly rather than guessing.
