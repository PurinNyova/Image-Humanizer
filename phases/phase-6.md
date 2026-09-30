# Phase 6: Reconstruction Pairs for Every Enabled Family

**Goal:** Make the Phase 5 reconstruct function the cached training dataset: one
PNG and one manifest row per enabled, locally resolved family per source.

**Build:** One loop in `src/image_humanizer/generate_pairs.py`, reached by
`uv run image-humanizer generate-pairs --output data/processed`, plus
`load_png` beside `save_png` in `image_io.py`. Family is the outer loop, so one
VAE serves a whole family. Ids, groups, and splits come from `manifest.jsonl`.

```text
for report in validate_registry():               # family outer
    if not report.ok: print(findings); continue   # disabled or unresolved
    vae = load_vae(report.resolved)               # Phase 5 loader, one per family
    for row in manifest:                          # source inner
        chw = load_png(row.clean_target)           # CHW float32 [0,1]
        bchw = reconstruct(report.name, chw[None], model=vae)  # BCHW in, BCHW out
        save_png(bchw[0], groups/<source_id>/<family>.png)     # batch dropped
        append(row, family=report.name, reconstruction_sha256=file_sha256(bchw[0]))
    del vae; empty_cache()

# one row per pair: group_id, source_id, split, family, clean_target, reconstruction, reconstruction_sha256
```

**Rules:**

- Ranks: `load_png` returns CHW `float32` `[0,1]`, the rank `save_png` writes;
  `reconstruct` is BCHW. Phase 6 adds the batch dimension before the call and
  drops it before `save_png`, and nothing else touches C, H, W, dtype, or range.
- Reuse the Phase 5 `reconstruct`, extended by one argument, `model=`, so one
  `load_vae(report.resolved)` serves a whole family: float32 `[0,1]` in,
  identical geometry out, no tiling, no denoiser latent scaling, no posterior
  sampling, no new layer, adapter base class, handler registry, or 2nd path.
- Family differences are two branches on `Resolved`, never a hierarchy:
  `native_image` is the plain path; `supported_single_frame` adds `T=1` plus
  documented padding and frame selection; an unexported `_class_name` maps to
  the registry `adapter` name, metadata, not an import path.
- Disabled or unresolved families are printed and skipped — `minimax_h3_image`
  is disabled, and anything `resolve()` finds findings for is skipped with that
  text. Nothing is downloaded, substituted, or approximated.
- One VAE resident at a time: `del` plus `empty_cache()` after each family.
- The manifest is appended, never rewritten, and `clean_target` stays the
  source row's existing path: a family whose PNG already exists is neither
  regenerated nor re-appended — one row per (`source_id`, `family`).

**Out of scope:** contact sheets, per-group `metadata.json`, atomic group
publishing, a resume engine, dataset identity/versioning, fallback tables.

**Tests** (`tests/test_generate_pairs.py` — the acceptance check):

- One fixture source and a hand-written source row in a temporary tree;
  `model=` is ignored by a one-line fake, so no weights are needed.
- After one run: the PNG matches what was reconstructed, and the manifest
  gained exactly one row with those seven fields, same `group_id` and `split`.
- A second run adds no row and leaves the source row unchanged; a disabled and
  an unresolved family add no file and no row, only a line naming why.

**Done:** one reconstruction per enabled resolved family per source, and a
manifest that is append-only and duplicate-free.
