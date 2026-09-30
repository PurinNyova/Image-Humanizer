## Phase 2: Canonical Image Pipeline and Dataset Splits

**Status: complete.** Exit criterion met — clean targets and splits are stable
and reproducible with no VAE loaded (nothing in `image_io.py` or `dataset.py`
imports `diffusers`).

### What was built

| File | Contents |
|---|---|
| `src/image_humanizer/image_io.py` | `canonicalize()` (one decode, then D2 then D3 exactly once each), `compute_phash()`, `load_canon_config()`, `save_png()`. |
| `src/image_humanizer/dataset.py` | D4 `union_all_keys` grouping, deterministic split assignment, `freeze_splits()`. |
| `src/image_humanizer/cli.py` | `uv run image-humanizer preprocess`. |
| `tests/test_pipeline.py` | EXIF (all 8 orientations), ICC, alpha and palette transparency, high-bit refusal, CMYK refusal, crop determinism, `source_id`, duplicate collapse, near-duplicate grouping, split stability, freeze refusal, end-to-end CLI. |
| `configs/preprocess.toml` | `grouping.strategy = "union_all_keys"`, `grouping.parent_max_fraction`. |

`data/processed/` layout follows Phase 6 so nothing moves later:
`dataset.json`, `manifest.jsonl`, `splits.json`, `groups/<source_id>/original.png`.
One `original.png` per source today; Phase 6 adds reconstruction rows sharing
`group_id`.

### Grouping, refusals, and drift

1. **Grouping is `union_all_keys`.** Sources sharing *any* D4 key — byte hash,
   pHash within `phash_max_distance`, or a parent directory — merge into one
   group, so a near-duplicate in a different folder still groups with its
   parent. Group id is the lexicographically smallest member id, so it never
   depends on input order.

2. **Parent-directory grouping is capped.** A single flat directory holding the
   whole corpus is not a burst collection, and using it as a key would put 100%
   of the corpus in one split. `parent_max_fraction = 0.2` skips any parent
   covering more than 20% of sources.

3. **Unsupported sources are refused, not guessed.** CMYK *without* an ICC
   profile raises, and CMYK carrying a profile LittleCMS cannot build a
   transform from raises as well — Pillow's blind `CMYK -> RGB` would silently
   poison every clean target derived from it. High-bit grayscale (`I`, `F`,
   `I;16`) is refused too, because Pillow's `convert()` would clip it to 8 bits
   silently. CMYK is only ever converted through a real ICC transform.

4. **Pixel-identical dedup is free.** D6 defines `source_id` as a hash of the
   canonical pixels, so identical pixels *are* identical `source_id`s. The CLI
   collapses on `source_id` as it decodes rather than running a second
   comparison pass.

5. **Corpus and canon-version drift are rejected, never merged.**
   `--canon-version` must match `preprocess.toml`; a mismatch aborts the run.
   An existing `splits.json` is authoritative: a differing `canon_version` (which
   means every `source_id` changed) or any differing `source_to_split`,
   `group_to_split`, or `parent_to_split` mapping — including a corpus that grew
   or shrank — raises instead of re-splitting. A frozen split is a promise;
   re-splitting is a deliberate act.

### Verification

The suite covers the above plus a CLI end-to-end run. 60 real JPEGs (varied
sizes, two subdirectories) were also run through the CLI twice into separate
output directories:

```text
splits identical:   True
manifest identical: True
clean target:       (512, 512) RGB
train/val/test:     54 / 3 / 3
```

54/3/3 is exactly 90/5/5 of 60 groups. No group crossed a split boundary.
