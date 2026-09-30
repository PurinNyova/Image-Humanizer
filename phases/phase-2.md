## Phase 2: Canonical Image Pipeline and Dataset Splits

**Status: complete.** Exit criterion met — clean targets and splits are stable
and reproducible with no VAE loaded (nothing in `image_io.py` or `dataset.py`
imports `diffusers`).

### What was built

| File | Contents |
|---|---|
| `src/image_humanizer/image_io.py` | `canonicalize()` (D2 then D3, once each), `compute_phash()`, `load_canon_config()`, `save_png()`. |
| `src/image_humanizer/dataset.py` | D4 union-find grouping, deterministic split assignment, `freeze_splits()`. |
| `src/image_humanizer/cli.py` | `uv run image-humanizer preprocess`. |
| `tests/test_pipeline.py` | 26 tests: EXIF (all 8 orientations), ICC, alpha, CMYK, crop determinism, source_id, duplicate collapse, near-duplicate grouping, split stability, freeze refusal, end-to-end. |
| `configs/preprocess.toml` | Added `grouping.parent_max_fraction`. |

`data/processed/` layout follows Phase 7 so nothing moves later:
`dataset.json`, `manifest.jsonl`, `splits.json`, `groups/<source_id>/original.png`.

### Deviations from the task list, and why

1. **Pixel-identical dedup is free, not a separate pass.** D6 defines
   `source_id` as a hash of the canonical pixels, so identical pixels *are*
   identical `source_id`s. The CLI collapses on `source_id` as it decodes rather
   than running a second comparison pass.

2. **Grouping is a union of all D4 keys, not "first non-trivial wins".** The plan
   says first-key-wins; a union is strictly safer (a near-duplicate in a
   different folder still groups with its parent) and is the same amount of code.
   Group id is the lexicographically smallest member id, so it never depends on
   input order.

3. **Parent-directory grouping is capped.** A single flat directory holding the
   whole corpus is not a burst collection, and using it as a key would put 100%
   of the corpus in one split. `parent_max_fraction = 0.2` skips any parent
   covering more than 20% of sources. Raise it if real collections are that big.

4. **CMYK without an ICC profile is a hard error**, not Pillow's blind
   `CMYK -> RGB`. A wrong colour conversion would silently poison every clean
   target derived from it.

5. **`--canon-version` is a required-by-default cross-check** against
   `preprocess.toml`. A stale config would otherwise produce a corpus whose ids
   mean nothing.

6. **A changed corpus cannot silently reuse `splits.json`.** Re-running with
   added or removed sources raises instead of merging. A frozen split is a
   promise; re-splitting is a deliberate act.

### Verification beyond the unit tests

60 real JPEGs (varied sizes, two subdirectories) were run through the CLI twice
into separate output directories:

```text
splits identical:   True
manifest identical: True
clean target:       (512, 512) RGB
train/val/test:     54 / 3 / 3
```

54/3/3 is exactly 90/5/5 of 60 groups. No group crossed a split boundary.

### Carried forward

| Item | Blocks | Note |
|---|---|---|
| `manifest.jsonl` gains per-pair rows in Phase 7 | 7 | The Phase 2 row is the `original`; Phase 7 adds ten reconstruction rows sharing `group_id`. |
| pHash bucketing is O(n²) within a 16-bit band | — | Fine at current scale; BK-tree if a band ever holds thousands of images. |
| No real near-duplicate corpus available yet | — | Clustering is tested on synthetic near-duplicates only. Worth a run over the real dataset before Phase 7. |
