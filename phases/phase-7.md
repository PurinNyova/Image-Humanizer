## Phase 7: Pair Dataset

### Goal

Expose the cached reconstructions as training pairs: one `PairDataset` in the
existing `dataset.py`, reading only `data/processed/` PNGs plus `manifest.jsonl`
and `splits.json`. No VAE import, no generated pixel.

### Build

| File | Contents |
|---|---|
| `src/image_humanizer/dataset.py` | `PairDataset(Dataset)`, `__len__`, `__getitem__`. |
| `tests/test_dataset.py` | One acceptance test. |

`PairDataset(processed_dir: Path, split: str)`:
- At init, read `manifest.jsonl` and keep only rows whose `group_id` maps to `split`
  in `splits.json` `group_to_split`. A row whose `original.png` or family PNG is not on
  disk is incomplete: drop that row, silently. Partial loss only shrinks its family.
  Group survivors by family, sorted for determinism. Only a family with zero surviving
  complete pairs for the split raises `ValueError` naming the split and the family.
- `__len__` = `len(families) * max(len(sources) per family)`. `__getitem__(i)` takes
  `f = i % len(families)` and `sources[f][(i // len(families)) % len(sources[f])]`.
  Families are uniform by construction; each family cycles through its own complete
  sources. `DataLoader` shuffles the flat index; there is no sampler.
- Returns `{"reconstruction": t_recon, "original": t_orig}`, `float32` 3×H×W in
  `[0, 1]`, read from PNG through `numpy`/`torch` as `save_png` writes it.

### Rules

- No VAE import, no `reconstruct` call, no on-the-fly generation.
- No augmentation, custom sampler, grouped batch, cache, `DataModule`, or plugin hook.
  Standard `torch.utils.data.Dataset` and `DataLoader` only.
- Return exactly those two tensors. `family_name` and `group_id` stay internal;
  `group_to_split` is the only metadata read, used solely to filter the split. Phase 10
  per-family evaluation walks `manifest.jsonl` itself, so no metadata is returned here.
- Reconstruction and original must already share geometry: no resize or crop.
- The frozen `splits.json` is authoritative; a changed `canon_version` raises rather
  than re-splitting silently.

### Done

One test, `test_pair_dataset`, over a temporary two-family manifest:
- Shape/range: a train and a validation item are `float32` 3×H×W, finite, in `[0, 1]`, same H, and `reconstruction is not original`.
- Split isolation: no train `group_id` appears in the validation or test dataset.
- Family coverage: each family appears `max sources` times per epoch, every source once.
- Missing pair: delete one family PNG of a two-source family — no raise, that family contributes one source instead of two. Empty a family — that raises `ValueError` naming the split and the family.

Skipped: `vae_name` for per-family eval. Add when a trainer must attribute a loss.
