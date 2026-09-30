## Phase 10: Evaluate and Use the Trained Model

**Goal:** The project ends here — measure the selected `best.safetensors` and run it on real images.

**Build:** Two `cli.py` subcommands. Both load `best.safetensors` plus the adjacent `model_config.json` from one run directory; neither touches a VAE.

```text
evaluate --run checkpoints/dewatermarker/<run_id> --data data/processed
infer    --run checkpoints/dewatermarker/<run_id> --input in.png --output out.png
```

`evaluate` reads `data/processed/manifest.jsonl` directly, keeps the rows whose `split`
is `test`, and walks each row's `reconstruction`/`clean_target` pair with its `family` in
hand — `PairDataset` returns bare tensors, and its per-family cycling would revisit
minority families. It reports, per family and overall:

```text
l1         = mean |dewatermarked - original|
psnr       = -10*log10(mse) between dewatermarked and original
drift_l1   = mean |dewatermarked(clean) - clean|
drift_psnr = -10*log10(mse) between dewatermarked(clean) and clean
```

Drift is the canonical `original.png` fed straight to the model and scored against itself.
Both drift metrics come from that one clean-input pass, so nothing gates an unmeasured
number.

`infer` handles one image:

- canonicalize via `image_io.canonicalize` with `configs/preprocess.toml`;
- reflect-pad to a multiple of 8, run the model, crop back;
- clamp only the final tensor to `[0,1]`;
- write with `image_io.save_png`.

**Rules:**

- Metrics are exactly paired L1/PSNR and clean-input drift L1/PSNR. No LPIPS, no metric battery, no SSIM or edge reporting, and no further `[clean_drift]` keys beyond the two thresholds named below.
- `preprocess.toml [clean_drift]` is the acceptance gate: `l1_max` on `drift_l1`, `psnr_min_db` on `drift_psnr`. The worst family decides, not the mean; paired L1/PSNR are reported, not gated.
- No regularizer, retraining, or checkpoint selection here; the checkpoint is already chosen.
- One image. No tiling, no batching, no multi-image path.
- Load `safetensors` directly: no model hub, no `from_pretrained`, no model card, no release bundle, no service or deployment code.

**Done:**

- `infer` on one test image is byte-identical to a direct `model(canonicalize(path))` call in the same process, at canonical 512×512 geometry.
- Drift L1 is at or below `clean_drift.l1_max` and drift PSNR at or above `clean_drift.psnr_min_db` for every family in the test split; one family over either limit fails the command.
- One watermarked test image round-trips to a readable PNG.

This phase completes the goal. Extra metrics, regularizers, packaging, or deployment require a measured failure first.
