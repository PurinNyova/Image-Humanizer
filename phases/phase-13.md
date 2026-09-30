## Phase 13: Inference and Release

**Goal:** Package VAE-free inference with reliable arbitrary-size handling.

Inference pipeline:

1. Apply EXIF orientation.
2. Convert to sRGB RGB.
3. Convert pixels to `[0,1]`.
4. Reflect-pad to the U-Net’s architectural multiple.
5. Run only the U-Net.
6. Remove padding exactly.
7. Clamp output and encode it.

CLI:

```text
uv run image-humanizer infer \
  --checkpoint checkpoints/dewatermarker/<run_id>/best.safetensors \
  --input input.png \
  --output output.png
```

Python interface:

```python
model = Dewatermarker.from_pretrained(path)
dewatermarked = model(watermarked)
```

Add overlapping tiled U-Net inference only when full-resolution memory demands it. Validate tiled output against full-frame output and inspect blend boundaries.

Release artifacts:

- Safetensors model weights.
- Model architecture configuration.
- Inference documentation.
- Training and dataset manifest hashes.
- Per-VAE evaluation report.
- Clean no-op report.
- Known checkpoint and dataset licenses.
- Model card documenting limitations.

**Exit criterion:** A clean environment can run inference with only the dewatermarker package and weights; no VAE dependency or checkpoint is required.
