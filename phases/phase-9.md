## Phase 9: Residual Pixel-Space U-Net Baseline

**Goal:** Implement the smallest adequate restoration model.

Recommended architecture:

```text
RGB input
→ convolution stem
→ four resolution levels
→ two residual convolution blocks per level
→ strided-convolution downsampling
→ bottleneck
→ bilinear upsampling plus convolution
→ skip connections
→ zero-initialized RGB residual head
```

Initial choices:

- Input/output channels: `3`
- Base width: `48` or `64`
- Channel multipliers: `[1, 2, 4, 8]`
- Activation: SiLU
- Normalization: GroupNorm
- No attention
- No VAE identity conditioning
- No pretrained encoder

Model behavior:

```python
predicted_watermark = unet(watermarked)
dewatermarked = watermarked - predicted_watermark
```

Public interface:

```python
dewatermarked = model(watermarked)
```

Zero-initialize the residual head so the initial model is an identity mapping.

Use the unclamped result for loss calculation when practical:

```python
raw_output = watermarked - predicted_watermark
loss = charbonnier(raw_output, original)
```

Clamp the public inference output to `[0,1]`.

**Tests:**

- Shape preservation.
- Initial identity behavior.
- Odd dimensions survive inference pad/unpad.
- One synthetic optimization step reduces loss.
- Save and reload preserves output.

**Exit criterion:** The model overfits a tiny cached subset and performs inference without importing or loading a VAE.
