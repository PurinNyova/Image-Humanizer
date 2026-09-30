## Phase 8: Residual Pixel-Space U-Net

**Goal:** Map a VAE reconstruction to the clean original in pixel space.

**Build:** One residual 4-level convolutional U-Net in `model.py`.

```text
3ch RGB → Conv3x3 stem (64ch)
→ levels [64, 128, 256, 512]: 2× (GroupNorm → SiLU → Conv3x3) per level
→ downsample (stride-2 conv) between levels, upsample (bilinear + Conv3x3) on return
→ skip connections
→ Conv3x3 → zero-init (3ch)
→ add input
```

Widths and multipliers come from `configs/train.toml` (`base_width = 64`, `channel_multipliers = [1, 2, 4, 8]`). GroupNorm uses 8 groups. Forward returns the corrected pixels directly, no clamping.

**Rules:**

- Input and output are RGB float32 tensors with identical geometry.
- Reflect-pad H and W up to a multiple of 8, run the net, crop back to the original size.
- Zero-initialize the 3-channel residual head so the first forward is the identity map.
- Forward returns unclamped corrected pixels; clamping happens at inference only.
- No attention, no conditioning, no pretrained encoders, no model registry, no config class, no factory, no second architecture.

**Tests** (`tests/test_model.py`):

- Identity initialization: `model(x) == x` for random `x` at init.
- Odd-size geometry: `x` of shape `(2, 3, 37, 53)` returns exactly that shape.
- One synthetic optimization step: a single AdamW step on a fixed pair lowers the Charbonnier loss. The test inlines it as `torch.sqrt((model(x) - y) ** 2 + 1e-6).mean()` — no losses module, no helper, nothing to import. Phase 9 owns the production objective.

**Done:** `model.py` defines one class, tests pass, no VAE import anywhere in the path.
