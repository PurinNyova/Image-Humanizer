# Phased Implementation Plan

## Phase 0: Pin Dataset and Checkpoint Decisions

**Goal:** Resolve choices that affect every later phase.

Tasks:

- Select the initial canonical training resolution, recommended `512×512`.
- Define image canonicalization:
  - Apply EXIF orientation.
  - Convert ICC-managed images to sRGB.
  - Composite alpha over a configured background.
  - Represent pixels as float32 RGB in `[0,1]`.
- Decide source resize and crop policy.
- Identify grouping keys for videos, bursts, subjects, artists, products, or near-duplicate collections.
- Select an exact checkpoint revision for every VAE.
- Record licensing and redistribution restrictions.
- Determine whether each video-derived VAE will use:
  - An image-retrained checkpoint, preferred when authoritative and available.
  - Its officially supported single-frame encode-decode path otherwise.
- Choose reference-generation precision and whether VAE tiling is allowed.
- Define a maximum acceptable clean-image drift for model selection.

**Deliverables:**

- Checkpoint inventory covering all ten VAE families.
- Decision record for image processing and temporal handling.
- Initial dependency versions to pin in `uv.lock`.

**Exit criterion:** Every VAE has a named candidate checkpoint and documented construction source. Unknown standalone `.safetensors` files are not considered sufficient.

---

## Phase 1: Minimal Repository Scaffold

**Goal:** Establish the package, configuration, and required local directory structure without implementing model-specific behavior.

Proposed structure:

```text
Image-Humanizer/
├── pyproject.toml
├── uv.lock
├── README.md
├── configs/
│   ├── vae_registry.toml
│   ├── preprocess.toml
│   └── train.toml
├── checkpoints/
│   ├── vae/
│   │   ├── qwen_image/
│   │   ├── sdxl/
│   │   ├── flux/
│   │   ├── chroma/
│   │   ├── sd15/
│   │   ├── sd3/
│   │   ├── wan/
│   │   ├── minimax_h3_image/
│   │   ├── hunyuan/
│   │   └── ltx/
│   └── dewatermarker/
├── src/image_humanizer/
│   ├── __init__.py
│   ├── image_io.py
│   ├── vae_registry.py
│   ├── vae_adapters.py
│   ├── generate_pairs.py
│   ├── dataset.py
│   ├── model.py
│   ├── losses.py
│   ├── train.py
│   ├── evaluate.py
│   └── infer.py
└── tests/
```

Tasks:

- Configure a Python/PyTorch project managed through `uv`.
- Ignore checkpoint binaries, generated datasets, runs, and temporary files.
- Keep a short `README.md` in every VAE checkpoint directory describing accepted contents.
- Define plain TOML configuration and `argparse` entry points.
- Avoid Hydra, Lightning, plugin discovery, and one-file-per-adapter scaffolding until needed.

**Exit criterion:** The package imports, commands expose help, and all required checkpoint directories are documented.

---

## Phase 2: Canonical Image Pipeline and Dataset Splits

**Goal:** Establish exact target pixels and leakage-safe splits before generating any reconstructions.

Tasks:

- Decode each source exactly once.
- Apply EXIF orientation.
- Convert color-managed images to sRGB.
- Convert grayscale, CMYK, and other supported modes explicitly to RGB.
- Define alpha compositing behavior rather than silently dropping alpha.
- Resize once with a pinned filter.
- Crop once before invoking any VAE.
- Assign a stable `source_id` from source content and canonicalization version.
- Detect byte-identical and pixel-identical duplicates.
- Cluster perceptual near-duplicates before splitting.
- Split by source or parent group, never by generated pair.
- Keep all crops and VAE reconstructions from one source in one split.
- Freeze the resulting `splits.json`.

Recommended split:

```text
train: 90%
validation: 5%
test: 5%
```

**Tests:**

- EXIF orientation fixtures.
- ICC and alpha fixtures.
- Deterministic crop generation.
- Duplicate groups never cross split boundaries.
- The same source always receives the same split.

**Exit criterion:** Clean targets and split assignments are stable and reproducible without loading a VAE.

---

## Phase 3: VAE Registry and Checkpoint Discovery

**Goal:** Resolve checkpoints and architecture metadata safely and predictably.

Registry fields should cover:

```toml
[vae.sdxl]
directory = "checkpoints/vae/sdxl"
adapter = "diffusers_autoencoder_kl"
checkpoint = "diffusion_pytorch_model.safetensors"
config = "config.json"
hf_fallback = "stabilityai/stable-diffusion-xl-base-1.0"
hf_subfolder = "vae"
enabled = true
```

Selection precedence:

1. Explicit command-line checkpoint override.
2. Registry `checkpoint` field.
3. The only `.safetensors` candidate in the local directory.
4. Clear failure if multiple candidates remain.
5. Hugging Face fallback only when explicitly enabled.

Validation must check:

- Local directory existence.
- Selected checkpoint existence.
- Architecture configuration availability.
- Adapter-specific metadata.
- Model class compatibility.
- Missing and unexpected checkpoint keys.
- Spatial and temporal configuration.
- Latent scaling, shift, mean, and standard deviation.
- Optional fallback repository and revision.
- Checkpoint and configuration hashes.

Provide a read-only validation command:

```text
uv run image-humanizer validate-vaes
```

Example failure:

```text
VAE 'wan' contains wan_vae.safetensors, but no compatible architecture
configuration was found. Provide config.json or configure an explicit
Hugging Face architecture fallback.
```

**Tests:**

- Explicit checkpoint selection.
- Single-file automatic selection.
- Multiple-file ambiguity.
- Missing configuration.
- Disabled fallback behavior.
- Path traversal rejection.

**Exit criterion:** Every registry entry either resolves unambiguously or produces a precise corrective error before allocating substantial GPU memory.

---

## Phase 4: Common VAE Adapter Contract

**Goal:** Normalize genuinely different VAE implementations behind one narrow interface.

External contract:

```text
Input:  float32 RGB, B×3×H×W, values in [0,1]
Output: float32 RGB, B×3×H×W, identical geometry, values in [0,1]
```

Conceptual interface:

```python
class VAEAdapter:
    def load(self, checkpoint, device, dtype): ...
    def required_spatial_multiple(self) -> int: ...
    def reconstruct(self, rgb_01): ...
    def provenance(self) -> dict: ...
```

Every adapter should:

- Freeze parameters and call `eval()`.
- Run under `torch.inference_mode()`.
- Convert canonical input to the model’s documented normalization.
- Add a temporal dimension only when required.
- Use posterior mode or mean, never random posterior sampling.
- Apply the model’s exact direct reconstruction latent contract.
- Decode and return RGB `[0,1]`.
- Remove only adapter-owned padding through exact cropping.
- Reject any unexplained geometry mismatch.
- Record checkpoint, precision, tiling, normalization, and temporal provenance.

Do not assume that a diffusion pipeline’s denoiser latent scaling should also be applied during direct VAE reconstruction. Verify this separately for each family.

**Tests:**

- Shape and range preservation.
- Finite output.
- Deterministic repeated output.
- Posterior sampling disabled.
- Correct normalization.
- Correct pad/unpad behavior.
- Useful checkpoint mismatch errors.

**Exit criterion:** A synthetic adapter proves the complete registry-to-reconstruction contract independently of large real checkpoints.

---

## Phase 5: Native 2D VAE Adapters

**Goal:** Implement the lower-risk image-native adapters first.

Implementation order:

1. SD 1.5 / LDM
2. SDXL
3. Flux
4. SD3
5. Chroma

Key requirements:

- **SD 1.5:** support a pinned Diffusers layout first; add original LDM key remapping only for a checkpoint that actually requires it.
- **SDXL:** use its configured `AutoencoderKL`; do not substitute an arbitrary SD 1.5 VAE.
- **Flux:** preserve the checkpoint’s latent channel and shift/scale metadata.
- **SD3:** construct from its actual architecture configuration and latent conventions.
- **Chroma:** verify whether the selected checkpoint reuses a Flux-compatible VAE; compare config and hashes rather than assuming compatibility.

For each adapter, record:

- Model class.
- Required spatial multiple.
- Input and output range.
- Latent transform behavior.
- Precision.
- Tiling configuration.
- Checkpoint and config hash.

**Tests:**

- One small deterministic real-image reconstruction per adapter.
- Exact output geometry.
- Golden output tolerance in a pinned environment.
- Tiled mode treated as a separate reconstruction configuration.

**Exit criterion:** All five adapters produce deterministic, aligned reconstructions from named pinned checkpoints.

---

## Phase 6: Specialized and Temporal VAE Adapters

**Goal:** Add the remaining adapters without guessing their temporal or architecture behavior.

Implementation order:

1. Qwen Image
2. Wan
3. Hunyuan
4. LTX
5. MiniMax H3 Image

Per-family policy:

| VAE | Preferred path | Fallback path |
|---|---|---|
| Qwen Image | Official image reconstruction path | Supported single-frame spatiotemporal path |
| Wan | Authoritative image-retrained checkpoint | Official single-frame video path |
| Hunyuan | Authoritative image-retrained checkpoint | Official single-frame video path |
| LTX | Image-retrained checkpoint | Supported `T=1` encode-decode path |
| MiniMax H3 Image | Dedicated image checkpoint and authoritative architecture | No guessed loader; remain unresolved until construction metadata is available |

Temporal handling must record:

```text
checkpoint_variant: image_retrained | video
reconstruction_method: native_image | supported_single_frame
temporal_input_frames: 1
temporal_padding: ...
output_frame_selection: ...
```

For single-frame paths:

- Add `T=1` in the dimension expected by the model.
- Follow documented temporal padding or causal behavior.
- Decode using the corresponding supported path.
- Extract the documented output frame.
- Assert no unexplained temporal duplication or frame shift.

**Tests:**

- Temporal dimension placement.
- Frame count before and after decode.
- Determinism.
- Image-retrained versus single-frame provenance.
- Correct spatial and temporal compression constraints.

**Exit criterion:** Each adapter is verified against a named checkpoint, or explicitly remains unsupported with an actionable validation error. No architecture is inferred from an arbitrary weight file.

---

## Phase 7: Offline Group Generation

**Goal:** Generate the complete eleven-image group for every clean source.

Expected group:

```text
original
├── Qwen reconstruction
├── SDXL reconstruction
├── Flux reconstruction
├── Chroma reconstruction
├── SD 1.5 reconstruction
├── SD3 reconstruction
├── Wan reconstruction
├── MiniMax H3 Image reconstruction
├── Hunyuan reconstruction
└── LTX reconstruction
```

Recommended storage:

```text
data/processed/
├── dataset.json
├── manifest.jsonl
├── splits.json
└── groups/<source_id>/
    ├── original.png
    ├── qwen_image.png
    ├── sdxl.png
    ├── flux.png
    ├── chroma.png
    ├── sd15.png
    ├── sd3.png
    ├── wan.png
    ├── minimax_h3_image.png
    ├── hunyuan.png
    ├── ltx.png
    └── metadata.json
```

Generation behavior:

- Give every adapter the same canonical target tensor.
- Use lossless PNG initially.
- Write to a temporary group directory.
- Validate all enabled reconstructions.
- Atomically publish only complete groups.
- Resume using hashes and metadata.
- Treat checkpoint, adapter version, precision, and tiling as dataset identity.
- Create contact sheets and amplified residual previews for an audit subset.
- Fail instead of interpolating mismatched reconstruction dimensions.

Manifest rows should represent individual training pairs while retaining a shared `group_id`.

**Caching decision:** Use cached reconstructions for production. On-the-fly generation should be limited to adapter smoke tests with one VAE at a time.

**Exit criterion:** A representative subset regenerates deterministically and passes shape, color, split, completeness, and hash checks.

---

## Phase 8: Dataset Loader and Balanced Sampling

**Goal:** Expose cached reconstructions as balanced pixel-space training pairs.

Each sample returns:

```text
watermarked: VAE reconstruction
original: clean target
vae_name: evaluation and sampling metadata
group_id: source group identity
```

Sampling strategy:

1. Choose a VAE family uniformly.
2. Choose a valid original uniformly for that VAE.
3. Load the reconstruction and clean target.

This avoids accidental dominance caused by failed groups, duplicate rows, or unequal availability.

Training augmentation must preserve alignment:

- Apply the same crop and flip to input and target.
- Avoid independent color augmentation.
- Avoid arbitrary color augmentation that changes the clean target relationship.
- Keep validation and test deterministic.

Do not load all ten reconstructions in every baseline batch. Group batches are only needed if group consistency is later enabled.

**Tests:**

- Input and target remain aligned.
- Split boundaries are respected.
- Sampling frequencies are approximately uniform.
- Corrupt or incomplete groups fail clearly.
- Every training pair uses decoded pixels, not latent tensors.

**Exit criterion:** The loader produces balanced, aligned batches without loading any VAE.

---

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

---

## Phase 10: Baseline Training

**Goal:** Establish a reproducible baseline with one paired objective.

Baseline loss:

```text
Charbonnier(dewatermarked, original)
```

Do not add a duplicate residual objective.

Training components:

- AdamW.
- Mixed precision after float32 smoke tests pass.
- Gradient clipping.
- Batch accumulation where necessary.
- Deterministic validation.
- Balanced per-VAE sampling.
- Full-state resume.
- Periodic visual comparison panels.
- Aggregate and per-VAE metric logging.

Checkpoint contents:

```text
checkpoints/dewatermarker/<run_id>/
├── best.safetensors
├── last.safetensors
├── optimizer.pt
├── trainer_state.json
├── model_config.json
├── train_config.toml
├── dataset_manifest_hash.txt
└── metrics.jsonl
```

Select checkpoints using equal weighting across VAE families, subject to a clean-image drift limit.

**Exit criterion:** A fully reproducible baseline run completes, resumes correctly, and exports inference-only Safetensors weights.

---

## Phase 11: Evaluation and Clean-Image Safety

**Goal:** Measure restoration by VAE family while detecting unnecessary modification of legitimate content.

Per-VAE metrics:

- Input and output L1 or Charbonnier.
- PSNR before and after.
- SSIM before and after.
- Optional LPIPS.
- Percentage of images improved.
- Worst-decile performance.

Aggregate metrics:

- Equal-weight mean across VAE families.
- Worst VAE family.
- Worst original group.
- Variance among outputs from the same original.
- Improvement consistency across VAE types.

Clean-input metrics:

- `L1(model(clean), clean)`
- Clean PSNR and SSIM.
- Color drift percentiles.
- Edge-energy change.
- Fraction of pixels changed above practical thresholds.
- Amplified predicted residuals.

Clean evaluation sets should include:

- Camera photographs.
- Illustrations.
- Text-heavy images.
- JPEG-compressed clean images.
- Resized clean images.
- Out-of-domain dimensions and aspect ratios.

**Exit criterion:** The baseline improves each target VAE family without exceeding the agreed clean-image drift threshold.

---

## Phase 12: Optional Regularizers

**Goal:** Add only the minimum correction justified by baseline evidence.

Experiment order:

1. **Clean-input identity**
2. **Edge preservation**
3. **SSIM**
4. **Group consistency**

Clean identity should be the first response to excessive clean-image changes:

```text
input = clean
target = clean
loss = same Charbonnier objective
```

Mix a small identity fraction, initially around 5-10%, and evaluate the restoration-versus-no-op tradeoff.

Only add edge preservation if legitimate high-frequency detail is being smoothed. Only add SSIM if structural metrics or visual inspection justify it. Group consistency requires loading several reconstructions from one group and should be deferred unless per-VAE outputs diverge materially.

Run one regularizer change at a time.

**Exit criterion:** Retain a regularizer only if it improves held-out behavior without hiding degradation in any VAE family or increasing clean-image drift.

---

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

---

## Phase Dependencies

```text
Phase 0
  ↓
Phase 1
  ↓
Phase 2 ─────────────┐
  ↓                  │
Phase 3              │
  ↓                  │
Phase 4              │
  ↓                  │
Phases 5 and 6       │
  └──────────┬───────┘
             ↓
          Phase 7
             ↓
          Phase 8
             ↓
          Phase 9
             ↓
         Phase 10
             ↓
         Phase 11
             ↓
         Phase 12
             ↓
         Phase 13
```

Phases 5 and 6 can proceed in parallel after the common adapter contract is stable. Full dataset generation should wait until every required adapter is validated.
