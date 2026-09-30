# Phase 0 — Decision Record

Every decision below is binding on later phases. Rationale is one line; the
evidence is in `phase0_checkpoints.md`.

---

## D1. Canonical training resolution

**`512 × 512`.**

Every VAE in the inventory divides 512 by its spatial compression: 8 (sd15, sdxl,
flux, chroma, sd3, qwen_image, wan), 16 (wan2.2 — not selected), 32 (hunyuan,
ltx). 512 is a common multiple of 8, 16, and 32, so a single canonical tensor
feeds all ten adapters without padding disagreement.

Phase 7 requires "give every adapter the same canonical target tensor" and to
"fail instead of interpolating mismatched reconstruction dimensions". 512 is what
makes that trivially true.

---

## D2. Image canonicalization

Order of operations, applied once per source:

1. **EXIF orientation** — `ImageOps.exif_transpose`. First, so the canonical
   geometry is the displayed geometry.
2. **sRGB conversion** — ICC profile applied via ImageCms; images with no ICC
   profile are assumed sRGB and passed through. CMYK and other non-RGB modes are
   converted to RGB **explicitly**, never silently dropped.
3. **Alpha compositing** — over the configured background, default white
   `[1.0, 1.0, 1.0]`. Alpha is never dropped.
4. **Represent as float32 RGB in `[0, 1]`**, NCHW.

Rationale for white: VAEs are trained on photographic and web imagery where white
is the common matte. Black would inject a false edge at every alpha boundary and
teach the model to remove a matte that is not there.

`canon_version = "1"`. Any change to this pipeline increments it, which
invalidates every `source_id` (see D6).

---

## D3. Resize and crop policy

**Resize shorter side to 512 with Lanczos, then center-crop 512×512. One resize,
one crop, before any VAE is touched.**

- Lanczos, not bicubic. Lanczos is the correct reconstruction kernel for
  downsampling; bicubic's ringing is itself a high-frequency artifact and would
  contaminate the watermark training signal.
- Center crop, not random. Cropping is part of dataset identity, so it must be
  deterministic. Random crops are an *augmentation* applied at training time
  (Phase 8), identically to input and target.
- **Exactly once.** Resizing twice (once to normalize, once to fit a VAE) is how
  resampling artifacts get mistaken for VAE artifacts.

Images already `512×512` skip the resize entirely.

---

## D4. Grouping keys for leakage-safe splitting

Applied in this order; the first that produces a non-trivial grouping wins.

1. **Byte-identical** — identical file SHA-256.
2. **Pixel-identical** — identical post-canonicalization tensor hash. Catches
   the same image re-encoded as JPEG/PNG.
3. **Perceptual near-duplicate** — pHash Hamming distance ≤ 6 within the same
   pHash band. Catches crops, resizes, and mild re-compression.
4. **Parent collection** — directory or archive-of-origin. Catches bursts,
   video frames, and shot sequences that share a folder.
5. **Fallback: `source_id` alone** as its own group.

All crops and all ten VAE reconstructions from one source stay in one split.
Splits are by group, never by generated pair — Phase 2 already states this; D4
just names the keys.

---

## D5. Reference-generation precision

**fp32 for all reference generation. Tiling disabled. No mixed precision.**

Rationale: the dataset *is* the measurement. fp16 or bf16 encode-decode would
inject quantization error that is indistinguishable from the VAE's own
reconstruction error, and the model would learn to remove our noise instead of
the VAE's signature. This is the one place where "just use fp16" is wrong.

Cost: LTX at 512×512 with 128 latent channels and Hunyuan at 32× spatial
compression are the memory risks. Both are handled by moving one adapter to GPU
at a time, which Phase 7 already mandates.

Mixed precision is permitted **for the U-Net in Phase 10**, after float32
smoke tests pass.

---

## D6. Stable `source_id`

```
source_id = sha256( canonicalized_pixel_bytes || canon_version )
```

Content-derived, not path-derived, so moving a file does not change its identity
and identical content under two paths collapses to one source. `canon_version`
is mixed in so that a canonicalization change invalidates every id.

---

## D7. Video-derived VAE policy

| Family | `checkpoint_variant` | `reconstruction_method` |
|---|---|---|
| qwen_image | image (native) | `native_image` |
| sdxl / sd15 / flux / chroma / sd3 | image (native) | `native_image` |
| wan | video | `supported_single_frame` |
| hunyuan | video | `supported_single_frame` |
| ltx | video | `supported_single_frame` |
| minimax_h3_image | video | **`UNSUPPORTED`** |

An image-retrained checkpoint is preferred wherever one is authoritative and
available. Research found **none** for Wan, Hunyuan, or LTX — Wan-AI,
Tencent, and Lightricks publish video VAEs only. Qwen Image is the exception:
its VAE is natively 2D and needs no fallback.

**`minimax_h3_image` is deliberately unsupported.** Its config declares
`clip_length: 17`, meaning it is trained on 17-frame clips. `T=1` is outside
that regime, and unlike Wan/Hunyuan/LTX there is no documented single-frame
path to fall back on. It stays in the registry as a *named, resolved* checkpoint
that fails loudly, rather than being silently skipped or guessed at.

---

## D8. Maximum acceptable clean-image drift

Used for checkpoint selection in Phase 10 and as the safety gate in Phase 11.

**Hard limits on clean inputs:**

| Metric | Limit |
|---|---|
| `L1(model(clean), clean)` | ≤ `0.002` |
| PSNR | ≥ `48 dB` |
| SSIM | ≥ `0.995` |
| Edge energy change | ≤ `1%` |
| Pixels changed by > 1/255 | ≤ `0.5%` |

**Selection rule:** equal-weight mean across the ten VAE families, subject to
every one of these holding. A checkpoint that improves the mean while violating
any clean limit on any single family is rejected.

Rationale for the numbers: a model that alters a clean image by more than ~0.2%
mean absolute error is no longer "removing a watermark", it is "editing
everything". These thresholds are deliberately strict; loosen them only with
evidence from Phase 11, and record why.

---

## D9. Dependency pins

From `uv.lock`, resolved 2026-09-30:

| Package | Version |
|---|---|
| `torch` | 2.14.0 |
| `torchvision` | 0.29.0 |
| `diffusers` | 0.40.0 |
| `numpy` | 2.x |
| `pillow` | 11.x |
| `safetensors` | 0.5+ |
| `transformers` | 4.x |
| `accelerate` | 1.x |

`diffusers==0.40.0` is load-bearing: it is the version verified to export
`AutoencoderKLQwenImage`, `AutoencoderKLWan`, `AutoencoderKLHunyuanVideo`,
`AutoencoderKLLTXVideo`, and `AutoencoderKLMiniMaxH3`, all with `enable_tiling`
and `enable_slicing`. It does **not** export the deprecated
`AutoencoderKLCausal3D` that Hunyuan's config declares — hence the required
remap in Phase 6.

Re-lock only deliberately. A lockfile change invalidates every provenance record
that includes a dependency hash.

---

## D10. Licensing and redistribution

Blocking, non-blocking, or restrictive per family:

| Family | License | Effect |
|---|---|---|
| qwen_image | Apache-2.0 | none |
| sdxl | OpenRAIL++ | use-based restrictions on redistribution |
| sd15 | MIT | none |
| flux | Apache-2.0 | none (but repo is **gated** — access terms, not license) |
| chroma | Apache-2.0 | none |
| sd3 | Stability Non-Commercial | **restrictive.** No commercial use of a dewatermarker whose training data derives from these reconstructions. |
| wan | Apache-2.0 | none |
| hunyuan | Tencent custom | review before release |
| ltx | LTX Open Weights 0.X | review before release |
| minimax_h3_image | H3 Community | review before release |

**Gated repos:** `flux` and `sd3` require `huggingface_hub login` before
validation can hash their weights. This blocks their Phase 3 exit criterion
until a token is present, and it is a licensing/access question, not a
technical one.

**Derived-work note:** VAE reconstructions are outputs of these models. A model
trained on SD3 reconstructions inherits a practical claim on them. Release
artifacts in Phase 13 must state this explicitly.

---

## Open items carried forward

| Item | Blocks | Resolution phase |
|---|---|---|
| HF token for `flux` + `sd3` | 3 (validation) | 3 |
| `minimax_h3_image` `T=1` verification | 6, 7 | 6 |
| Hunyuan `.pt` + deprecated class handling | 3, 6 | 3 |
| Wan 2.2 `in_channels: 12` — excluded | — | closed, not selected |
| Chroma-vs-Flux weight hash equality | — | closed, architecture confirmed / weights unverifiable |
