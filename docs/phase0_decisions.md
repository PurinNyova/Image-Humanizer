# Phase 0 — Decision Record

Every decision below is binding on later phases. Rationale is one line; the
evidence is in `phase0_checkpoints.md`. Where a decision has an implementation,
`configs/preprocess.toml` and `configs/vae_registry.toml` are the authoritative
form of it and the code reads them; this record is the reasoning, not a second
source of truth.

---

## D1. Canonical training resolution

**`512 × 512`.**

Every family in the registry divides 512 by its declared `spatial_multiple`: 8
(qwen_image, sdxl, flux, chroma, sd15, sd3, wan), 16 (minimax_h3_image —
registered but `enabled = false`), 32 (hunyuan, ltx). 512 is a common multiple of
8, 16, and 32, so a single canonical tensor feeds reconstruction for every
family without padding disagreement.

Reconstruction receives that tensor and must return identical geometry and "fail
instead of interpolating mismatched reconstruction dimensions" (Phase 4). 512 is
what makes that trivially true.

---

## D2. Image canonicalization

Order of operations, applied once per source:

1. **EXIF orientation** — `ImageOps.exif_transpose`. First, so the canonical
   geometry is the displayed geometry.
2. **sRGB conversion** — ICC profile applied via ImageCms; images with no ICC
   profile are assumed sRGB and passed through. CMYK and other non-RGB modes are
   converted to RGB **explicitly**, never silently dropped. Two cases are refused
   rather than guessed, because both have a silent wrong answer: high-bit-depth
   grayscale (`I`, `F`, `I;16`), which `convert()` would clip to 8 bits without
   saying so, and CMYK with no ICC profile, which has no defensible automatic
   conversion.
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
  deterministic.
- **Exactly once.** Resizing twice (once to normalize, once to fit a VAE) is how
  resampling artifacts get mistaken for VAE artifacts.

Images already `512×512` skip the resize entirely.

---

## D4. Grouping keys for leakage-safe splitting

**Union of every key — never a precedence order.** Each key below contributes,
and two sources that share *any* key land in the same group, transitively. So one
pair of burst frames can be held together by their shared folder while a third,
matching image stored elsewhere is pulled in by pHash. The earlier rule — one key
chosen per group, the remaining keys ignored — is withdrawn.
`configs/preprocess.toml` records this as `grouping.strategy = "union_all_keys"`,
and `dataset.assign_groups` builds one union-find over all of them.

1. **Byte hash** — identical file SHA-256. Byte-identical copies are dropped
   before they are ever decoded, and unioned again at grouping time.
2. **Canonical source id** — pixel-identical sources (the same image re-encoded as
   JPEG/PNG) hash to the same `source_id` under D6, so they are already one
   source before grouping starts. Listed because it is a key, not because it
   needs a separate pass.
3. **Perceptual near-duplicate** — pHash Hamming distance ≤
   `grouping.phash_max_distance` (6) between two sources that share a pHash band
   (4 bands of 16 bits). Catches crops, resizes, and mild re-compression.
4. **Parent directory** — directory relative to the input root, which catches
   bursts, video frames, and shot sequences that share a folder. A directory
   holding more than `grouping.parent_max_fraction` (0.2) of the corpus is
   treated as a flat dump rather than a collection, so it is not used as a key.

A group's id is the lexicographically smallest `source_id` in the component, so
it never depends on scan order. Splits are assigned by group, never by generated
pair: all crops and all reconstructions of one source stay in one split. A
written `splits.json` is a promise — a `canon_version` mismatch or any difference
in the source, group, or parent map aborts rather than silently re-splitting.

---

## D5. Reference-generation precision

**fp32 for all reference generation. Tiling disabled. No mixed precision.**

Rationale: the dataset *is* the measurement. fp16 or bf16 encode-decode would
inject quantization error that is indistinguishable from the VAE's own
reconstruction error, and the model would learn to remove our noise instead of
the VAE's signature. This is the one place where "just use fp16" is wrong.

Cost: LTX at 512×512 with 128 latent channels and Hunyuan at 32× spatial
compression are the memory risks. Both are handled by keeping one VAE resident
at a time, which Phase 6 already mandates.

Training runs in fp32 — Phase 9 states "fp32 only" — and **no phase as written
enables mixed precision.** Enable it only against a measured need (fp32
throughput or memory failure), and record that measurement here first.

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
| minimax_h3_image | video | `unsupported` (`enabled = false`) |

An image-retrained checkpoint is preferred wherever one is authoritative and
available. Research found **none** for Wan, Hunyuan, or LTX — Wan-AI,
Tencent, and Lightricks publish video VAEs only. Qwen Image is the exception:
its VAE is natively 2D and needs no fallback.

**`minimax_h3_image` is deliberately unsupported.** Its config declares
`clip_length: 17`, meaning it is trained on 17-frame clips. `T=1` is outside
that regime, and unlike Wan/Hunyuan/LTX there is no documented single-frame
path to fall back on. It stays in the registry as a *named, disabled* entry —
`enabled = false` with an `unsupported_reason` — so `validate-vaes` reports one
`unsupported` finding and pair generation skips it by name. Reported, never
silently approximated, never deleted from the inventory.

---

## D8. Maximum acceptable clean-image drift

The evaluation gate in Phase 10: per-family metrics on the test split, with
these limits holding for every family.

**Hard limits on clean inputs:**

| Metric | Limit |
|---|---|
| `L1(model(clean), clean)` | ≤ `0.002` |
| PSNR | ≥ `48 dB` |

**Selection rule: the worst family decides.** Both limits hold for every
enabled family and no mean is computed. A checkpoint that improves the average
while breaking either limit on one family is rejected.

SSIM, edge energy, and changed-pixel share are **withdrawn.** Phase 10 reports
exactly paired L1/PSNR and clean-drift L1/PSNR — no metric battery, no SSIM, no
edge reporting.

Rationale for the numbers: a model that alters a clean image by more than ~0.2%
mean absolute error is no longer "removing a watermark", it is "editing
everything". These thresholds are deliberately strict; loosen them only with
evidence from Phase 10, and record why.

The limits live in `configs/preprocess.toml` (`[clean_drift]`) and are recorded
here as the gate for reference generation and evaluation. No code reads those keys
yet — they are pinned input, not a passing check.

---

## D9. Dependency pins

`pyproject.toml` states the floors (`torch>=2.6`, `diffusers>=0.33`, and so on);
**`uv.lock` is the authoritative record of what a run actually resolved.** Read
the version from there. This document deliberately duplicates no version
numbers, because a copied table is a table that goes stale silently.

What is load-bearing about `diffusers` is not a number but two class facts,
verified by import against the version locked at the time of writing (0.40.0):
it exports `AutoencoderKLQwenImage`, `AutoencoderKLWan`,
`AutoencoderKLHunyuanVideo`, `AutoencoderKLLTXVideo`, `AutoencoderKLMiniMaxH3`,
and `AutoencoderKL`, all with `enable_tiling` and `enable_slicing`; and it does
**not** export the deprecated `AutoencoderKLCausal3D` that Hunyuan's config
declares — hence the required remap at load time. Re-verify both facts whenever
the lock moves; nothing in the codebase checks them for you.

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

**Gated repos:** `flux` and `sd3` are gated on the Hub, and nothing in this
project gets around that. `validate-vaes` resolves local files only: it never
logs in, never opens a socket, and never downloads, and `hf_repo`/`hf_revision`
are provenance citations that no code path reads. The operator accepts the
upstream terms out of band and places the pinned files in
`checkpoints/vae/<family>/` by hand.

Until a real weight hash is known, the registry declares `sha256 = ""` for both
entries. That is not a failure: resolution reports the file as **UNVERIFIED** in
a deferred note. Filling in the true LFS hash is what upgrades the entry to
enforced — the enforcement is the registry field, nothing else.

**Derived-work note:** VAE reconstructions are outputs of these models. A model
trained on SD3 reconstructions inherits a practical claim on them. Any released
artifact must state this explicitly.

---

## Open items carried forward

| Item | Blocks | Resolution phase |
|---|---|---|
| Local weights for all 9 enabled families — registry pins them, operator places them | 5, 6 | manual, deliberate |
| Gated fetch of `flux` + `sd3` (operator accepts upstream terms; no login in code) | 6 | manual, deliberate |
| `minimax_h3_image` `T=1` verification | 6 | 6 |
| Hunyuan `.pt` + deprecated class handling | — | closed: both declared in the registry, class remap happens at load |
| Wan 2.2 `in_channels: 12` — excluded | — | closed, not selected |
| Chroma-vs-Flux weight hash equality | — | closed, architecture confirmed / weights unverifiable |
