# Phase 0 — Checkpoint Inventory

Ten VAE families. Every entry names a source repository, a pinned commit SHA
from the Hugging Face API, the exact file paths, and the architecture class.

**Provenance is record-only.** The repo and revision below are a citation of
which upstream commit the local bytes should match. `configs/vae_registry.toml`
carries them as `hf_repo` + `hf_revision` + `hf_subfolder`, and no code path
reads them: `validate-vaes` resolves local files only, never logs in, and never
downloads. Placing a family's files in `checkpoints/vae/<family>/` is a manual,
deliberate operator step.

Provenance of facts below:

- **VERIFIED** — fetched from `huggingface.co/api/models/<repo>` or a
  `huggingface.co/<repo>/raw/<rev>/<path>` request on 2026-09-30.
- **COMFYUI** — read from a local `ComfyUI` checkout at commit fetched
  2026-09-30 (`comfy/sd.py`, `comfy/latent_formats.py`).
- **UNCONFIRMED** — could not verify. Not usable as a pinned decision.

**VERIFIED here does not mean validated locally.** A hash in this document is
what was read from the Hub on the date above. It becomes *enforced* only when it
is written into the registry as a non-empty `sha256` (or
`checkpoint_shard_sha256`, for shards) — the registry field is the only thing
`validate-vaes` checks. Entries the registry ships with `sha256 = ""` are
reported **UNVERIFIED** at validation time, deliberately, rather than silently
passing.

All diffusers classes referenced below exist in the diffusers version resolved
in `uv.lock` (0.40.0 at the time of writing) and all expose both `enable_tiling`
and `enable_slicing` (VERIFIED by import). The one exception is load-time
remapping, not a missing class: see `hunyuan`.

---

## 1. `qwen_image`

| Field | Value | Status |
|---|---|---|
| Repo | `Qwen/Qwen-Image` | VERIFIED |
| Revision | `75e0b4be04f60ec59a75f475837eced720f823b6` | VERIFIED |
| Config | `vae/config.json` | VERIFIED |
| Weights | `vae/diffusion_pytorch_model.safetensors` | VERIFIED |
| Weight SHA-256 (LFS) | `0c8bc8b758c649abef9ea407b95408389a3b2f610d0d10fcb054fe171d0a8344` | VERIFIED |
| Class | `AutoencoderKLQwenImage` | VERIFIED (config `_class_name`) |
| Latent channels | 16 | VERIFIED |
| Spatial / temporal | 8 / 4 | COMFYUI (`QwenImage` uses `Wan21` latent format; ComfyUI `vae.py` `dim_mult=[1,2,4,4]`, `temperal_downsample=[F,T,T]`) |
| Latent normalization | `latents_mean` / `latents_std`, 16 values each, baked into config | VERIFIED |
| License | Apache-2.0 | VERIFIED |
| Gated | No | VERIFIED |

**Reconstruction method: `native_image`.** This is a genuinely 2D-capable VAE — it is
the image model of the Qwen-Image family and needs no temporal fudging.

> Note: `latents_mean`/`latents_std` here are numerically identical to the Wan
> 2.1 constants in `Wan-AI/Wan2.1-T2V-1.3B-Diffusers/vae/config.json`. The
> architectures are close relatives. Recorded so Phase 6 does not "discover" this
> and treat it as a bug.

---

## 2. `sdxl`

| Field | Value | Status |
|---|---|---|
| Repo | `stabilityai/stable-diffusion-xl-base-1.0` | VERIFIED |
| Revision | `462165984030d82259a11f4367a4eed129e94a7b` | VERIFIED |
| Config | `vae/config.json` | VERIFIED |
| Weights | `vae/diffusion_pytorch_model.safetensors` | VERIFIED |
| Weight SHA-256 (LFS) | `1598f3d24932bcfe6634e8b618ea1e30ab1d57f5aad13a6d2de446d2199f2341` | VERIFIED |
| Class | `AutoencoderKL` | VERIFIED |
| Latent channels | 4 | VERIFIED |
| Spatial / temporal | 8 / n/a | COMFYUI |
| Denoiser latent scale | `0.13025` | COMFYUI (`latent_formats.SDXL`) |
| License | OpenRAIL++ | VERIFIED |
| Gated | No | VERIFIED |

**Reconstruction method: `native_image`.** Use the configured `AutoencoderKL`. Do not
substitute `madebyollin/sdxl-vae-fp16-fix` — it is a community fp16 cast whose
`config.json` was hand-edited, and our references run in fp32.

An fp16 variant `vae/diffusion_pytorch_model.fp16.safetensors` also exists in
the repo. Do not select it; it is a different file and would need its own hash.

---

## 3. `flux`

| Field | Value | Status |
|---|---|---|
| Repo | `black-forest-labs/FLUX.1-schnell` | VERIFIED |
| Revision | `741f7c3ce8b383c54771c7003378a50191e9efe9` | VERIFIED |
| Config | `vae/config.json` | **BLOCKED — repo is gated** |
| Weights | `vae/diffusion_pytorch_model.safetensors` | **BLOCKED — repo is gated** |
| Weight SHA-256 | — | **not recorded — repo is gated. Registry declares `sha256 = ""`, so validation reports the local file UNVERIFIED** |
| Class | `AutoencoderKL` | COMFYUI + Chroma config (see Chroma, below) |
| Latent channels | 16 | VERIFIED (via Chroma's copy of this config) |
| Spatial / temporal | 8 / n/a | COMFYUI |
| Denoiser latent transform | `(z - 0.1159) * 0.3611` | COMFYUI (`latent_formats.Flux`) |
| License | Apache-2.0 | VERIFIED |
| Gated | **Yes** (`gated: "auto"`) | VERIFIED |

**UNCONFIRMED: the exact weight hash and the raw Flux `config.json` bytes.**

I could not fetch the Flux VAE files: the repo is gated and this environment has
no HF token. What *is* verified is that Chroma ships a VAE whose `config.json`
declares `_name_or_path: "/home/ubuntu/FLUX.1-schnell"` with
`scaling_factor: 0.3611` and `shift_factor: 0.1159` — see `chroma` below.

**Consequence:** the pinned revision and file path are known, but the content
hash is not, so the registry ships `sha256 = ""` for this family. That is the
designed state, not a failure: once the operator has fetched the gated weights
by hand at the pinned revision, `validate-vaes` resolves the entry and reports
the weight file **UNVERIFIED** in a deferred note. Writing the true LFS hash into
`configs/vae_registry.toml` is what makes it enforced. Validation itself never
authenticates and never fetches.

---

## 4. `chroma`

| Field | Value | Status |
|---|---|---|
| Repo | `lodestones/Chroma1-HD` | VERIFIED |
| Revision | `0e0c60ece1e82b17cb7f77342d765ba5024c40c0` | VERIFIED |
| Config | `vae/config.json` | VERIFIED |
| Weights | `vae/diffusion_pytorch_model.safetensors` | VERIFIED |
| Weight SHA-256 (LFS) | `f5b59a26851551b67ae1fe58d32e76486e1e812def4696a4bea97f16604d40a3` | VERIFIED |
| Class | `AutoencoderKL` | VERIFIED |
| Latent channels | 16 | VERIFIED |
| Spatial / temporal | 8 / n/a | COMFYUI (`Chroma.latent_format is Flux`) |
| License | Apache-2.0 | VERIFIED |
| Gated | No | VERIFIED |

**Chroma is Flux-derived, and we can now say by how much.** Phase 5 asked to
"verify whether the selected checkpoint reuses a Flux-compatible VAE; compare
config and hashes rather than assuming compatibility." Evidence:

- Chroma's `vae/config.json` sets `"_name_or_path": "/home/ubuntu/FLUX.1-schnell"`.
- Its architecture block is byte-for-byte the standard Flux VAE block:
  `block_out_channels [128,256,512,512]`, `layers_per_block 2`,
  `norm_num_groups 32`, `mid_block_add_attention true`,
  `use_post_quant_conv false`, `use_quant_conv false`, `latent_channels 16`,
  `scaling_factor 0.3611`, `shift_factor 0.1159`, `sample_size 1024`,
  `force_upcast true`.

So the **architecture is confirmed identical** to Flux's. What is **not** proven
is that the **weights** are the same file — Chroma's VAE is 167,666,902 bytes,
and we cannot hash Flux's to compare because Flux is gated. Treat as
"Flux-compatible architecture, weights unverified" and record both hashes
separately in provenance.

---

## 5. `sd15`

| Field | Value | Status |
|---|---|---|
| Repo | `stabilityai/sd-vae-ft-ema` | VERIFIED |
| Revision | `f04b2c4b98319346dad8c65879f680b1997b204a` | VERIFIED |
| Config | `config.json` (repo root, not a subfolder) | VERIFIED |
| Weights | `diffusion_pytorch_model.safetensors` | VERIFIED |
| Weight SHA-256 (LFS) | `32db726da04f06c1b6b14c0043ce115cc87a501482945c5add89a40d838fcb46` | VERIFIED |
| Class | `AutoencoderKL` | VERIFIED |
| Latent channels | 4 | VERIFIED |
| Spatial / temporal | 8 / n/a | COMFYUI |
| Denoiser latent scale | `0.18215` | COMFYUI (`latent_formats.SD15`) |
| License | MIT | VERIFIED |
| Gated | No | VERIFIED |

Note the config has **no** `scaling_factor` key at all. The `0.18215` value is a
*denoiser* convention, not a VAE config field — precisely the confusion Phase 4
warns about. Direct VAE reconstruction applies no scaling.

The repo also contains a legacy `diffusion_pytorch_model.bin`. Only the
safetensors file is accepted.

---

## 6. `sd3`

| Field | Value | Status |
|---|---|---|
| Repo | `stabilityai/stable-diffusion-3-medium-diffusers` | VERIFIED (repo exists) |
| Revision | `ea42f8cef0f178587cf766dc8129abd379c90671` | VERIFIED |
| Config | `vae/config.json` | VERIFIED (file is listed) |
| Weights | `vae/diffusion_pytorch_model.safetensors` | VERIFIED (file is listed) |
| Weight SHA-256 | — | **not recorded — repo is gated. Registry declares `sha256 = ""`, so validation reports the local file UNVERIFIED** |
| Class | `AutoencoderKL` | COMFYUI (`AutoencoderKL` path; `latent_channels 16`) |
| Latent channels | 16 | COMFYUI |
| Spatial / temporal | 8 / n/a | COMFYUI |
| Denoiser latent transform | `(z - 0.0609) * 1.5305` | COMFYUI (`latent_formats.SD3`) |
| License | Stability AI Non-Commercial Research Community | VERIFIED |
| Gated | **Yes** (`gated: "auto"`) | VERIFIED |

**UNCONFIRMED: weight hash, and the exact `config.json` contents.**

Note `stabilityai/stable-diffusion-3-medium` (the non-`-diffusers` repo) has
**no `vae/` directory at all** — it ships only single-file `.safetensors`. Only
`stable-diffusion-3-medium-diffusers` has the split layout. That distinction is
recorded because it is an easy and silent mistake.

Both `.safetensors` and `.safetensors.fp16.safetensors` exist; the tree listing
shows the same `oid` for both, i.e. they are the same content in different
containers. Use the plain one.

**Access, not tooling.** Because the repo is gated, the weight hash is not
recorded and the registry ships `sha256 = ""`. The operator accepts the SD3 terms
and places the files in `checkpoints/vae/sd3/` by hand; validation then resolves
the entry and reports the weight **UNVERIFIED** rather than failing it, and logs
in to nothing.

Licensing note: the SD3 weights are **non-commercial**. This constrains any
redistribution of a trained dewatermarker whose dataset derives from them.

---

## 7. `wan`

Two genuinely different candidates exist. They are **not** interchangeable —
different latent dimensionality and different spatial compression.

### 7a. Wan 2.1 (z_dim 16, 8x spatial)

| Field | Value | Status |
|---|---|---|
| Repo | `Wan-AI/Wan2.1-T2V-1.3B-Diffusers` | VERIFIED |
| Revision | `0fad780a534b6463e45facd96134c9f345acfa5b` | VERIFIED |
| Config | `vae/config.json` | VERIFIED |
| Weights | `vae/diffusion_pytorch_model.safetensors` | VERIFIED |
| Weight SHA-256 (LFS) | `d6e524b3fffede1787a74e81b30976dce5400c4439ba64222168e607ed19e793` | VERIFIED |
| Class | `AutoencoderKLWan` | VERIFIED |
| Latent channels | 16 | VERIFIED (`z_dim`) |
| Spatial / temporal | 8 / 4 | COMFYUI |
| Latent normalization | `latents_mean` / `latents_std` (16 values) in config | VERIFIED |
| License | Apache-2.0 | VERIFIED |
| Gated | No | VERIFIED |

Used as the canonical `wan` family entry. The 1.3B repo is chosen over the 14B
repo purely because both ship the **same** VAE, and the smaller repo makes the
config diff trivial.

### 7b. Wan 2.2 TI2V-5B (z_dim 48, 16x spatial)

| Field | Value | Status |
|---|---|---|
| Repo | `Wan-AI/Wan2.2-TI2V-5B-Diffusers` | VERIFIED |
| Revision | `b8fff7315c768468a5333511427288870b2e9635` | VERIFIED |
| Config | `vae/config.json` | VERIFIED |
| Weights | `vae/diffusion_pytorch_model.safetensors` | VERIFIED |
| Weight SHA-256 (LFS) | `62cd18f19438e35b32ac63020e2852f566e9b02f46b6cdbd87972a356e3c6f4b` | VERIFIED |
| Class | `AutoencoderKLWan` | VERIFIED |
| Latent channels | **48** | VERIFIED (`z_dim`) |
| `in_channels`/`out_channels` | **12 / 12** | VERIFIED — an intermediate YUV-like representation, not RGB |
| Spatial / temporal | **16** / 4 | VERIFIED (`scale_factor_spatial`) |
| Extra config | `patch_size 2`, `is_residual true`, `decoder_base_dim 256` | VERIFIED |
| License | Apache-2.0 | VERIFIED |
| Gated | No | VERIFIED |

The `in_channels: 12` is the important part. This VAE does not take RGB at its
interface in the ordinary way, so it does **not** satisfy the Phase 4 external
contract without a documented colour transform we have not verified.
**Not selected.** Recorded so Phase 6 does not treat it as an equivalent swap.

**Reconstruction method for `wan`: `supported_single_frame`.** No authoritative
image-retrained Wan VAE exists; every Wan VAE published by Wan-AI is the video
model. (UNCONFIRMED-negative: I searched Wan-AI's full model list, 30+ repos,
and found no image-only VAE release.)

---

## 8. `hunyuan`

| Field | Value | Status |
|---|---|---|
| Repo | `tencent/HunyuanVideo` | VERIFIED |
| Revision | `6204ad6aea1a77ff5aba337c88278bb9500eb37d` | VERIFIED |
| Config | `hunyuan-video-t2v-720p/vae/config.json` | VERIFIED |
| Weights | `hunyuan-video-t2v-720p/vae/pytorch_model.pt` | VERIFIED |
| Weight SHA-256 (LFS) | `95d1fc707c1421ccd88ea542838ab4c5d45a5babb48205bac9ce0985525f9818` | VERIFIED |
| Config `_class_name` | `AutoencoderKLCausal3D` | VERIFIED |
| Modern class | `AutoencoderKLHunyuanVideo` | VERIFIED (exists in diffusers 0.40.0) |
| Latent channels | 16 | VERIFIED |
| Spatial / temporal | 32 / 4 | COMFYUI (`HunyuanVideo`: 32x spatial via `ffactor_spatial: 32`; `temporal_downscale_ratio 4`) |
| Denoiser latent scale | `0.476986` | VERIFIED (in `config.json` as `scaling_factor`) |
| License | Tencent custom | VERIFIED (`license: other`) |
| Gated | No | VERIFIED |

**Three traps, all recorded:**

1. The weights are `.pt`, **not safetensors** (VERIFIED above). The Phase 1
   scaffold assumed `.safetensors` in the checkpoint README rule and in registry
   selection, so a `.pt`-only checkpoint could not be picked up by auto-discovery.
   Resolved since: the registry now names `pytorch_model.pt` explicitly, which
   takes precedence over the `.safetensors` auto-discovery fallback;
   `checkpoints/vae/hunyuan/README.md` records the same.
2. The config declares the **deprecated** class `AutoencoderKLCausal3D`
   (`_diffusers_version 0.4.2`). diffusers 0.40.0 does not export that name;
   loading requires mapping it to `AutoencoderKLHunyuanVideo`.
3. The VAE lives under a **resolution-coded subfolder**
   (`hunyuan-video-t2v-720p/`). The repo root has no `vae/`.

**Reconstruction method: `supported_single_frame`.** No authoritative image-retrained
Hunyuan VAE found. Note `tencent/HunyuanImage-3.0` exists and is a different,
image-native model with its own architecture, but its repo ships **no `vae/`
directory** — the VAE is embedded in the transformer. It is therefore not a
usable checkpoint for this project as published. (VERIFIED: `vae does not exist
on "main"`.)

---

## 9. `ltx`

| Field | Value | Status |
|---|---|---|
| Repo | `Lightricks/LTX-Video` | VERIFIED |
| Revision | `8984fa25007f376c1a299016d0957a37a2f797bb` | VERIFIED |
| Config | `vae/config.json` | VERIFIED |
| Weights | `vae/diffusion_pytorch_model.safetensors` | VERIFIED |
| Weight SHA-256 (LFS) | `265ca87cb5dff5e37f924286e957324e282fe7710a952a7dafc0df43883e2010` | VERIFIED |
| Class | `AutoencoderKLLTXVideo` | VERIFIED |
| Latent channels | 128 | VERIFIED |
| Spatial / temporal | 32 / 8 | VERIFIED (`spatio_temporal_scaling`) |
| `scaling_factor` | 1.0 | VERIFIED |
| Asymmetry | `encoder_causal: true`, `decoder_causal: false` | VERIFIED |
| License | LTX-Video Open Weights License 0.X | VERIFIED |
| Gated | No | VERIFIED |

**Reconstruction method: `supported_single_frame`.** No authoritative image-retrained
LTX VAE was found; Lightricks publishes video VAEs only. The
`encoder_causal / decoder_causal` asymmetry means the single-frame path is the
*only* asymmetry-free option at `T=1` — ComfyUI explicitly works around a
"first latent = 1 pixel frame" asymmetry for multi-frame guides
(`comfy_extras/nodes_lt.py`), which confirms `T=1` is the sane case.

Successor candidates exist and are **not** interchangeable:
`Lightricks/LTX-2` (rev `dfcc2108383fe1aaa0584bdf55d368a4bdadd90c`) uses class
`AutoencoderKLLTX2Video` with `decoder_num_layers`-style blocks,
`spatial_compression_ratio 32`, `temporal_compression_ratio 8`. Different
architecture. Pinned to LTX-Video; revisit only deliberately.

---

## 10. `minimax_h3_image`

| Field | Value | Status |
|---|---|---|
| Repo | `MiniMaxAI/MiniMax-H3` | VERIFIED |
| Revision | `42ed227ee7df40d41602854ae760620d6eb651fe` | VERIFIED |
| Config | `vae/config.json` | VERIFIED |
| Weights | `vae/diffusion_pytorch_model-00001/2/3-of-00003.safetensors` + `vae/diffusion_pytorch_model.safetensors.index.json` | VERIFIED |
| Weight SHA-256 (LFS) | shard 1 `72f4c6be84ac0674f27398cde991dd9d719762f3952c4921aa66b2ce542f6374`<br>shard 2 `2e05e8bc23fa4071043e17fd242be8acd0685e781a43987432b2eae925be4198`<br>shard 3 `c05d6ac4b1a33de372799d708531da6320f6a3ce6d1ce6d895e770988e004a39` | VERIFIED (read from the Hub; **not enforced — see below**) |
| Class | `AutoencoderKLMiniMaxH3` | VERIFIED |
| Latent channels | 24 | VERIFIED |
| Spatial / temporal | 16 / 4 | VERIFIED (`spatial_downsample_factors [2,2,2,2,1,1]`, `temporal_downsample_factors [1,2,2,1,1,1]`) |
| Decoder | 36-layer ViT, 32 heads × 64 dim, RoPE θ=100 | VERIFIED |
| Extra | `clip_length: 17`, `token_drop: 3`, `spatial_padding_mode: reflect` | VERIFIED |
| License | MiniMax H3 Community License | VERIFIED (`license: other`) |
| Gated | No | VERIFIED |

**This family is NOT unresolved, but it is not an image VAE either.** Phase 0's
premise was that no construction metadata was available. That is now false: the
architecture is fully specified in `vae/config.json` and diffusers ships the
loader. What is true instead:

- MiniMax H3 publishes **only a video VAE**. There is no H3 *Image* VAE release
  to prefer. (`MiniMax-H3-Image` does not exist as a separate VAE release.)
- The legacy config at `FL2VA/video_vae/source/config.json` declares class
  `AutoencoderKLLegacy` with `vae_ratio 16`, `vae_ratio_t 4`,
  `pixel_norm_type: imagenet`, `use_vit_decoder: true` — i.e. the same model in
  its pre-diffusers form. Recorded for key-remapping work, not selected.
- **`clip_length: 17` is a hard constraint.** The VAE is trained on 17-frame
  clips (1 + 4×4). Feeding `T=1` is outside the documented regime. Unlike Wan,
  Hunyuan, and LTX, this family has **no verified single-frame path**.
- Weights are **sharded** (3 shards + index). All four files are required, and
  the registry treats them as one logical checkpoint: `checkpoint` names the
  index, `checkpoint_shards` names the three shards, and the index's
  `weight_map` must resolve to exactly that set.
- **The three shard hashes above are not enforced.** They are recorded here as
  Hub-read facts, but the registry declares no `checkpoint_shard_sha256` and
  leaves `sha256 = ""`, so nothing compares a shard against them — the files
  would land in the UNVERIFIED list. Recording them per shard is the change that
  makes them enforced, and it matters more here than for a single-file
  checkpoint, where a partial or swapped download is much easier to spot.
- This entry is `enabled = false`, so resolution short-circuits to the
  `unsupported` finding before any file or hash is examined at all. Both points
  above are what will apply on the day it is enabled.

**Decision: `minimax_h3_image` is registered and named but disabled —
`enabled = false` — and remains unsupported for reference generation until the
`T=1` behavior is verified against a real clip.** It must fail loudly in
`validate-vaes`, not guess.

---

## Summary

| Family | Repo | Rev | Class | Z | S/T | License | Gated | Status |
|---|---|---|---|---|---|---|---|---|
| qwen_image | `Qwen/Qwen-Image` | `75e0b4be` | `AutoencoderKLQwenImage` | 16 | 8/4 | Apache-2.0 | no | Ready |
| sdxl | `stabilityai/stable-diffusion-xl-base-1.0` | `46216598` | `AutoencoderKL` | 4 | 8/— | OpenRAIL++ | no | Ready |
| flux | `black-forest-labs/FLUX.1-schnell` | `741f7c3c` | `AutoencoderKL` | 16 | 8/— | Apache-2.0 | **yes** | Gated, manual fetch; hash UNVERIFIED |
| chroma | `lodestones/Chroma1-HD` | `0e0c60ec` | `AutoencoderKL` | 16 | 8/— | Apache-2.0 | no | Ready |
| sd15 | `stabilityai/sd-vae-ft-ema` | `f04b2c4b` | `AutoencoderKL` | 4 | 8/— | MIT | no | Ready |
| sd3 | `stabilityai/stable-diffusion-3-medium-diffusers` | `ea42f8ce` | `AutoencoderKL` | 16 | 8/— | Non-Commercial | **yes** | Gated, manual fetch; hash UNVERIFIED |
| wan | `Wan-AI/Wan2.1-T2V-1.3B-Diffusers` | `0fad780a` | `AutoencoderKLWan` | 16 | 8/4 | Apache-2.0 | no | Ready |
| hunyuan | `tencent/HunyuanVideo` | `6204ad6a` | `AutoencoderKLHunyuanVideo` | 16 | 32/4 | Tencent | no | Ready, needs remap |
| ltx | `Lightricks/LTX-Video` | `8984fa25` | `AutoencoderKLLTXVideo` | 128 | 32/8 | LTX OW 0.X | no | Ready |
| minimax_h3_image | `MiniMaxAI/MiniMax-H3` | `42ed227e` | `AutoencoderKLMiniMaxH3` | 24 | 16/4 | H3 Community | no | Registered, `enabled = false`; shard hashes unenforced |

"Ready" means the registry entry is complete and its weight hash is declared, so
a local file is checked against a pinned value. It says nothing about whether the
file is on this disk — today none are, and every enabled family reports
`missing_checkpoint`.

**Exit criterion status: MET**, with the gaps stated rather than hidden. `flux`
and `sd3` are gated, so the operator fetches them manually at the pinned revision
and their registry `sha256` is empty: validation reports those files UNVERIFIED
and never authenticates to fix it. `minimax_h3_image` is registered and named but
`enabled = false`, and MiniMax H3's three shard hashes are recorded above without
being declared in the registry, so they are not enforced either.

Every VAE has a named candidate checkpoint and a documented construction
source. No entry relies on an anonymous `.safetensors`.
