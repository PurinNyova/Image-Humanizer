## Phase 5: One Direct Reconstruction Path

### Goal

Implement `load_vae(resolved) -> model` plus
`reconstruct(family, rgb_01, model=None) -> rgb_01` for `sd15` alone, using the
checkpoint `vae_registry.resolve()` already found. Two functions, no framework.

### Build

| File | Contents |
|---|---|
| `src/image_humanizer/reconstruct.py` | `load_vae`, `reconstruct`. |
| `tests/test_reconstruct.py` | One acceptance test. |

`load_vae(resolved) -> model` loads that `Resolved`'s `directory` and `config` as
`AutoencoderKL` with `local_files_only=True`, then `.to(float32)`,
`requires_grad_(False)`, `eval()`. A plain helper: no class, no cache.

`reconstruct` takes the canonical `float32` B×3×H×W tensor in `[0,1]` and returns
the same shape, dtype, and range. `model=None` resolves and loads for the one-off
test; Phase 6 calls `load_vae` once per family and passes `model=`. Under
`torch.inference_mode()`: normalize input to the config's documented range (SD
1.5 is `[-1,1]`); encode, take `posterior.mode()` — never `.sample()` — decode,
map back to `[0,1]`. `spatial_multiple = 8`: pad H and W up to a multiple of 8
only when needed, then crop back exactly. Any output geometry other than the
input's is an error, never a resize. A `Finding` from `resolve()` propagates
unchanged; there is no fallback path.

### Rules

- No downloads. `local_files_only=True`, no `hf_hub_download`, no repo-id load.
  Every byte comes from paths `Resolved` already names on this disk.
- No adapter class, registry, or dispatch. The registry's `adapter` string stays
  metadata; the loader reads `config.json` and does not switch on it.
- No denoiser latent scaling. `denoiser_scale_factor` (SD 1.5's 0.18215) is a
  denoiser convention and must not appear in this path.
- No tiling, no VAE slicing, no new provenance fields beyond what `Resolved`
  already carries.
- No second family. Anything but `sd15` raises `NotImplementedError` until a real
  local checkpoint needs it.

### Done

One test, `test_sd15_reconstruct`: resolve `sd15`, reconstruct a `1×3×64×64`
`[0,1]` tensor twice with `model=None` (the one-off path Phase 6 skips), and
assert output geometry, dtype, and range match the input, values are finite, and
the two calls are identical. If no sd15 checkpoint is present locally the test
`pytest.skip`s with the registry's finding code and message verbatim.

sd15 weights are **not** in the repo today — `checkpoints/vae/sd15/` holds only
`README.md` — so the test skips until they are placed locally. That skip is the
correct result, not a failure.
