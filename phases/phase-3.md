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
