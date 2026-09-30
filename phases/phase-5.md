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
