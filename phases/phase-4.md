## Phase 4: Simplified Reconstruction Boundary

**Goal:** Fix the boundary direct VAE reconstruction must satisfy. Phase 3
resolves metadata; this phase defines the contract only.

```text
Input:  float32 RGB, B×3×H×W, values in [0,1]
Output: float32 RGB, B×3×H×W, identical geometry, values in [0,1]
```

**Current state**

- The planned adapter framework was removed on purpose: no adapter base class,
  interface, protocol, or stub test double exists, and none is needed.
- Reconstruction is not implemented. `vae_registry.resolve()` returns a
  `Resolved` (paths, geometry, latent metadata, hashes) or a list of `Finding`s,
  and never imports or allocates a model.
- The registry `adapter` string is retained legacy metadata: a name in
  `configs/vae_registry.toml` kept for provenance. It is not an import path and
  not a dispatch contract; no loader switches on it.
- Reconstruction stays one direct function, `Resolved` -> canonical tensor.
  Factor only if a second genuinely different implementation appears.

**Rules carried forward**

- Freeze parameters, call `eval()`, run under `torch.inference_mode()`.
- Convert canonical input to the model's documented normalization, not assumed
  identity. Add a temporal dimension only when the family needs it.
- Use posterior mode or mean, never random posterior sampling. Apply the
  model's exact direct-reconstruction latent contract. No denoiser latent
  scaling: `denoiser_scale_factor` (SD 1.5's 0.18215) is a denoiser convention.
- Decode to RGB `[0,1]`; remove only padding we added, by exact cropping, and
  reject any unexplained geometry mismatch.
- Map deprecated config classes to a current export at load (Hunyuan
  `AutoencoderKLCausal3D` -> `AutoencoderKLHunyuanVideo`).

**Exit state**

Phase 4 is complete now, at the local-resolution boundary. No adapter framework
exists, nothing here waits on one, and `resolve()` stops at local path
resolution and metadata: it never loads weights, and reports a `Finding` for
anything unsupported rather than a silent approximation. The rules above are the
recorded contract for the reconstruction path. Phase 5 owns the first
reconstruction function and its tests; no Phase 5 deliverable gates this phase.
