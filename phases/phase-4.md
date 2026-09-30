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
