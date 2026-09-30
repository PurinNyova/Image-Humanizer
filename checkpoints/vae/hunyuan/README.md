# Hunyuan VAE checkpoints

Accepted contents:

- `config.json` — architecture configuration, spatial and temporal compression.
  It declares the deprecated `AutoencoderKLCausal3D`, which diffusers does not
  export; it is mapped to `AutoencoderKLHunyuanVideo` at load.
- `pytorch_model.pt` — the declared checkpoint. The upstream repo ships `.pt`
  only, so the registry names it explicitly rather than resolving a
  `.safetensors`.

A checkpoint with no `config.json` is rejected. Preferred: an authoritative
image-retrained checkpoint. Otherwise use the official single-frame video path
with `T=1`. Record the exact revision and source in `metadata.json`.
