# MiniMax H3 Image VAE checkpoints

Accepted contents:

- `config.json` — architecture configuration.
- `diffusion_pytorch_model.safetensors.index.json` — the index.
- All three declared shards, as one logical checkpoint with the index:
  `diffusion_pytorch_model-00001-of-00003.safetensors`,
  `diffusion_pytorch_model-00002-of-00003.safetensors`,
  `diffusion_pytorch_model-00003-of-00003.safetensors`.

A bare `.safetensors` with no `config.json` is rejected.

`enabled = false`. `config.json` declares `clip_length = 17` (trained on
17-frame clips), so `T=1` is outside the documented regime and no single-frame
path is documented: direct reconstruction is unsupported. `validate-vaes`
reports `unsupported` before reading any file.
