# LTX VAE checkpoints

Accepted contents:

- `config.json` — architecture configuration, spatial and temporal compression.
- Exactly one checkpoint file.

Preferred: an image-retrained checkpoint. Otherwise use the supported `T=1`
encode-decode path. A bare `.safetensors` with no `config.json` is rejected.
Record the exact revision and source in `metadata.json`.
