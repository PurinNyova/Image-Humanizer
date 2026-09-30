# Wan VAE checkpoints

Accepted contents:

- `config.json` — architecture configuration, spatial and temporal compression.
- Exactly one checkpoint file.

Preferred: an authoritative image-retrained checkpoint. Otherwise use the
official single-frame video path with `T=1`. A bare `.safetensors` with no
`config.json` is rejected. Record the exact revision and source in
`metadata.json`.
