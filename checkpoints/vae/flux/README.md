# Flux VAE checkpoints

Accepted contents:

- `config.json` — architecture configuration, including latent channels and
  shift/scale metadata.
- Exactly one checkpoint file.

A bare `.safetensors` with no `config.json` is rejected. Record the exact
revision and source repository in `metadata.json` once selected.
