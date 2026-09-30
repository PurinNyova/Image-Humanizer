# Chroma VAE checkpoints

Accepted contents:

- `config.json` — architecture configuration.
- Exactly one checkpoint file.

Whether this family reuses a Flux-compatible VAE must be proven by comparing
configs and hashes, not assumed. A bare `.safetensors` with no `config.json`
is rejected. Record the exact revision and source in `metadata.json`.
