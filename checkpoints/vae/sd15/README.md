# Stable Diffusion 1.5 VAE checkpoints

Accepted contents:

- `config.json` — architecture configuration.
- Exactly one checkpoint file.

Diffusers layout is supported first; original LDM key remapping is added only
for a checkpoint that actually requires it. A bare `.safetensors` with no
`config.json` is rejected. Record the exact revision and source in
`metadata.json`.
