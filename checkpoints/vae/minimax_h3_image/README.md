# MiniMax H3 Image VAE checkpoints

Accepted contents:

- `config.json` — architecture configuration.
- Exactly one checkpoint file.

No loader is guessed for this family. Until construction metadata is
available the registry entry stays unresolved and produces an actionable
validation error. A bare `.safetensors` with no `config.json` is rejected.
