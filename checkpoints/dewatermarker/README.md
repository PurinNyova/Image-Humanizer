# Dewatermarker runs

Training outputs live here, one directory per `run_id`:

```text
checkpoints/dewatermarker/<run_id>/
├── best.safetensors
├── last.safetensors
├── optimizer.pt
├── trainer_state.json
├── model_config.json
├── train_config.toml
├── dataset_manifest_hash.txt
└── metrics.jsonl
```

Contents are git-ignored; nothing here is required to run inference, which
needs only the exported `best.safetensors` and its `model_config.json`.
