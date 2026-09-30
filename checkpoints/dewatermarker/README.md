# Dewatermarker runs

Training outputs live here, one directory per `run_id`:

```text
checkpoints/dewatermarker/<run_id>/
├── last.safetensors
├── best.safetensors
└── model_config.json
```

Nothing else: no `optimizer.pt`, `trainer_state.json`, `train_config.toml`
copy, manifest hash, or `metrics.jsonl`.

Contents are git-ignored; nothing here is required to run inference, which
needs only the exported `best.safetensors` and its `model_config.json`.
