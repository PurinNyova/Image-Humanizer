## Phase 10: Baseline Training

**Goal:** Establish a reproducible baseline with one paired objective.

Baseline loss:

```text
Charbonnier(dewatermarked, original)
```

Do not add a duplicate residual objective.

Training components:

- AdamW.
- Mixed precision after float32 smoke tests pass.
- Gradient clipping.
- Batch accumulation where necessary.
- Deterministic validation.
- Balanced per-VAE sampling.
- Full-state resume.
- Periodic visual comparison panels.
- Aggregate and per-VAE metric logging.

Checkpoint contents:

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

Select checkpoints using equal weighting across VAE families, subject to a clean-image drift limit.

**Exit criterion:** A fully reproducible baseline run completes, resumes correctly, and exports inference-only Safetensors weights.
