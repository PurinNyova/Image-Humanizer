## Phase 9: Baseline Training Run

**Goal:** One local `image-humanizer train` run: AdamW, one Charbonnier objective, three output files.

**Build:** `train.py` is a straight-line script, not a framework.

```text
image-humanizer train --data data/processed --config configs/train.toml --out checkpoints/dewatermarker/<run_id>
```

```text
[model] from train.toml  → Phase 8 U-Net
PairDataset(split="train") → shuffle=True, batch_size from [optim]
repeat steps times:
    loss = charbonnier(model(watermarked), original)   # unclamped output, no target residual
    loss.backward(); clip_grad_norm_(params, grad_clip); optimizer.step()
    every step: charbonnier on split="validation" (deterministic, no augmentation) → keep lowest
```

- Optimizer: `AdamW(params, lr)` from `[optim]`. No scheduler, no weight decay tuning, no per-family loss weights, no second objective.
- Config: `lr`, `batch_size`, `grad_clip` are read as written. No schedule exists yet, so add exactly one field, `steps`, and stop there. `[model]` is consumed by `model.py`, not here.
- fp32 only. One device, one process, no accumulation, no gradient checkpointing, no resume state.
- Validation runs in `eval()` under `no_grad` and never feeds a backward pass. `test` split is untouched.

**Outputs:**

```text
checkpoints/dewatermarker/<run_id>/
├── best.safetensors    # lowest validation Charbonnier
├── last.safetensors    # final step
└── model_config.json   # base_width, channel_multipliers, groups
```

Nothing else. No `optimizer.pt`, `trainer_state.json`, `train_config.toml` copy, manifest hash, `metrics.jsonl`, dashboard, or visual panels. Re-running the same command is the resume story.

**Done:** `tests/test_train.py` covers the two acceptance checks.

- Overfit: 8 cached pairs, a few hundred steps, train Charbonnier far below its starting value. Selection reads validation only.
- Round-trip: `safetensors.torch.load_file` into a fresh Phase 8 model reproduces the saved run's output on a fixed batch.

Done when one command trains, one objective is logged, three files exist, tests pass.
