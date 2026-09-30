## Phase 8: Dataset Loader and Balanced Sampling

**Goal:** Expose cached reconstructions as balanced pixel-space training pairs.

Each sample returns:

```text
watermarked: VAE reconstruction
original: clean target
vae_name: evaluation and sampling metadata
group_id: source group identity
```

Sampling strategy:

1. Choose a VAE family uniformly.
2. Choose a valid original uniformly for that VAE.
3. Load the reconstruction and clean target.

This avoids accidental dominance caused by failed groups, duplicate rows, or unequal availability.

Training augmentation must preserve alignment:

- Apply the same crop and flip to input and target.
- Avoid independent color augmentation.
- Avoid arbitrary color augmentation that changes the clean target relationship.
- Keep validation and test deterministic.

Do not load all ten reconstructions in every baseline batch. Group batches are only needed if group consistency is later enabled.

**Tests:**

- Input and target remain aligned.
- Split boundaries are respected.
- Sampling frequencies are approximately uniform.
- Corrupt or incomplete groups fail clearly.
- Every training pair uses decoded pixels, not latent tensors.

**Exit criterion:** The loader produces balanced, aligned batches without loading any VAE.
