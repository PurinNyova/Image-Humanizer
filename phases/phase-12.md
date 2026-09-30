## Phase 12: Optional Regularizers

**Goal:** Add only the minimum correction justified by baseline evidence.

Experiment order:

1. **Clean-input identity**
2. **Edge preservation**
3. **SSIM**
4. **Group consistency**

Clean identity should be the first response to excessive clean-image changes:

```text
input = clean
target = clean
loss = same Charbonnier objective
```

Mix a small identity fraction, initially around 5-10%, and evaluate the restoration-versus-no-op tradeoff.

Only add edge preservation if legitimate high-frequency detail is being smoothed. Only add SSIM if structural metrics or visual inspection justify it. Group consistency requires loading several reconstructions from one group and should be deferred unless per-VAE outputs diverge materially.

Run one regularizer change at a time.

**Exit criterion:** Retain a regularizer only if it improves held-out behavior without hiding degradation in any VAE family or increasing clean-image drift.
