## Phase 11: Evaluation and Clean-Image Safety

**Goal:** Measure restoration by VAE family while detecting unnecessary modification of legitimate content.

Per-VAE metrics:

- Input and output L1 or Charbonnier.
- PSNR before and after.
- SSIM before and after.
- Optional LPIPS.
- Percentage of images improved.
- Worst-decile performance.

Aggregate metrics:

- Equal-weight mean across VAE families.
- Worst VAE family.
- Worst original group.
- Variance among outputs from the same original.
- Improvement consistency across VAE types.

Clean-input metrics:

- `L1(model(clean), clean)`
- Clean PSNR and SSIM.
- Color drift percentiles.
- Edge-energy change.
- Fraction of pixels changed above practical thresholds.
- Amplified predicted residuals.

Clean evaluation sets should include:

- Camera photographs.
- Illustrations.
- Text-heavy images.
- JPEG-compressed clean images.
- Resized clean images.
- Out-of-domain dimensions and aspect ratios.

**Exit criterion:** The baseline improves each target VAE family without exceeding the agreed clean-image drift threshold.
