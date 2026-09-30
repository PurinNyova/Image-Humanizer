## Phase 0: Pin Dataset and Checkpoint Decisions

**Goal:** Resolve choices that affect every later phase.

Tasks:

- Select the initial canonical training resolution, recommended `512×512`.
- Define image canonicalization:
  - Apply EXIF orientation.
  - Convert ICC-managed images to sRGB.
  - Composite alpha over a configured background.
  - Represent pixels as float32 RGB in `[0,1]`.
- Decide source resize and crop policy.
- Identify grouping keys for videos, bursts, subjects, artists, products, or near-duplicate collections.
- Select an exact checkpoint revision for every VAE.
- Record licensing and redistribution restrictions.
- Determine whether each video-derived VAE will use:
  - An image-retrained checkpoint, preferred when authoritative and available.
  - Its officially supported single-frame encode-decode path otherwise.
- Choose reference-generation precision and whether VAE tiling is allowed.
- Define a maximum acceptable clean-image drift for model selection.

**Deliverables:**

- Checkpoint inventory covering all ten VAE families.
- Decision record for image processing and temporal handling.
- Initial dependency versions to pin in `uv.lock`.

**Exit criterion:** Every VAE has a named candidate checkpoint and documented construction source. Unknown standalone `.safetensors` files are not considered sufficient.
