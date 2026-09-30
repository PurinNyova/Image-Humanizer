## Phase 6: Specialized and Temporal VAE Adapters

**Goal:** Add the remaining adapters without guessing their temporal or architecture behavior.

Implementation order:

1. Qwen Image
2. Wan
3. Hunyuan
4. LTX
5. MiniMax H3 Image

Per-family policy:

| VAE | Preferred path | Fallback path |
|---|---|---|
| Qwen Image | Official image reconstruction path | Supported single-frame spatiotemporal path |
| Wan | Authoritative image-retrained checkpoint | Official single-frame video path |
| Hunyuan | Authoritative image-retrained checkpoint | Official single-frame video path |
| LTX | Image-retrained checkpoint | Supported `T=1` encode-decode path |
| MiniMax H3 Image | Dedicated image checkpoint and authoritative architecture | No guessed loader; remain unresolved until construction metadata is available |

Temporal handling must record:

```text
checkpoint_variant: image_retrained | video
reconstruction_method: native_image | supported_single_frame
temporal_input_frames: 1
temporal_padding: ...
output_frame_selection: ...
```

For single-frame paths:

- Add `T=1` in the dimension expected by the model.
- Follow documented temporal padding or causal behavior.
- Decode using the corresponding supported path.
- Extract the documented output frame.
- Assert no unexplained temporal duplication or frame shift.

**Tests:**

- Temporal dimension placement.
- Frame count before and after decode.
- Determinism.
- Image-retrained versus single-frame provenance.
- Correct spatial and temporal compression constraints.

**Exit criterion:** Each adapter is verified against a named checkpoint, or explicitly remains unsupported with an actionable validation error. No architecture is inferred from an arbitrary weight file.
