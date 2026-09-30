## Phase 7: Offline Group Generation

**Goal:** Generate the complete eleven-image group for every clean source.

Expected group:

```text
original
├── Qwen reconstruction
├── SDXL reconstruction
├── Flux reconstruction
├── Chroma reconstruction
├── SD 1.5 reconstruction
├── SD3 reconstruction
├── Wan reconstruction
├── MiniMax H3 Image reconstruction
├── Hunyuan reconstruction
└── LTX reconstruction
```

Recommended storage:

```text
data/processed/
├── dataset.json
├── manifest.jsonl
├── splits.json
└── groups/<source_id>/
    ├── original.png
    ├── qwen_image.png
    ├── sdxl.png
    ├── flux.png
    ├── chroma.png
    ├── sd15.png
    ├── sd3.png
    ├── wan.png
    ├── minimax_h3_image.png
    ├── hunyuan.png
    ├── ltx.png
    └── metadata.json
```

Generation behavior:

- Give every adapter the same canonical target tensor.
- Use lossless PNG initially.
- Write to a temporary group directory.
- Validate all enabled reconstructions.
- Atomically publish only complete groups.
- Resume using hashes and metadata.
- Treat checkpoint, adapter version, precision, and tiling as dataset identity.
- Create contact sheets and amplified residual previews for an audit subset.
- Fail instead of interpolating mismatched reconstruction dimensions.

Manifest rows should represent individual training pairs while retaining a shared `group_id`.

**Caching decision:** Use cached reconstructions for production. On-the-fly generation should be limited to adapter smoke tests with one VAE at a time.

**Exit criterion:** A representative subset regenerates deterministically and passes shape, color, split, completeness, and hash checks.
