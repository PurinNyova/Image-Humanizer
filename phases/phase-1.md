## Phase 1: Minimal Repository Scaffold

**Goal:** Establish the `uv`-managed package, plain-TOML configuration, documented checkpoint
directories, and CLI entry points, with no training or inference behavior.

**Implemented structure:**

```text
Image-Humanizer/
├── pyproject.toml          # uv project, Python >=3.12, `image-humanizer` script entry
├── uv.lock
├── README.md
├── configs/                # plain TOML, no schema framework
│   ├── preprocess.toml
│   ├── train.toml
│   └── vae_registry.toml
├── checkpoints/
│   ├── README.md
│   ├── vae/{qwen_image,sdxl,flux,chroma,sd15,sd3,wan,minimax_h3_image,hunyuan,ltx}/README.md
│   └── dewatermarker/README.md
├── src/image_humanizer/
│   ├── __init__.py
│   ├── cli.py              # argparse subcommands: preprocess, validate-vaes
│   ├── image_io.py
│   ├── dataset.py
│   └── vae_registry.py
└── tests/
    ├── test_pipeline.py
    ├── test_scaffold.py
    └── test_vae_registry.py
```

**Constraints held:**

- Configuration is plain TOML parsed with the standard library; entry points are `argparse`.
- No Hydra, Lightning, plugin discovery, or adapter framework.
- No one-file-per-family adapter scaffolding.
- `.gitignore` drops checkpoint binaries, generated `data/`, `runs/`, caches, and logs, while
  keeping every checkpoint `README.md` tracked.

**Exit state:**

- The package resolves under `uv` and `image-humanizer --help` lists `preprocess` and
  `validate-vaes`.
- Every required checkpoint directory exists and carries a `README.md` describing its accepted
  contents.
- The full test suite passes under `uv run pytest`.
