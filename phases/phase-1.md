## Phase 1: Minimal Repository Scaffold

**Goal:** Establish the package, configuration, and required local directory structure without implementing model-specific behavior.

Proposed structure:

```text
Image-Humanizer/
├── pyproject.toml
├── uv.lock
├── README.md
├── configs/
│   ├── vae_registry.toml
│   ├── preprocess.toml
│   └── train.toml
├── checkpoints/
│   ├── vae/
│   │   ├── qwen_image/
│   │   ├── sdxl/
│   │   ├── flux/
│   │   ├── chroma/
│   │   ├── sd15/
│   │   ├── sd3/
│   │   ├── wan/
│   │   ├── minimax_h3_image/
│   │   ├── hunyuan/
│   │   └── ltx/
│   └── dewatermarker/
├── src/image_humanizer/
│   ├── __init__.py
│   ├── image_io.py
│   ├── vae_registry.py
│   ├── vae_adapters.py
│   ├── generate_pairs.py
│   ├── dataset.py
│   ├── model.py
│   ├── losses.py
│   ├── train.py
│   ├── evaluate.py
│   └── infer.py
└── tests/
```

Tasks:

- Configure a Python/PyTorch project managed through `uv`.
- Ignore checkpoint binaries, generated datasets, runs, and temporary files.
- Keep a short `README.md` in every VAE checkpoint directory describing accepted contents.
- Define plain TOML configuration and `argparse` entry points.
- Avoid Hydra, Lightning, plugin discovery, and one-file-per-adapter scaffolding until needed.

**Exit criterion:** The package imports, commands expose help, and all required checkpoint directories are documented.
