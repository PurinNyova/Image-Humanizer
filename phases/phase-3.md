## Phase 3: VAE Registry and Checkpoint Discovery

**Status: complete.** Resolution is local-files-only: `vae_registry.py` reads the
filesystem and nothing else — no `torch`, no `diffusers`, no socket, no weights.
`configs/vae_registry.toml` is the authoritative schema.

### Goal

Give every family one unambiguous local checkpoint — or one finding naming the
fix — before any substantial GPU memory is allocated.

### Implemented

| File | Contents |
|---|---|
| `src/image_humanizer/vae_registry.py` | `Finding`, `Resolved`, `Report`, `load_registry()`, `validate_registry()`, `resolve()`. |
| `src/image_humanizer/cli.py` | `uv run image-humanizer validate-vaes` with `--registry`, `--family`, `--checkpoint`. |
| `configs/vae_registry.toml` | 10 families: 9 enabled, `minimax_h3_image` disabled with `unsupported_reason`. |
| `tests/test_vae_registry.py` | Precedence, config, path traversal, hashes, shards, disabled families, CLI. |

`hf_repo`, `hf_revision` and `hf_subfolder` are **provenance only** — a citation
of which pinned upstream commit the local bytes should match. Resolution never
reads them: no network path exists to consume them and no flag enables one.
Provenance says where weights came from; resolution says only where they are on
this disk.

### Resolution order

First match wins, over local files only:

1. `override=` / `--checkpoint` (requires `--family`; applies to one entry).
2. The entry's `checkpoint`, a relative name inside `directory`.
3. The sole `*.safetensors` in `directory`.
4. Otherwise a finding: `ambiguous` (2+ candidates, never guessed) or `no_checkpoint`. There is no further rung.

`enabled = false` short-circuits to `unsupported` before any of this, even with
a valid override. `Resolved.source` records which rung fired.

### Validation

Every finding names its fix:

- `missing_field` — `directory`, `adapter`, or `reconstruction_method` absent.
- `unknown_family`, `unsupported`, `missing_directory`; `path_escape` for
  absolute or `..` names.
- `missing_checkpoint`, `missing_override`, `ambiguous`, `no_checkpoint`.
- `missing_config` — names the orphaned checkpoint and the expected path.
- `missing_shard` / `shard_mismatch` / `bad_index` — a sharded checkpoint is the
  index plus every shard its `weight_map` names; a partial set is refused, and
  Hunyuan's `.pt` is an ordinary declared name.
- `bad_adapter`, `bad_method`, `bad_metadata` — identifier plus
  `latent_channels` / `spatial_multiple` / `temporal_compression` sanity.
- `bad_hash` / `hash_mismatch` — a declared `sha256` must be 64 hex characters and
  must match. An **empty** `sha256` is not an error: the file lands in
  `Resolved.unverified` and is reported UNVERIFIED. Per-shard hashes come from
  `checkpoint_shard_sha256` when declared; without them, sharded weights stay
  unverified rather than silently passing.

### Deferred

Deferred by design, because Phase 3 loads no weights. Reported per successful
report rather than silently skipped:

- Model-class compatibility, full architecture match, and missing/unexpected
  state-dict keys, shapes, and dtypes — all need loaded weights to test.
- Mapping a deprecated `_class_name` (Hunyuan's `AutoencoderKLCausal3D`) to the
  registry adapter — the Phase 5 loader's job, not this resolver's.
- Integrity of files with no declared hash.

### Exit state

- 37 tests pass (`test_vae_registry.py`, `test_scaffold.py`) over a few-byte
  fixture each, so the shipped weights are never touched. `validate-vaes` is
  read-only and exits 1 while any enabled family has a finding.
- **No family currently validates.** The repo ships only
  `checkpoints/vae/*/README.md`, so all 9 enabled families report
  `missing_checkpoint` + `missing_config` and `minimax_h3_image` reports
  `unsupported` — `0 resolved, 9 enabled with errors`. The registry pins which
  artifacts belong; placing them locally stays a manual, deliberate step.
