# Phase 0: Pin Dataset and Checkpoint Decisions

## Goal

Resolve the choices that bind every later phase — canonical image form, source
identity, leakage-safe grouping, checkpoint provenance — so none is re-opened.

## Implemented

**Canonical form.** Every source becomes a `512×512` RGB float32 tensor in
`[0,1]`; 512 is a common multiple of every spatial multiple in the registry
(8, 16, 32), so one tensor feeds all ten adapters. Shorter side is resized to
512 with Lanczos and center-cropped to `512×512` — one resize, one crop,
before any VAE is touched, and skipped when a source is already `512×512`.

**Canonicalization order**, once per source: EXIF orientation, ICC-to-sRGB,
then alpha compositing over the configured matte (white by default). No ICC
profile means the image is already sRGB, and alpha is composited rather than
dropped. High-bit-depth grayscale and unprofiled CMYK are refused rather than
guessed.

**`source_id`** is `sha256(canonical_pixels || canon_version)`: content-derived,
so moving a file preserves its identity, identical content under two paths
collapses to one source, and changing `canon_version` invalidates every id.

**Grouping** unions sources across *all* keys, never first-key-wins: byte hash,
`source_id`, pHash Hamming distance ≤ 6 within a band, and parent directory. A
parent directory over `parent_max_fraction` of the corpus is a flat dump rather
than a burst or shot collection, so it is not a key. Group ids are deterministic.

**Frozen splits** are assigned by group, and the written `splits.json` is
authoritative: a `canon_version` mismatch or any difference in the source,
group, or parent map aborts rather than silently re-splitting a corpus.

**Checkpoint registry.** Ten VAE families are registered, nine enabled. Each
entry names its adapter, checkpoint and config filenames, latent shape,
temporal compression, reconstruction method, and license, with a verified
sha256 wherever one is known. `minimax_h3_image` is registered and
deliberately disabled: its config declares `clip_length: 17`, so `T=1` is
outside the trained regime and no single-frame path is documented, so it
fails loudly instead of guessing.

**Provenance is record-only.** `hf_repo`, `hf_revision`, and `hf_subfolder`
document where each artifact came from. Resolution reads local files only: no
network access, nothing fetched, no weights loaded during validation.

**Future input, not yet consumed.** `preprocess.toml` records the
reference-generation choices (fp32, tiling disabled) and the clean-image drift
limits (L1, PSNR). No code reads those keys yet; they are the acceptance gate
for reference generation.

## Artifacts

- `configs/preprocess.toml` — canonicalization, resize/crop, grouping, drift.
- `configs/vae_registry.toml` — ten families, provenance, hashes, licenses.
- `docs/phase0_decisions.md` — decision record with rationale, D1–D10.
- `docs/phase0_checkpoints.md` — per-family inventory with verification status.
- `image-humanizer validate-vaes` and `image-humanizer preprocess` — the commands.

## Exit state

Every VAE family has a named checkpoint, a pinned revision, a construction
source, and a license. The canonical image contract is fixed and versioned,
and grouping and splits are leakage-safe and refuse drift rather than
re-deciding it.

Weights are not present in this repository, and the registry resolves against
local files only, so validation currently reports each enabled family as
missing its local checkpoint and config, and MiniMax H3 as unsupported.
