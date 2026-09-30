"""Phase 3: VAE registry resolution and pre-load validation.

Nothing here constructs, imports, or allocates a model, and nothing is
downloaded: resolution is local files only. A result is either a `Resolved`
(paths + metadata a Phase 5 loader/reconstruction step needs) or a list of precise `Finding`s
naming the fix.

Schema follows `configs/vae_registry.toml` (authoritative), not the Phase 3
snippet in `plan.md`/`phases/phase-3.md`, which is stale in two ways:
provenance is `hf_repo` + `hf_revision` + `hf_subfolder`, and Hunyuan's
checkpoint is a `.pt` while MiniMax H3 is a sharded index plus shards.
Decisions: docs/phase0_decisions.md, docs/phase0_checkpoints.md.
"""

from __future__ import annotations

import json
import tomllib
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REGISTRY = REPO_ROOT / "configs" / "vae_registry.toml"

_REQUIRED = ("directory", "adapter", "reconstruction_method")
_METHODS = {"native_image", "supported_single_frame", "unsupported"}
_LATENT_KEYS = ("scaling_factor", "shift_factor", "latents_mean", "latents_std")
_HEX = frozenset("0123456789abcdef")


@dataclass(frozen=True)
class Finding:
    """One problem plus the action that fixes it."""

    code: str
    message: str


@dataclass(frozen=True)
class Resolved:
    name: str
    adapter: str
    directory: Path
    checkpoint: Path | None  # None only if the entry names shards instead
    shards: tuple[Path, ...]
    config: Path | None
    source: str  # override | registry | sole_local_safetensors
    reconstruction_method: str
    latent_channels: int
    spatial_multiple: int
    temporal: bool
    temporal_compression: int | None
    denoiser_scale_factor: float | None  # denoiser convention; never a VAE reconstruction scale
    class_name: str | None  # config.json _class_name
    latent_transform: tuple[str, ...]  # which of _LATENT_KEYS the config declares
    hashes: dict[str, str]  # str(path) -> verified sha256
    unverified: tuple[str, ...]  # file names with no non-empty declared sha256


@dataclass(frozen=True)
class Report:
    name: str
    resolved: Resolved | None
    findings: tuple[Finding, ...] = ()
    deferred: tuple[str, ...] = ()  # checks deliberately not performed here

    @property
    def ok(self) -> bool:
        return self.resolved is not None and not self.findings


def load_registry(path: Path = DEFAULT_REGISTRY) -> dict[str, dict]:
    """Duplicated from cli.py on purpose: cli will import this module, and a
    reverse import would be a cycle."""
    with Path(path).open("rb") as f:
        return tomllib.load(f)["vae"]


def validate_registry(
    registry: dict[str, dict] | None = None,
    *,
    root: Path = REPO_ROOT,
) -> list[Report]:
    """Every entry, sorted, each resolved independently."""
    return [
        resolve(name, registry, root=root)
        for name in sorted(registry if registry is not None else load_registry())
    ]


def resolve(
    name: str,
    registry: dict[str, dict] | None = None,
    *,
    root: Path = REPO_ROOT,
    override: Path | None = None,
) -> Report:
    """Resolve one entry by precedence: override, registry `checkpoint`, the sole
    local `.safetensors`. Local files only; nothing is ever downloaded."""
    registry = load_registry() if registry is None else registry
    entry = registry.get(name)
    if entry is None:
        return Report(name, None, (Finding("unknown_family", f"no registry entry {name!r}; known: {sorted(registry)}"),))
    if not entry.get("enabled", True):
        reason = entry.get("unsupported_reason") or "registry omits `unsupported_reason`"
        return Report(
            name,
            None,
            (Finding("unsupported", f"VAE {name!r} is not loadable: {reason}"),),
            (),
        )

    findings: list[Finding] = []
    deferred: list[str] = []
    for key in _REQUIRED:
        if not entry.get(key):
            findings.append(Finding("missing_field", f"VAE {name!r} has no {key!r} field."))
    if findings:
        return Report(name, None, tuple(findings))

    declared = str(entry.get("checkpoint") or "")
    declared_config = str(entry.get("config") or "")
    shards = tuple(str(s) for s in entry.get("checkpoint_shards") or ())

    raw_dir = Path(str(entry["directory"]))
    directory = (raw_dir if raw_dir.is_absolute() else root / raw_dir).resolve()
    if not directory.is_dir():
        return Report(
            name,
            None,
            [
                *findings,
                Finding("missing_directory", f"VAE {name!r} directory {str(entry['directory'])!r} is not a directory (resolved to {directory}). Create it or fix `directory`."),
            ],
        )

    selected: Path | None = None
    source = ""
    if override is not None:
        picked = Path(override)
        selected = (picked if picked.is_absolute() else root / picked).resolve()
        source = "override"
        if not selected.is_file():
            findings.append(Finding("missing_override", f"VAE {name!r} checkpoint override {str(override)!r} is not an existing file ({selected})."))
    elif declared:
        # Registry paths are names inside the declared directory. Never absolute, never traversing.
        inside = _inside(directory, declared)
        if inside is None:
            findings.append(Finding("path_escape", f"VAE {name!r} checkpoint {declared!r} must be a relative name inside {directory}, not an absolute or traversing path."))
        elif not inside.is_file():
            findings.append(Finding("missing_checkpoint", f"VAE {name!r} registry checkpoint {declared!r} is missing from {directory}. Fetch it, or fix `checkpoint`."))
        else:
            selected, source = inside, "registry"
    else:
        candidates = sorted(p.name for p in directory.glob("*.safetensors") if p.is_file())
        if len(candidates) == 1:
            selected, source = directory / candidates[0], "sole_local_safetensors"
        elif not candidates:
            findings.append(Finding("no_checkpoint", f"VAE {name!r} declares no `checkpoint` and {directory} holds no .safetensors. Place the pinned weights there, or set `checkpoint`."))
        else:
            findings.append(Finding("ambiguous", f"VAE {name!r} has {len(candidates)} .safetensors candidates in {directory} ({', '.join(candidates)}). Set `checkpoint` in the registry or pass an explicit checkpoint override; ambiguity is never resolved by guessing."))

    config = None
    if declared_config:
        config = _inside(directory, declared_config)
        if config is None:
            findings.append(Finding("path_escape", f"VAE {name!r} config {declared_config!r} must be a relative name inside {directory}, not an absolute or traversing path."))
        elif not config.is_file():
            config = None
            what = f"resolves checkpoint {selected.name!r}" if selected else "has no resolved checkpoint"
            findings.append(Finding("missing_config", f"VAE {name!r} {what} but no compatible architecture configuration was found: expected {directory / declared_config}. Provide {declared_config} in the checkpoint directory."))

    shard_paths: tuple[Path, ...] = ()
    if shards:
        found = []
        for shard in shards:
            inside = _inside(directory, shard)
            if inside is None:
                findings.append(Finding("path_escape", f"VAE {name!r} shard {shard!r} must be a relative name inside {directory}, not an absolute or traversing path."))
            elif not inside.is_file():
                findings.append(Finding("missing_shard", f"VAE {name!r} shard {shard!r} is missing from {directory}; a sharded checkpoint needs the index and every shard."))
            else:
                found.append(inside)
        shard_paths = tuple(found)
        if selected is not None and selected.name.endswith(".index.json"):
            findings.extend(_index_findings(name, selected, found))

    findings.extend(_metadata_findings(name, entry))

    hashes, unverified, hash_findings = _verify_hashes(name, entry, (selected, config, *shard_paths))
    findings.extend(hash_findings)
    deferred += [f"{n}: no registry sha256 declared, so integrity is UNVERIFIED (not verified)" for n in unverified]
    if findings:
        return Report(name, None, tuple(findings), tuple(deferred))

    class_name, latent_transform = _read_config(config)
    if class_name and entry.get("deprecated_class") == class_name:
        deferred.append(f"config declares {class_name!r}, which the pinned diffusers does not export; the adapter must map it to the registry adapter {entry['adapter']!r} at load time")
    deferred.append("model class compatibility, missing/unexpected state-dict keys, and full architecture match are DEFERRED: no weights are loaded in Phase 3")
    return Report(
        name,
        Resolved(
            name=name,
            adapter=str(entry["adapter"]),
            directory=directory,
            checkpoint=selected,
            shards=shard_paths,
            config=config,
            source=source,
            reconstruction_method=str(entry["reconstruction_method"]),
            latent_channels=int(entry["latent_channels"]),
            spatial_multiple=int(entry["spatial_multiple"]),
            temporal=bool(entry["temporal"]),
            temporal_compression=(int(entry["temporal_compression"]) if entry.get("temporal_compression") is not None else None),
            denoiser_scale_factor=(float(entry["denoiser_scale_factor"]) if entry.get("denoiser_scale_factor") is not None else None),
            class_name=class_name,
            latent_transform=latent_transform,
            hashes=hashes,
            unverified=tuple(unverified),
        ),
        (),
        tuple(deferred),
    )


def _inside(directory: Path, rel: str) -> Path | None:
    """Resolve a registry-declared name under `directory`, or None if it escapes."""
    p = Path(rel)
    if p.is_absolute() or p.drive or ".." in p.parts:
        return None
    resolved = (directory / p).resolve()
    return resolved if resolved.is_relative_to(directory) else None


def _pos_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _metadata_findings(name: str, entry: dict) -> list[Finding]:
    out = []
    adapter = str(entry["adapter"])
    if not adapter.isidentifier():
        out.append(Finding("bad_adapter", f"VAE {name!r} adapter {adapter!r} must be a valid identifier used as registry metadata."))
    if entry["reconstruction_method"] not in _METHODS:
        out.append(Finding("bad_method", f"VAE {name!r} reconstruction_method {str(entry['reconstruction_method'])!r} is not one of {sorted(_METHODS)}."))
    if not isinstance(entry.get("temporal"), bool):
        out.append(Finding("bad_metadata", f"VAE {name!r} must declare boolean `temporal`; single-frame reconstruction cannot be planned without it."))
    if not _pos_int(entry.get("latent_channels")):
        out.append(Finding("bad_metadata", f"VAE {name!r} latent_channels must be a positive integer, got {entry.get('latent_channels')!r}."))
    if not _pos_int(entry.get("spatial_multiple")):
        out.append(Finding("bad_metadata", f"VAE {name!r} spatial_multiple must be a positive integer, got {entry.get('spatial_multiple')!r}."))
    if entry.get("temporal") and not _pos_int(entry.get("temporal_compression")):
        out.append(Finding("bad_metadata", f"VAE {name!r} is temporal, so temporal_compression must be a positive integer, got {entry.get('temporal_compression')!r}."))
    return out


def _index_findings(name: str, index: Path, shards: list[Path]) -> list[Finding]:
    """A sharded checkpoint is the index plus every shard it maps; a partial set is worse than none."""
    try:
        weight_map = json.loads(index.read_text(encoding="utf-8")).get("weight_map") or {}
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        return [Finding("bad_index", f"VAE {name!r} index {index.name} is not readable JSON ({e}). Re-fetch the index; a sharded checkpoint cannot be resolved from it.")]
    mapped = set(weight_map.values())
    declared = {p.name for p in shards}
    if mapped != declared:
        return [Finding("shard_mismatch", f"VAE {name!r} index {index.name} maps {sorted(mapped)} but the registry declares {sorted(declared)}; index and shards must be the same set.")]
    return []


def _declared_hash(entry: dict, filename: str) -> str:
    """One `sha256` for the named checkpoint. Per-shard hashes are optional via
    `checkpoint_shard_sha256`; the shipped schema declares none, so sharded
    weights stay unverifiable rather than silently passing."""
    per_shard = entry.get("checkpoint_shard_sha256") or {}
    if filename in per_shard:
        return str(per_shard[filename])
    return str(entry.get("sha256") or "") if filename == entry.get("checkpoint") else ""


def _verify_hashes(name: str, entry: dict, paths: tuple[Path | None, ...]) -> tuple[dict[str, str], list[str], list[Finding]]:
    from image_humanizer.image_io import file_sha256  # lazy: keeps torch out of this module's import path

    hashes: dict[str, str] = {}
    unverified: list[str] = []
    findings: list[Finding] = []
    for path in paths:
        if path is None or not path.is_file():
            continue  # already reported as missing
        want = _declared_hash(entry, path.name)
        if want and (len(want) != 64 or set(want) - _HEX):
            findings.append(Finding("bad_hash", f"VAE {name!r} {path.name}: declared sha256 {want!r} is not 64 hex characters."))
            continue
        got = file_sha256(path)
        if not want:
            unverified.append(path.name)  # empty declared hash means unverifiable, never verified
        elif got != want:
            findings.append(Finding("hash_mismatch", f"VAE {name!r} {path.name}: registry declares sha256 {want} but the file hashes to {got}. This is not the pinned artifact."))
        else:
            hashes[str(path)] = got
    return hashes, unverified, findings


def _read_config(config: Path | None) -> tuple[str | None, tuple[str, ...]]:
    if config is None:
        return None, ()
    try:
        raw = json.loads(config.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError, OSError):
        return None, ()  # unreadable config already blocks resolution
    return raw.get("_class_name"), tuple(k for k in _LATENT_KEYS if k in raw)
