"""Phase 3: VAE registry resolution and pre-load validation.

Read-only by construction: no weights, no model, no GPU, no network. Every
fixture is a few bytes under `tmp_path`, and the shipped registry's on-disk
checkpoints are never touched.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from image_humanizer.cli import main
from image_humanizer.vae_registry import resolve, validate_registry

BASE = {
    "adapter": "diffusers_autoencoder_kl",
    "hf_repo": "org/weights",
    "hf_revision": "0" * 40,
    "reconstruction_method": "native_image",
    "latent_channels": 4,
    "spatial_multiple": 8,
    "temporal": False,
    "enabled": True,
    "directory": "vae",
}


def _put(directory: Path, name: str, data: bytes = b"w") -> Path:
    p = directory / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(data)
    return p


def _sha(data: bytes = b"w") -> str:
    return hashlib.sha256(data).hexdigest()


def _entry(**over) -> dict:
    return {**BASE, **over}


def _codes(report) -> set[str]:
    return {f.code for f in report.findings}


def _message(report, code: str) -> str:
    return next(f.message for f in report.findings if f.code == code)


def _toml(name: str, entry: dict) -> str:
    lines = [f"[vae.{name}]"]
    for key, value in entry.items():
        if isinstance(value, bool):
            lines.append(f"{key} = {'true' if value else 'false'}")
        elif isinstance(value, (int, float)):
            lines.append(f"{key} = {value}")
        elif isinstance(value, list):
            lines.append(f"{key} = [{', '.join(json.dumps(str(v)) for v in value)}]")
        else:
            lines.append(f"{key} = {json.dumps(str(value))}")
    return "\n".join(lines) + "\n"


@pytest.fixture
def vae_dir(tmp_path: Path) -> Path:
    d = tmp_path / "vae"
    d.mkdir()
    return d


# --- selection precedence ------------------------------------------------


def test_override_beats_the_registry_checkpoint(vae_dir, tmp_path):
    reg = {"x": _entry(checkpoint="pinned.safetensors")}
    _put(vae_dir, "pinned.safetensors")
    picked = _put(vae_dir, "other.safetensors", b"other")

    r = resolve("x", reg, root=tmp_path, override=picked)
    assert r.ok and r.resolved.source == "override"
    assert r.resolved.checkpoint.name == "other.safetensors"


def test_missing_override_is_reported_and_resolves_nothing(vae_dir, tmp_path):
    reg = {"x": _entry(checkpoint="pinned.safetensors")}
    _put(vae_dir, "pinned.safetensors")

    r = resolve("x", reg, root=tmp_path, override=vae_dir / "absent.safetensors")
    assert _codes(r) == {"missing_override"} and r.resolved is None
    assert "absent.safetensors" in _message(r, "missing_override")


def test_registry_checkpoint_beats_a_lone_safetensors_and_a_pt(vae_dir, tmp_path):
    # Hunyuan shape: the weights are .pt, and a stray .safetensors must not be
    # picked up in its place, nor make the directory ambiguous.
    _put(vae_dir, "pytorch_model.pt")
    _put(vae_dir, "decoy.safetensors")

    r = resolve("x", {"x": _entry(checkpoint="pytorch_model.pt")}, root=tmp_path)
    assert r.ok and r.resolved.source == "registry"
    assert r.resolved.checkpoint.name == "pytorch_model.pt"


def test_sole_local_safetensors_is_selected(vae_dir, tmp_path):
    _put(vae_dir, "only.safetensors")

    r = resolve("x", {"x": _entry()}, root=tmp_path)
    assert r.ok and r.resolved.source == "sole_local_safetensors"
    assert r.resolved.checkpoint.name == "only.safetensors"


def test_two_candidates_are_ambiguous_and_never_guessed(vae_dir, tmp_path):
    _put(vae_dir, "b.safetensors")
    _put(vae_dir, "a.safetensors")
    _put(vae_dir, "ignored.pt")  # not a .safetensors candidate

    r = resolve("x", {"x": _entry()}, root=tmp_path)
    assert _codes(r) == {"ambiguous"} and r.resolved is None
    message = _message(r, "ambiguous")
    assert "a.safetensors, b.safetensors" in message  # candidates, sorted
    assert "never resolved by guessing" in message


def test_empty_directory_names_the_local_ways_out(vae_dir, tmp_path):
    r = resolve("x", {"x": _entry()}, root=tmp_path)
    assert _codes(r) == {"no_checkpoint"}
    message = _message(r, "no_checkpoint")
    assert "Place the pinned weights there" in message and "set `checkpoint`" in message


# --- architecture configuration -------------------------------------------


def test_missing_config_gives_the_corrective_error(vae_dir, tmp_path):
    # Plan's example failure: weights present, architecture config absent.
    _put(vae_dir, "wan_vae.safetensors")

    r = resolve("x", {"x": _entry(config="config.json")}, root=tmp_path)
    assert _codes(r) == {"missing_config"}
    message = _message(r, "missing_config")
    assert "no compatible architecture configuration was found" in message
    assert "config.json" in message
    assert "wan_vae.safetensors" in message  # says which checkpoint is orphaned


def test_missing_config_without_a_checkpoint_says_so(vae_dir, tmp_path):
    _put(vae_dir, "weights.pt")  # no .safetensors to select

    r = resolve("x", {"x": _entry(config="config.json")}, root=tmp_path)
    assert _codes(r) == {"no_checkpoint", "missing_config"}
    assert "no resolved checkpoint" in _message(r, "missing_config")


def test_config_metadata_and_deprecated_class_are_reported_not_guessed(vae_dir, tmp_path):
    _put(vae_dir, "pinned.safetensors")
    _put(vae_dir, "config.json", b'{"_class_name": "AutoencoderKLCausal3D", "shift_factor": 0.1}')
    entry = _entry(checkpoint="pinned.safetensors", config="config.json", deprecated_class="AutoencoderKLCausal3D")

    r = resolve("x", {"x": entry}, root=tmp_path)
    assert r.ok and r.resolved.class_name == "AutoencoderKLCausal3D"
    assert r.resolved.latent_transform == ("shift_factor",)  # only what the config declares
    assert any("AutoencoderKLCausal3D" in d and "diffusers_autoencoder_kl" in d for d in r.deferred)


# --- whole-registry validation --------------------------------------------


def test_default_validation_is_local_and_sorted(vae_dir, tmp_path):
    # Neither entry has its weights on disk, so both must fail locally and say
    # where to get them -- resolution never leaves the filesystem.
    reg = {name: _entry(checkpoint="absent.safetensors") for name in ("b", "a")}

    reports = validate_registry(reg, root=tmp_path)
    assert [r.name for r in reports] == ["a", "b"]
    assert all(_codes(r) == {"missing_checkpoint"} for r in reports)
    assert "Fetch it, or fix `checkpoint`" in _message(reports[0], "missing_checkpoint")


# --- path traversal -------------------------------------------------------


@pytest.mark.parametrize("rel", ["../escape.safetensors", "sub/../../escape.safetensors", "ABSOLUTE"])
@pytest.mark.parametrize("field", ["checkpoint", "config", "shard"])
def test_registry_paths_may_not_traverse(vae_dir, tmp_path, field, rel):
    _put(vae_dir, "pinned.safetensors")
    _put(vae_dir, "config.json", b"{}")
    _put(tmp_path, "escape.safetensors")  # the escape target really exists

    entry = _entry(checkpoint="pinned.safetensors", config="config.json")
    escape = str(tmp_path / "escape.safetensors") if rel == "ABSOLUTE" else rel
    if field == "shard":
        entry["checkpoint_shards"] = [escape]
    else:
        entry[field] = escape

    r = resolve("x", {"x": entry}, root=tmp_path)
    assert "path_escape" in _codes(r)
    message = _message(r, "path_escape")
    assert "not an absolute or traversing path" in message
    assert repr(escape) in message  # the offending value is quoted back verbatim


# --- integrity ------------------------------------------------------------


def test_sha256_is_verified_or_flagged_never_assumed(vae_dir, tmp_path):
    weights = _put(vae_dir, "pinned.safetensors")
    _put(vae_dir, "config.json", b"{}")
    reg = {"x": _entry(checkpoint="pinned.safetensors", config="config.json", sha256=_sha())}

    r = resolve("x", reg, root=tmp_path)
    assert r.ok and r.resolved.hashes == {str(weights.resolve()): _sha()}
    # config.json is never hash-verified: the schema declares no sha256 for it.
    assert r.resolved.unverified == ("config.json",)
    assert any("UNVERIFIED" in d for d in r.deferred)

    wrong = resolve("x", {"x": {**reg["x"], "sha256": _sha(b"other")}}, root=tmp_path)
    assert _codes(wrong) == {"hash_mismatch"} and wrong.resolved is None
    assert "not the pinned artifact" in _message(wrong, "hash_mismatch")

    empty = resolve("x", {"x": {**reg["x"], "sha256": ""}}, root=tmp_path)
    assert empty.ok and empty.resolved.hashes == {}
    assert "pinned.safetensors" in empty.resolved.unverified
    assert any("pinned.safetensors" in d and "UNVERIFIED" in d for d in empty.deferred)

    malformed = resolve("x", {"x": {**reg["x"], "sha256": "nothex"}}, root=tmp_path)
    assert _codes(malformed) == {"bad_hash"} and "64 hex characters" in _message(malformed, "bad_hash")


def test_per_shard_hashes_are_optional_and_used_when_declared(vae_dir, tmp_path):
    names = ["shard-1.safetensors", "shard-2.safetensors"]
    for n in names:
        _put(vae_dir, n)
    index = _put(vae_dir, "model.safetensors.index.json", json.dumps({"weight_map": {"a": names[0], "b": names[1]}}).encode())
    entry = _entry(checkpoint=index.name, checkpoint_shards=names)

    r = resolve("x", {"x": entry}, root=tmp_path)
    assert r.ok and r.resolved.hashes == {} and set(r.resolved.unverified) >= {*names, index.name}

    declared = {**entry, "checkpoint_shard_sha256": {names[0]: _sha(), names[1]: _sha(b"wrong")}}
    r = resolve("x", {"x": declared}, root=tmp_path)
    assert _codes(r) == {"hash_mismatch"} and names[1] in _message(r, "hash_mismatch")


# --- sharded checkpoint ---------------------------------------------------


def test_sharded_checkpoint_is_the_index_plus_every_shard(vae_dir, tmp_path):
    names = ["shard-1.safetensors", "shard-2.safetensors"]
    for n in names:
        _put(vae_dir, n)
    index = _put(vae_dir, "model.safetensors.index.json", json.dumps({"weight_map": {"a": names[0], "b": names[1]}}).encode())
    reg = {"x": _entry(checkpoint=index.name, checkpoint_shards=names)}

    r = resolve("x", reg, root=tmp_path)
    assert r.ok and r.resolved.checkpoint.name == index.name
    assert [p.name for p in r.resolved.shards] == names

    # The index maps fewer files than the registry declares: a partial set is refused.
    _put(vae_dir, index.name, json.dumps({"weight_map": {"a": names[0]}}).encode())
    r = resolve("x", reg, root=tmp_path)
    assert _codes(r) == {"shard_mismatch"} and index.name in _message(r, "shard_mismatch")

    # A declared shard that is not on disk is fatal, not skippable.
    reg = {"x": {**reg["x"], "checkpoint_shards": [*names, "shard-3.safetensors"]}}
    _put(vae_dir, index.name, json.dumps({"weight_map": {"a": names[0], "b": names[1], "c": "shard-3.safetensors"}}).encode())
    r = resolve("x", reg, root=tmp_path)
    assert "missing_shard" in _codes(r) and "shard-3.safetensors" in _message(r, "missing_shard")

    # An unreadable index cannot be resolved from.
    _put(vae_dir, index.name, b"not json")
    r = resolve("x", reg, root=tmp_path)
    assert "bad_index" in _codes(r) and "Re-fetch the index" in _message(r, "bad_index")


# --- disabled families and entry-level diagnostics ------------------------


def test_disabled_family_is_never_loadable(vae_dir, tmp_path):
    # Everything a healthy entry needs is present and still must not resolve.
    _put(vae_dir, "only.safetensors")
    _put(vae_dir, "config.json", b"{}")
    entry = _entry(config="config.json", enabled=False, unsupported_reason="clip_length=17 is outside the documented regime")

    for override in (None, vae_dir / "only.safetensors"):
        r = resolve("x", {"x": entry}, root=tmp_path, override=override)
        assert _codes(r) == {"unsupported"} and r.resolved is None
        assert "not loadable" in _message(r, "unsupported")
        assert "clip_length=17" in _message(r, "unsupported")


def test_disabled_family_without_a_reason_is_still_flagged(vae_dir, tmp_path):
    r = resolve("x", {"x": _entry(enabled=False)}, root=tmp_path)
    assert _codes(r) == {"unsupported"}
    assert "registry omits `unsupported_reason`" in _message(r, "unsupported")


def test_entry_level_failures_name_the_fix(vae_dir, tmp_path):
    _put(vae_dir, "pinned.safetensors")
    reg = {"x": _entry(checkpoint="pinned.safetensors")}

    unknown = resolve("nope", reg, root=tmp_path)
    assert _codes(unknown) == {"unknown_family"} and "known: ['x']" in _message(unknown, "unknown_family")

    absent = resolve("x", {"x": {**reg["x"], "directory": "not_created"}}, root=tmp_path)
    assert _codes(absent) == {"missing_directory"} and "not_created" in _message(absent, "missing_directory")

    incomplete = resolve("x", {"x": {**reg["x"], "adapter": ""}}, root=tmp_path)
    assert _codes(incomplete) == {"missing_field"} and "adapter" in _message(incomplete, "missing_field")


# --- CLI ------------------------------------------------------------------


def test_cli_checkpoint_requires_family(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["validate-vaes", "--checkpoint", "weights.safetensors"])
    assert exc.value.code == 2
    assert "--checkpoint requires --family" in capsys.readouterr().err


def test_cli_override_wins_and_stays_local(vae_dir, tmp_path, capsys):
    _put(vae_dir, "pinned.safetensors")
    other = _put(vae_dir, "other.safetensors", b"other")
    _put(vae_dir, "config.json", b"{}")
    registry = tmp_path / "registry.toml"
    # The CLI resolves `directory` against the repo root, so it must be absolute.
    registry.write_text(
        _toml("x", _entry(directory=vae_dir.as_posix(), checkpoint="pinned.safetensors", config="config.json", sha256=_sha())),
        encoding="utf-8",
    )
    base = ["validate-vaes", "--registry", str(registry), "--family", "x"]

    assert main([*base, "--checkpoint", str(other)]) == 0
    out = capsys.readouterr().out
    assert "source=override checkpoint=other.safetensors" in out
    # The override is not the declared artifact, so it is UNVERIFIED, not verified.
    assert "UNVERIFIED" in out and "resolved, 0 enabled with errors" in out

    assert main(base) == 0
    assert "source=registry checkpoint=pinned.safetensors" in capsys.readouterr().out


def test_cli_reports_findings_as_codes(capsys):
    main(["validate-vaes"])  # exit status depends on the caller's local weights
    out = capsys.readouterr().out
    assert any(f"{code}: VAE " in out for code in ("missing_checkpoint", "missing_config", "path_escape", "unsupported"))
