from pathlib import Path

import pytest

from image_humanizer.cli import main
from image_humanizer.vae_registry import load_registry

REPO_ROOT = Path(__file__).resolve().parents[1]
_REGISTRY_TEXT = (REPO_ROOT / "configs" / "vae_registry.toml").read_text(encoding="utf-8")
# Phase 3 moved VAE_NAMES out of cli.py, so the family list is pinned here instead.
EXPECTED_FAMILIES = {
    "qwen_image",
    "sdxl",
    "flux",
    "chroma",
    "sd15",
    "sd3",
    "wan",
    "hunyuan",
    "ltx",
    "minimax_h3_image",
}


def test_package_imports():
    import image_humanizer

    assert image_humanizer.__version__


def test_registry_covers_every_family():
    assert set(load_registry(REPO_ROOT / "configs" / "vae_registry.toml")) == EXPECTED_FAMILIES


def test_every_family_names_a_checkpoint_and_source():
    # Phase 0 exit criterion: no anonymous .safetensors.
    for name, entry in load_registry(REPO_ROOT / "configs" / "vae_registry.toml").items():
        assert entry["checkpoint"], name
        assert entry["hf_repo"], name
        assert len(entry["hf_revision"]) == 40, name
        assert entry["reconstruction_method"] in {
            "native_image",
            "supported_single_frame",
            "unsupported",
        }, name


def test_disabled_families_explain_themselves():
    for name, entry in load_registry(REPO_ROOT / "configs" / "vae_registry.toml").items():
        if not entry["enabled"]:
            assert entry.get("unsupported_reason"), name


def test_gated_families_are_marked():
    registry = load_registry(REPO_ROOT / "configs" / "vae_registry.toml")
    gated = {name for name, e in registry.items() if e.get("gated")}
    assert gated == {"flux", "sd3"}


def test_every_checkpoint_dir_is_documented():
    for name in EXPECTED_FAMILIES:
        readme = REPO_ROOT / "checkpoints" / "vae" / name / "README.md"
        assert readme.is_file(), name


def test_help_exits_zero(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0
    assert "validate-vaes" in capsys.readouterr().out


def test_validate_vaes_rejects_an_unknown_family(capsys):
    # No structural-only MISSING sweep any more: a name absent from the registry is
    # an unknown_family finding, and it still fails the command.
    assert main(["validate-vaes", "--family", "not_a_family"]) == 1
    assert "unknown_family" in capsys.readouterr().out


def test_validate_vaes_fails_when_entry_is_malformed(tmp_path, capsys):
    # Slice out exactly one entry's required keys: coverage is fine, structure is not.
    text = _REGISTRY_TEXT.partition("[vae.sdxl]\n")[0] + '[vae.sdxl]\ndirectory = "checkpoints/vae/sdxl"\n'
    bad = tmp_path / "malformed.toml"
    bad.write_text(text, encoding="utf-8")
    assert main(["validate-vaes", "--registry", str(bad)]) == 1
    assert "missing_field" in capsys.readouterr().out
