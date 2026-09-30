from pathlib import Path

import pytest

from image_humanizer.cli import VAE_NAMES, load_registry, main

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_package_imports():
    import image_humanizer

    assert image_humanizer.__version__


def test_registry_covers_every_family():
    registry = load_registry(REPO_ROOT / "configs" / "vae_registry.toml")
    assert sorted(registry) == sorted(VAE_NAMES)


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
    for name in VAE_NAMES:
        readme = REPO_ROOT / "checkpoints" / "vae" / name / "README.md"
        assert readme.is_file(), name


def test_help_exits_zero(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0
    assert "validate-vaes" in capsys.readouterr().out
