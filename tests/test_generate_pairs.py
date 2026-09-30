"""Phase 6: one reconstruction per enabled resolved family per source.

No weights: `validate_registry`, `load_vae` and `reconstruct` are replaced with
fakes, so the acceptance check is about the manifest and the printed findings,
not about a VAE.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import image_humanizer.generate_pairs as gp
from image_humanizer.generate_pairs import generate_pairs
from image_humanizer.image_io import file_sha256, load_png, save_png
from image_humanizer.vae_registry import Finding, Report, Resolved

FAMILY = "sd15"
DISABLED = "minimax_h3_image"
UNRESOLVED = "sdxl"
PAIR_FIELDS = {
    "group_id",
    "source_id",
    "split",
    "family",
    "clean_target",
    "reconstruction",
    "reconstruction_sha256",
}


def _resolved(name: str, directory: Path) -> Resolved:
    return Resolved(
        name=name,
        adapter="diffusers_autoencoder_kl",
        directory=directory,
        checkpoint=None,
        shards=(),
        config=None,
        source="override",
        reconstruction_method="native_image",
        latent_channels=4,
        spatial_multiple=8,
        temporal=False,
        temporal_compression=None,
        denoiser_scale_factor=None,
        class_name="AutoencoderKL",
        latent_transform=(),
        hashes={},
        unverified=(),
    )


@pytest.fixture
def tree(tmp_path, monkeypatch):
    processed = tmp_path / "processed"
    source_row = {
        "group_id": "g0",
        "source_id": "s0",
        "split": "train",
        "clean_target": "groups/s0/original.png",
        "canon_version": "1",
    }
    save_png(torch.rand(3, 6, 6, generator=torch.Generator().manual_seed(0)), processed / "groups" / "s0" / "original.png")
    (processed / "manifest.jsonl").write_text(json.dumps(source_row, sort_keys=True) + "\n", encoding="utf-8")

    ok = Report(FAMILY, _resolved(FAMILY, tmp_path))
    reports = [
        ok,
        Report(DISABLED, None, (Finding("unsupported", "T=1 is outside the documented regime"),)),
        Report(UNRESOLVED, None, (Finding("missing_checkpoint", "registry checkpoint 'x.safetensors' is missing"),)),
    ]

    sentinel = object()
    calls: list[str] = []
    loaded: list[Resolved] = []

    def fake_validate_registry():
        return reports

    def fake_load_vae(resolved):
        loaded.append(resolved)
        return sentinel

    def fake_reconstruct(family, bchw, model=None):
        assert model is sentinel, "one load_vae per family, passed as model="
        calls.append(family)
        # A rerun that regenerated would write different pixels, so the second
        # run's byte-identical PNG proves nothing was rebuilt.
        return bchw * 0.5 if len(calls) == 1 else bchw * 0.25

    monkeypatch.setattr(gp, "validate_registry", fake_validate_registry)
    monkeypatch.setattr(gp, "load_vae", fake_load_vae)
    monkeypatch.setattr(gp, "reconstruct", fake_reconstruct)

    return SimpleNamespace(
        dir=processed,
        original=processed / "groups" / "s0" / "original.png",
        pair_png=processed / "groups" / "s0" / f"{FAMILY}.png",
        row=source_row,
        ok=ok,
        calls=calls,
        loaded=loaded,
        tmp_path=tmp_path,
    )


def test_one_pair_png_and_one_seven_field_row(tree, capsys):
    assert generate_pairs(tree.dir) == 0

    assert sorted(p.name for p in (tree.dir / "groups" / "s0").glob("*.png")) == ["original.png", f"{FAMILY}.png"]
    lines = (tree.dir / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0]) == tree.row, "the source row is untouched"
    pair = json.loads(lines[1])
    assert set(pair) == PAIR_FIELDS
    for field in ("group_id", "split", "clean_target", "source_id"):
        assert pair[field] == tree.row[field]
    assert pair["family"] == FAMILY
    assert pair["reconstruction"] == f"groups/s0/{FAMILY}.png"
    assert pair["reconstruction_sha256"] == file_sha256(tree.pair_png)
    assert tree.calls == [FAMILY]
    assert tree.loaded == [tree.ok.resolved]

    # The PNG on disk is exactly what reconstruct returned.
    expected = tree.tmp_path / "expected.png"
    save_png(load_png(tree.original) * 0.5, expected)
    assert torch.equal(load_png(tree.pair_png), load_png(expected))

    out = capsys.readouterr().out
    assert "unsupported: T=1 is outside the documented regime" in out
    assert "missing_checkpoint: registry checkpoint 'x.safetensors' is missing" in out


def test_a_second_run_appends_nothing_and_changes_nothing(tree):
    generate_pairs(tree.dir)
    manifest = (tree.dir / "manifest.jsonl").read_bytes()
    png = tree.pair_png.read_bytes()

    generate_pairs(tree.dir)

    assert (tree.dir / "manifest.jsonl").read_bytes() == manifest, "append-only, never rewritten"
    assert json.loads(manifest.splitlines()[0]) == tree.row
    assert tree.pair_png.read_bytes() == png, "an existing pair is not regenerated"
    assert tree.calls == [FAMILY], "no second reconstruction"
    for skipped in (DISABLED, UNRESOLVED):
        assert not (tree.dir / "groups" / "s0" / f"{skipped}.png").exists()
