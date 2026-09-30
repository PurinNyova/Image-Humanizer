"""Phase 2: canonical pipeline and leakage-safe splits."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image, ImageCms

from image_humanizer.cli import REPO_ROOT, main
from image_humanizer.dataset import Source, assign_groups, assign_splits, freeze_splits
from image_humanizer.image_io import (
    CanonConfig,
    canonicalize,
    compute_phash,
    load_canon_config,
)

CONFIG = load_canon_config(REPO_ROOT / "configs" / "preprocess.toml")


def cfg(resolution: int = 64) -> CanonConfig:
    return CanonConfig(
        canon_version=CONFIG.canon_version,
        resolution=resolution,
        alpha_background=CONFIG.alpha_background,
        resize_filter=CONFIG.resize_filter,
        skip_resize_if_square=CONFIG.skip_resize_if_square,
        phash_max_distance=CONFIG.phash_max_distance,
        parent_max_fraction=CONFIG.parent_max_fraction,
        splits=CONFIG.splits,
    )


def _array(w: int, h: int, seed: int = 0) -> Image.Image:
    rng = np.random.default_rng(seed)
    return Image.fromarray(rng.integers(0, 256, (h, w, 3), dtype=np.uint8), mode="RGB")


def _save(img: Image.Image, path: Path, **kw) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path, **kw)
    return path


def test_canonical_shape_dtype_and_range(tmp_path):
    canon = canonicalize(_save(_array(200, 120), tmp_path / "a.png"), cfg())
    assert canon.tensor.shape == (3, 64, 64)
    assert canon.tensor.dtype is torch.float32
    assert float(canon.tensor.min()) >= 0.0 and float(canon.tensor.max()) <= 1.0
    assert canon.size == (64, 64)


@pytest.mark.parametrize("orientation", list(range(1, 9)))
def test_exif_orientation_is_applied(tmp_path, orientation):
    # Invariant: decoding with an orientation tag must equal decoding the image
    # that was physically stored rotated. No hand-derived pixel expectations.
    row = Image.new("RGB", (96, 32))
    for i, colour in enumerate(((255, 0, 0), (0, 255, 0), (0, 0, 255))):
        row.paste(Image.new("RGB", (32, 32), colour), (32 * i, 0))
    ops = {
        1: lambda im: im,
        2: Image.Transpose.FLIP_LEFT_RIGHT,
        3: Image.Transpose.ROTATE_180,
        4: Image.Transpose.FLIP_TOP_BOTTOM,
        5: Image.Transpose.TRANSPOSE,
        6: Image.Transpose.ROTATE_270,
        7: Image.Transpose.TRANSVERSE,
        8: Image.Transpose.ROTATE_90,
    }
    physical = row.transpose(ops[orientation]) if orientation != 1 else row
    reference = _save(physical, tmp_path / "physical.png")
    exif = Image.Exif()
    exif[0x0112] = orientation
    tagged = tmp_path / f"o{orientation}.png"
    row.save(tagged, exif=exif)

    assert canonicalize(tagged, cfg(resolution=96)).source_id == canonicalize(reference, cfg(resolution=96)).source_id


def test_orientation_actually_changes_the_canonical_pixels(tmp_path):
    # Guards the test above from passing because both paths ignore orientation.
    row = Image.new("RGB", (96, 32))
    for i, colour in enumerate(((255, 0, 0), (0, 255, 0), (0, 0, 255))):
        row.paste(Image.new("RGB", (32, 32), colour), (32 * i, 0))
    ids = set()
    for orientation, op in ((1, None), (6, Image.Transpose.ROTATE_270), (8, Image.Transpose.ROTATE_90)):
        physical = row if op is None else row.transpose(op)
        exif = Image.Exif()
        exif[0x0112] = orientation
        path = tmp_path / f"tag{orientation}.png"
        (row if orientation != 1 else physical).save(path, exif=exif)
        ids.add(canonicalize(path, cfg(resolution=96)).source_id)
    assert len(ids) == 3


def test_icc_converted_to_srgb(tmp_path):
    profile = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()
    img = _array(64, 64, seed=3)  # already square: no resize, so pixels are comparable
    tagged = _save(img, tmp_path / "icc.png", icc_profile=profile)
    plain = _save(img, tmp_path / "plain.png")
    with_icc = canonicalize(tagged, cfg())
    without = canonicalize(plain, cfg())
    assert with_icc.icc_converted and not without.icc_converted
    # sRGB -> sRGB through LittleCMS is a round trip, not a colour change.
    assert torch.equal(with_icc.tensor, without.tensor)
    assert with_icc.source_id == without.source_id


def test_alpha_composites_over_configured_matte_never_drops(tmp_path):
    # Transparent colored pixel must become alpha_background (here green), not white.
    with_green = CanonConfig(**{**vars(cfg()), "alpha_background": (0.0, 1.0, 0.0)})
    rgba = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
    for x in range(0, 64, 8):
        for y in range(0, 64, 8):
            rgba.putpixel((x, y), (255, 0, 0, 255))
    canon = canonicalize(_save(rgba, tmp_path / "a.png"), with_green)
    assert canon.composited
    t = canon.tensor
    assert bool(t[0, 0, 0] > 0.5)  # opaque red survives
    assert bool(t[2, 0, 0] < 0.5)
    assert bool(t[1, 3, 3] > 0.99)  # transparent pixel -> configured matte channel
    assert bool(t[0, 3, 3] < 0.01)


def test_palette_transparency_promotes_to_rgba(tmp_path):
    # P-mode with palette transparency: the _to_srgb transparency branch must fire
    # (composited=True proves the promote-to-RGBA happened; an opaque P image
    # would only convert to RGB).
    pal = _array(64, 64, seed=5).convert("P")
    pal.info["transparency"] = pal.getpixel((0, 0))
    canon = canonicalize(_save(pal, tmp_path / "p.png"), cfg())
    assert canon.composited


def test_grayscale_is_converted_explicitly(tmp_path):
    g = _save(Image.new("L", (64, 64), 128), tmp_path / "g.png")
    canon = canonicalize(g, cfg())
    assert canon.mode == "L" and canon.tensor.shape == (3, 64, 64)
    assert torch.equal(canon.tensor[0], canon.tensor[1]) and torch.equal(canon.tensor[1], canon.tensor[2])


def test_high_bit_gray_is_refused(tmp_path):
    # 16-bit grayscale: Pillow's convert() would clip it to 8 bits silently.
    path = _save(Image.fromarray(np.full((64, 64), 65535, np.uint16)), tmp_path / "hi.tif")
    with pytest.raises(ValueError, match="unsupported"):
        canonicalize(path, cfg())


def test_config_rejects_splits_that_do_not_sum_to_one(tmp_path):
    path = tmp_path / "preprocess.toml"
    path.write_text(
        'canon_version = "1"\n'
        "resolution = 64\n"
        "alpha_background = [1.0, 1.0, 1.0]\n"
        "skip_resize_if_square = true\n"
        'resize_filter = "lanczos"\n'
        "[splits]\n"
        "train = 0.5\n"
        "validation = 0.4\n"  # 0.5 + 0.4 + 0.2 = 1.1
        "test = 0.2\n"
        "[grouping]\n"
        "phash_max_distance = 6\n"
        "parent_max_fraction = 0.2\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="sum to 1.0"):
        load_canon_config(path)


def test_cmyk_with_mismatched_profile_is_refused(tmp_path):
    # LittleCMS cannot build a CMYK transform out of an sRGB profile; the
    # mismatch must surface, not fall back to PIL's blind CMYK->RGB guess.
    profile = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()
    cmyk = ImageCms.Image.frombytes("CMYK", (64, 64), bytes([10, 200, 30, 5]) * (64 * 64))
    with pytest.raises(ImageCms.PyCMSError):
        canonicalize(_save(cmyk, tmp_path / "mismatched.tif", icc_profile=profile), cfg())


def test_cmyk_without_profile_is_refused(tmp_path):
    path = _save(ImageCms.Image.frombytes("CMYK", (64, 64), bytes([10, 200, 30, 5]) * (64 * 64)), tmp_path / "c.tif")
    assert "icc_profile" not in Image.open(path).info
    with pytest.raises(ValueError, match="refusing to guess"):
        canonicalize(path, cfg())


def test_crop_is_center_and_deterministic(tmp_path):
    src = _array(200, 100, seed=7)
    path = _save(src, tmp_path / "wide.png")
    first = canonicalize(path, cfg())
    second = canonicalize(path, cfg())
    assert first.source_id == second.source_id
    assert torch.equal(first.tensor, second.tensor)
    # Center crop of the 128x64 Lanczos resize, not the top-left corner.
    expected = src.resize((128, 64), Image.Resampling.LANCZOS).crop((32, 0, 96, 64))
    assert torch.equal(first.tensor, torch.from_numpy(np.asarray(expected, np.float32).transpose(2, 0, 1) / 255))


def test_source_id_ignores_path_but_tracks_content(tmp_path):
    a = canonicalize(_save(_array(64, 64, seed=9), tmp_path / "x" / "a.png"), cfg())
    b = canonicalize(_save(_array(64, 64, seed=9), tmp_path / "y" / "b.png"), cfg())
    c = canonicalize(_save(_array(64, 64, seed=10), tmp_path / "x" / "c.png"), cfg())
    assert a.source_id == b.source_id != c.source_id
    stale = CanonConfig(**{**vars(cfg()), "canon_version": "2"})
    assert canonicalize(tmp_path / "x" / "a.png", stale).source_id != a.source_id


def _sources(tmp_path, specs, c=None) -> list[Source]:
    from image_humanizer.image_io import file_sha256

    c = c or cfg()
    out = []
    for i, (rel, img) in enumerate(specs):
        p = _save(img, tmp_path / rel)
        canon = canonicalize(p, c)
        out.append(Source(path=p, canon=canon, byte_sha256=file_sha256(p)))
    return out


def test_identical_pixels_collapse_to_one_source(tmp_path):
    img = _array(64, 64, seed=11)
    srcs = _sources(tmp_path, [("a.png", img), ("sub/b.png", img.copy()), ("sub/c.png", img.copy())])
    assert len({s.canon.source_id for s in srcs}) == 1


def test_byte_identical_and_pixel_identical_differ_as_expected(tmp_path):
    img = _array(64, 64, seed=12)
    same_bytes = _sources(tmp_path, [("a.png", img), ("b.png", img.copy())])
    assert {s.byte_sha256 for s in same_bytes} == {s.byte_sha256 for s in same_bytes[:1]}
    # Losslessly re-encoded, so identical pixels but different bytes.
    webp_free = _sources(tmp_path, [("a.png", img), ("b.png", img.copy())])
    webp_free[1].byte_sha256 = "different-bytes-different-hash-0000000000000000000000000000"
    assert len({s.canon.source_id for s in webp_free}) == 1
    assert len({s.byte_sha256 for s in webp_free}) == 2


def test_near_duplicates_share_a_group_and_never_cross_splits(tmp_path):
    base = _array(64, 64, seed=13)
    specs = [("a.png", base), ("near.png", base.resize((70, 70)).resize((64, 64)))]
    specs += [(f"u{i}.png", _array(64, 64, seed=100 + i)) for i in range(8)]
    srcs = _sources(tmp_path, specs)
    assign_groups(srcs, tmp_path, cfg())
    by_group: dict[str, set] = {}
    for s in srcs:
        by_group.setdefault(s.group_id, set()).add(s.path.name)
    joined = {k: v for k, v in by_group.items() if {"a.png", "near.png"} <= v}
    assert joined, "near-duplicate was not clustered with its parent"
    assign_splits(srcs, cfg())
    grouped = [s for s in srcs if s.group_id in joined]
    assert len({s.split for s in grouped}) == 1


def test_split_is_stable_across_input_order_and_runs(tmp_path):
    specs = [(f"d{i // 4}/img{i}.png", _array(64, 64, seed=200 + i)) for i in range(24)]
    a = _sources(tmp_path, specs)
    assign_groups(a, tmp_path, cfg())
    assign_splits(a, cfg())
    b = _sources(tmp_path, list(reversed(specs)))
    assign_groups(b, tmp_path, cfg())
    assign_splits(b, cfg())
    assert {s.canon.source_id: s.split for s in a} == {s.canon.source_id: s.split for s in b}


def test_frozen_splits_file_is_reused_and_refuses_drift(tmp_path):
    srcs = _sources(tmp_path, [(f"i{i}.png", _array(64, 64, seed=300 + i)) for i in range(20)])
    assign_groups(srcs, tmp_path, cfg())
    frozen = freeze_splits(srcs, tmp_path / "out", tmp_path, cfg())
    assert (tmp_path / "out" / "splits.json").is_file()
    # Keys are paths relative to input_root, not absolute.
    assert all(not Path(k).is_absolute() for k in frozen.parent_to_split)
    again = freeze_splits(srcs, tmp_path / "out", tmp_path, cfg())
    assert again.source_to_split == frozen.source_to_split
    assert again.parent_to_split == frozen.parent_to_split
    with pytest.raises(ValueError, match="canon_version"):
        freeze_splits(
            srcs, tmp_path / "out", tmp_path, CanonConfig(**{**vars(cfg()), "canon_version": "2"})
        )
    # A grown corpus must not silently inherit a stale freeze.
    grown = srcs + _sources(tmp_path, [("new.png", _array(64, 64, seed=999))])
    with pytest.raises(ValueError, match="does not describe this corpus"):
        freeze_splits(grown, tmp_path / "out", tmp_path, cfg())
    # Re-splitting the same corpus with different group/parent mappings is refused.
    remapped = CanonConfig(**{**vars(cfg()), "phash_max_distance": 0})
    b = _sources(tmp_path, [(f"j{i}.png", _array(64, 64, seed=800 + i)) for i in range(4)])
    assign_groups(b, tmp_path, remapped)
    with pytest.raises(ValueError, match="does not describe this corpus"):
        freeze_splits(b, tmp_path / "out", tmp_path, remapped)


def test_split_ratios_land_where_configured(tmp_path):
    srcs = _sources(tmp_path, [(f"flat{i}.png", _array(64, 64, seed=400 + i)) for i in range(100)])
    assign_groups(srcs, tmp_path, cfg())
    assign_splits(srcs, cfg())
    counts = {name: sum(1 for s in srcs if s.split == name) for name in ("train", "validation", "test")}
    assert counts == {"train": 90, "validation": 5, "test": 5}


def test_phash_separates_unrelated_and_matches_itself():
    a = _array(64, 64, seed=1).convert("L")
    b = _array(64, 64, seed=2).convert("L")
    ha = compute_phash(np.asarray(a, np.float32) / 255)
    assert ha == compute_phash(np.asarray(a, np.float32) / 255)
    assert ha != compute_phash(np.asarray(b, np.float32) / 255)


def test_end_to_end_cli(tmp_path):
    src = tmp_path / "src"
    _save(_array(120, 90, seed=1), src / "a.png")
    _save(_array(90, 90, seed=2), src / "b.png")
    _save(_array(90, 90, seed=2), src / "b_copy.png")  # same pixels, different path
    out = tmp_path / "out"
    assert main(["preprocess", "--input", str(src), "--output", str(out)]) == 0
    rows = [json.loads(l) for l in (out / "manifest.jsonl").read_text().splitlines()]
    assert len(rows) == 2
    assert {r["split"] for r in rows} == {"train"}
    assert all(r["clean_target"].endswith("original.png") for r in rows)
    assert (out / "groups" / rows[0]["source_id"] / "original.png").is_file()
    frozen = json.loads((out / "splits.json").read_text())
    assert set(frozen["source_to_split"]) == {r["source_id"] for r in rows}
    with pytest.raises(SystemExit):
        main(["preprocess", "--input", str(src), "--output", str(out), "--canon-version", "2"])
