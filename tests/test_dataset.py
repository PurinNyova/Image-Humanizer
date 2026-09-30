"""Phase 7: PairDataset over the cached PNGs.

No VAE and no generated pixel: everything on disk is a temp-file stand-in for
what Phase 6 would have written. Every source and every (family, source) gets a
unique constant grey, so a returned tensor identifies itself by pixel value.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import torch

from image_humanizer.dataset import PairDataset, SplitFile
from image_humanizer.image_io import file_sha256, save_png

FAMILIES = ("sd15", "sdxl")
# One byte per source, and one offset per family. 8-bit PNG quantisation keeps
# all five apart, so the constants name both the split and the source.
SOURCE_BYTES = {"train": (20, 25, 30), "validation": (35,), "test": (40,)}
FAMILY_OFFSET = {"sd15": 2, "sdxl": 4}


def _grey(byte: int) -> torch.Tensor:
    return torch.full((3, 4, 4), byte / 255.0)


@pytest.fixture
def processed(tmp_path):
    root = tmp_path / "processed"
    rows: list[dict] = []
    source_to_split: dict[str, str] = {}
    group_to_split: dict[str, str] = {}
    lookup: dict[int, tuple[str, str, str]] = {}
    for split, byte_list in SOURCE_BYTES.items():
        group = f"g_{split}"
        group_to_split[group] = split
        for index, byte in enumerate(byte_list):
            sid = f"s_{split}_{index}"
            source_to_split[sid] = split
            directory = root / "groups" / sid
            save_png(_grey(byte), directory / "original.png")
            rows.append(
                {
                    "group_id": group,
                    "source_id": sid,
                    "split": split,
                    "clean_target": f"groups/{sid}/original.png",
                    "canon_version": "1",
                }
            )
            lookup[byte] = (split, str(index), "")
            for family, offset in FAMILY_OFFSET.items():
                pair = directory / f"{family}.png"
                save_png(_grey(byte + offset), pair)
                rows.append(
                    {
                        "group_id": group,
                        "source_id": sid,
                        "split": split,
                        "family": family,
                        "clean_target": f"groups/{sid}/original.png",
                        "reconstruction": f"groups/{sid}/{family}.png",
                        "reconstruction_sha256": file_sha256(pair),
                    }
                )
                lookup[byte + offset] = (split, str(index), family)

    (root / "splits.json").write_text(
        SplitFile(canon_version="1", source_to_split=source_to_split, group_to_split=group_to_split).to_json() + "\n",
        encoding="utf-8",
    )
    (root / "manifest.jsonl").write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows), encoding="utf-8")
    return SimpleNamespace(dir=root, lookup=lookup)


def _key(tensor: torch.Tensor, lookup) -> tuple[str, str, str]:
    return lookup[round(float(tensor[0, 0, 0]) * 255)]


def test_items_are_float32_pixels_in_range_and_not_the_original(processed):
    for split in ("train", "validation"):
        item = PairDataset(processed.dir, split)[0]
        assert set(item) == {"reconstruction", "original"}
        for tensor in item.values():
            assert tensor.shape == (3, 4, 4)
            assert tensor.dtype is torch.float32
            assert torch.isfinite(tensor).all()
            assert 0.0 <= float(tensor.min()) and float(tensor.max()) <= 1.0
        assert not torch.equal(item["reconstruction"], item["original"])


def test_flat_length_is_families_times_the_longest_family(processed):
    # len(families) * max(sources per family); two families, three train sources.
    assert len(PairDataset(processed.dir, "train")) == 2 * 3
    assert len(PairDataset(processed.dir, "validation")) == 2 * 1
    assert len(PairDataset(processed.dir, "test")) == 2 * 1


def test_each_family_cycles_through_every_one_of_its_sources(processed):
    ds = PairDataset(processed.dir, "train")

    per_family: dict[str, list[str]] = {}
    for i in range(len(ds)):
        split, source, family = _key(ds[i]["reconstruction"], processed.lookup)
        assert split == "train"
        assert _key(ds[i]["original"], processed.lookup)[:2] == (split, source)
        per_family.setdefault(family, []).append(source)

    assert sorted(per_family) == list(FAMILIES)
    for family, sources in per_family.items():
        assert len(sources) == 3, "each family appears max(sources) times per epoch"
        assert sorted(set(sources)) == ["0", "1", "2"], "every source once"


def test_splits_see_only_their_own_sources(processed):
    for split, expected in SOURCE_BYTES.items():
        ds = PairDataset(processed.dir, split)
        seen = {round(float(ds[i]["original"][0, 0, 0]) * 255) for i in range(len(ds))}
        assert seen == set(expected), f"{split} leaked another split's sources"
    # The constants are what makes the assertion above non-vacuous.
    all_bytes = [b for bs in SOURCE_BYTES.values() for b in bs]
    assert len(set(all_bytes)) == len(all_bytes)


def test_one_missing_family_png_shrinks_only_that_family(processed):
    (processed.dir / "groups" / "s_train_2" / "sdxl.png").unlink()

    ds = PairDataset(processed.dir, "train")  # dropped silently, no raise
    assert len(ds) == 2 * 3, "the longest family still sets the flat length"
    per_family: dict[str, set[str]] = {}
    for i in range(len(ds)):
        _, source, family = _key(ds[i]["reconstruction"], processed.lookup)
        per_family.setdefault(family, set()).add(source)
    assert per_family == {"sd15": {"0", "1", "2"}, "sdxl": {"0", "1"}}


def test_emptying_a_family_raises_naming_the_split_and_the_family(processed):
    for path in (processed.dir / "groups").glob("s_train_*/sdxl.png"):
        path.unlink()

    with pytest.raises(ValueError) as exc:
        PairDataset(processed.dir, "train")
    message = str(exc.value)
    assert "train" in message and "sdxl" in message


def test_a_changed_canon_version_raises_instead_of_resplitting(processed):
    path = processed.dir / "splits.json"
    frozen = SplitFile.from_json(path.read_text(encoding="utf-8"))
    path.write_text(
        SplitFile(
            canon_version="2",
            source_to_split=frozen.source_to_split,
            group_to_split=frozen.group_to_split,
        ).to_json()
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError) as exc:
        PairDataset(processed.dir, "train")
    assert "canon" in str(exc.value).lower()
