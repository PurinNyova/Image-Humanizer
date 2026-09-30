"""Leakage-safe source grouping and frozen splits. Phase 2. Decisions D4, D6."""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from image_humanizer.image_io import PHASH_BANDS, PHASH_SIZE, CanonConfig, Canonical

SPLIT_NAMES = ("train", "validation", "test")


@dataclass
class Source:
    path: Path
    canon: Canonical
    byte_sha256: str
    group_id: str = ""
    split: str = ""


def _phash_bands(phash: int) -> list[int]:
    width = PHASH_SIZE * PHASH_SIZE // PHASH_BANDS
    return [(phash >> (i * width)) & ((1 << width) - 1) for i in range(PHASH_BANDS)]


def _hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


class _Union:
    def __init__(self, items: list[str]) -> None:
        self.parent = {i: i for i in items}

    def find(self, x: str) -> str:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            # Deterministic: the lexicographically smaller id becomes the root,
            # so group_id never depends on input order.
            lo, hi = sorted((ra, rb))
            self.parent[hi] = lo


def assign_groups(sources: list[Source], root: Path, cfg: CanonConfig) -> None:
    """Union sources by every D4 key, then name each group by its smallest id."""
    union = _Union([s.canon.source_id for s in sources])

    by_bytes: dict[str, list[str]] = defaultdict(list)
    for s in sources:
        by_bytes[s.byte_sha256].append(s.canon.source_id)
    for ids in by_bytes.values():
        for other in ids[1:]:
            union.union(ids[0], other)

    # Pixel-identical: same canonical pixels means the same source_id, so this is
    # already covered by collapsing on source_id. Recorded here for the manifest.
    by_band: list[dict[int, list[Source]]] = [defaultdict(list) for _ in range(PHASH_BANDS)]
    for s in sources:
        for band, value in enumerate(_phash_bands(s.canon.phash)):
            by_band[band][value].append(s)
    for table in by_band:
        for bucket in table.values():
            # ponytail: O(n^2) inside one band; bands are 16 bits so buckets stay
            # tiny. Move to a BK-tree if a band ever holds thousands of images.
            for i, a in enumerate(bucket):
                for b in bucket[i + 1 :]:
                    if _hamming(a.canon.phash, b.canon.phash) <= cfg.phash_max_distance:
                        union.union(a.canon.source_id, b.canon.source_id)

    by_parent: dict[str, list[str]] = defaultdict(list)
    for s in sources:
        by_parent[str(Path(s.path).parent.relative_to(root))].append(s.canon.source_id)
    n = max(1, len(sources))
    for ids in by_parent.values():
        unique = sorted(set(ids))
        if len(unique) / n <= cfg.parent_max_fraction:
            for other in unique[1:]:
                union.union(unique[0], other)

    roots: dict[str, str] = {}
    for s in sources:
        root_id = union.find(s.canon.source_id)
        roots[root_id] = min(roots.get(root_id, root_id), root_id)
    for s in sources:
        s.group_id = roots[union.find(s.canon.source_id)]
        s.split = ""


def assign_splits(sources: list[Source], cfg: CanonConfig) -> None:
    """Order groups by a stable hash, then fill exact target counts."""
    groups = sorted({s.group_id for s in sources})
    total = len(groups)
    order: list[str] = []
    assigned = 0
    for i, name in enumerate(SPLIT_NAMES):
        count = total - assigned if i == len(SPLIT_NAMES) - 1 else int(round(total * cfg.splits[name]))
        order += [name] * count
        assigned += count
    lookup = dict(zip(groups, order))
    for s in sources:
        s.split = lookup[s.group_id]


@dataclass
class SplitFile:
    canon_version: str
    source_to_split: dict[str, str] = field(default_factory=dict)
    group_to_split: dict[str, str] = field(default_factory=dict)
    parent_to_split: dict[str, str] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(
            {
                "canon_version": self.canon_version,
                "source_to_split": dict(sorted(self.source_to_split.items())),
                "group_to_split": dict(sorted(self.group_to_split.items())),
                "parent_to_split": dict(sorted(self.parent_to_split.items())),
            },
            indent=2,
            sort_keys=True,
        )

    @classmethod
    def from_json(cls, text: str) -> "SplitFile":
        raw = json.loads(text)
        return cls(
            canon_version=raw["canon_version"],
            source_to_split=raw["source_to_split"],
            group_to_split=raw["group_to_split"],
            parent_to_split=raw.get("parent_to_split", {}),
        )


def freeze_splits(sources: list[Source], out_dir: Path, cfg: CanonConfig) -> SplitFile:
    """Write splits.json. An existing file is authoritative and must match exactly."""
    out = Path(out_dir) / "splits.json"
    assign_splits(sources, cfg)
    frozen = SplitFile(
        canon_version=cfg.canon_version,
        source_to_split={s.canon.source_id: s.split for s in sources},
        group_to_split={s.group_id: s.split for s in sources},
        parent_to_split={str(Path(s.path).parent): s.split for s in sources},
    )
    if out.is_file():
        existing = SplitFile.from_json(out.read_text(encoding="utf-8"))
        if existing.canon_version != cfg.canon_version:
            raise ValueError(
                f"{out} was frozen for canon_version {existing.canon_version}, "
                f"config says {cfg.canon_version}. Every source_id changed; delete the file to re-split."
            )
        if existing.source_to_split != frozen.source_to_split:
            differing = sum(
                1
                for k, v in frozen.source_to_split.items()
                if existing.source_to_split.get(k, v) != v
            )
            raise ValueError(
                f"{out} does not describe this corpus: {len(existing.source_to_split)} frozen "
                f"sources, {len(frozen.source_to_split)} scanned, {differing} mismatched. "
                "A frozen split is a promise; delete the file to re-split deliberately."
            )
        return existing
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(frozen.to_json() + "\n", encoding="utf-8")
    return frozen
