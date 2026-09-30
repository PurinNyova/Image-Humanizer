import argparse
import json
import tomllib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REGISTRY = REPO_ROOT / "configs" / "vae_registry.toml"
DEFAULT_PREPROCESS = REPO_ROOT / "configs" / "preprocess.toml"
VAE_NAMES = [
    "qwen_image",
    "sdxl",
    "flux",
    "chroma",
    "sd15",
    "sd3",
    "wan",
    "minimax_h3_image",
    "hunyuan",
    "ltx",
]


def load_registry(path: Path) -> dict:
    with path.open("rb") as f:
        return tomllib.load(f)["vae"]


def cmd_validate_vaes(args: argparse.Namespace) -> int:
    registry = load_registry(args.registry)
    print(f"registry: {args.registry}")
    print(f"entries: {len(registry)}")
    for name in VAE_NAMES:
        entry = registry.get(name)
        if entry is None:
            print(f"  {name:18} MISSING")
        else:
            print(f"  {name:18} enabled={entry.get('enabled', True)}")
    return 0


def cmd_preprocess(args: argparse.Namespace) -> int:
    from image_humanizer.dataset import Source, assign_groups, freeze_splits
    from image_humanizer.image_io import (
        canonicalize,
        file_sha256,
        iter_source_files,
        load_canon_config,
        save_png,
    )

    cfg = load_canon_config(args.config)
    files = iter_source_files(args.input)
    if not files:
        print(f"no images under {args.input}")
        return 1
    if cfg.canon_version != args.canon_version:
        raise SystemExit(
            f"configs/preprocess.toml canon_version={cfg.canon_version} does not match "
            f"--canon-version {args.canon_version}. One of them is stale."
        )

    sources: list[Source] = []
    seen: dict[str, str] = {}  # source_id -> first path that produced it
    for path in files:
        canon = canonicalize(path, cfg)
        if canon.source_id in seen:
            continue  # same pixels under a second path: one source, one group
        seen[canon.source_id] = str(path)
        sources.append(Source(path=path, canon=canon, byte_sha256=file_sha256(path)))
    assign_groups(sources, args.input, cfg)
    frozen = freeze_splits(sources, args.output, cfg)

    out = Path(args.output)
    rows = []
    for s in sources:
        target = out / "groups" / s.canon.source_id / "original.png"
        save_png(s.canon.tensor, target)
        rows.append(
            {
                "group_id": s.group_id,
                "source_id": s.canon.source_id,
                "split": s.split,
                "clean_target": target.relative_to(out).as_posix(),
                "phash": s.canon.phash_hex,
                "sha256": s.byte_sha256,
                "size": list(s.canon.size),
                "source_mode": s.canon.mode,
                "resized": s.canon.resized,
                "icc_converted": s.canon.icc_converted,
                "alpha_composited": s.canon.composited,
                "source_path": s.path.relative_to(args.input).as_posix(),
                "canon_version": cfg.canon_version,
            }
        )
    with (out / "manifest.jsonl").open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")
    (out / "dataset.json").write_text(
        json.dumps(
            {
                "canon_version": cfg.canon_version,
                "resolution": cfg.resolution,
                "sources": len(sources),
                "duplicate_paths": len(files) - len(sources),
                "groups": len(frozen.group_to_split),
                "splits": {k: sum(1 for s in sources if s.split == k) for k in set(frozen.source_to_split.values())},
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"files scanned   : {len(files)}")
    print(f"sources decoded : {len(sources)}")
    print(f"groups          : {len(frozen.group_to_split)}")
    for name in ("train", "validation", "test"):
        print(f"  {name:12}: {sum(1 for v in frozen.source_to_split.values() if v == name)}")
    print(f"out             : {out}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="image-humanizer")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("validate-vaes", help="resolve and validate VAE registry entries")
    p.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    p.set_defaults(func=cmd_validate_vaes)

    p = sub.add_parser("preprocess", help="canonicalize sources and freeze leakage-safe splits")
    p.add_argument("--input", type=Path, required=True, help="directory of source images")
    p.add_argument("--output", type=Path, default=REPO_ROOT / "data" / "processed")
    p.add_argument("--config", type=Path, default=DEFAULT_PREPROCESS)
    p.add_argument("--canon-version", default="1", help="must match preprocess.toml; guards a stale config")
    p.set_defaults(func=cmd_preprocess)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
