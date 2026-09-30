import argparse
import json
from pathlib import Path

from image_humanizer.vae_registry import load_registry, resolve, validate_registry

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REGISTRY = REPO_ROOT / "configs" / "vae_registry.toml"
DEFAULT_PREPROCESS = REPO_ROOT / "configs" / "preprocess.toml"
DEFAULT_TRAIN = REPO_ROOT / "configs" / "train.toml"


def cmd_validate_vaes(args: argparse.Namespace) -> int:
    """Phase 3 pre-load validation. Read-only: no weights are loaded, so no
    substantial GPU memory is allocated. All checking lives in vae_registry."""
    if args.checkpoint and not args.family:
        args._parser.error("--checkpoint requires --family: an override applies to exactly one entry")
    registry = load_registry(args.registry)
    if args.family:
        reports = [
            resolve(args.family, registry, override=args.checkpoint)
        ]
    else:
        reports = validate_registry(registry)

    print(f"registry : {args.registry}")
    print(f"root     : {REPO_ROOT}  (relative `directory` values resolve here)")
    failed = 0
    for r in reports:
        enabled = bool(registry.get(r.name, {}).get("enabled", True))
        if r.ok:
            v = r.resolved
            picked = v.checkpoint.name if v.checkpoint else f"index + {len(v.shards)} shards"
            print(f"  {r.name:18} OK    source={v.source} checkpoint={picked}")
        else:
            print(f"  {r.name:18} {'SKIP' if not enabled else 'FAIL'}  enabled={enabled}")
            for f in r.findings:
                print(f"  {'':18}   {f.code}: {f.message}")
        for d in r.deferred:
            print(f"  {r.name:18}   deferred: {d}")
        if enabled and r.findings:
            failed += 1
    print(f"{len(reports)} entries: {sum(r.ok for r in reports)} resolved, {failed} enabled with errors")
    return 1 if failed else 0


def cmd_preprocess(args: argparse.Namespace) -> int:
    from dataclasses import replace

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
    seen: set[str] = set()  # byte-hash: byte-identical files are never decoded
    ids: set[str] = set()   # source_id: re-encoded pixel-identical images dedupe here
    out = Path(args.output)
    for path in files:
        file_hash = file_sha256(path)
        if file_hash in seen:
            continue  # byte-identical under a second path: never decoded
        canon = canonicalize(path, cfg)
        if canon.source_id in ids:
            seen.add(file_hash)
            continue  # same pixels re-encoded: same source_id, one group
        seen.add(file_hash)
        ids.add(canon.source_id)
        # Save immediately so no full canonical tensors are retained corpus-wide.
        save_png(canon.tensor, out / "groups" / canon.source_id / "original.png")
        sources.append(Source(path=path, canon=replace(canon, tensor=canon.tensor[:0]), byte_sha256=file_hash))
    assign_groups(sources, args.input, cfg)
    frozen = freeze_splits(sources, args.output, args.input, cfg)

    rows = []
    for s in sources:
        target = out / "groups" / s.canon.source_id / "original.png"
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


def cmd_generate_pairs(args: argparse.Namespace) -> int:
    from image_humanizer.generate_pairs import generate_pairs

    return generate_pairs(args.output)


def cmd_train(args: argparse.Namespace) -> int:
    from image_humanizer.train import train

    return train(args.data, args.config, args.out)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="image-humanizer")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser(
        "validate-vaes",
        help="resolve every registry entry read-only: checkpoint/config existence, sha256, adapter metadata",
        epilog="Local-only: no network access and nothing is downloaded. No weights are loaded, so no substantial GPU memory is allocated.",
    )
    p.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY, help="registry TOML (default: %(default)s)")
    p.add_argument("--family", help="check only this registry entry (default: every entry)")
    p.add_argument(
        "--checkpoint",
        type=Path,
        help="explicit checkpoint override for --family; takes precedence over registry `checkpoint`",
    )
    p.set_defaults(func=cmd_validate_vaes, _parser=p)

    p = sub.add_parser("preprocess", help="canonicalize sources and freeze leakage-safe splits")
    p.add_argument("--input", type=Path, required=True, help="directory of source images")
    p.add_argument("--output", type=Path, default=REPO_ROOT / "data" / "processed")
    p.add_argument("--config", type=Path, default=DEFAULT_PREPROCESS)
    p.add_argument("--canon-version", default="1", help="must match preprocess.toml; guards a stale config")
    p.set_defaults(func=cmd_preprocess)

    p = sub.add_parser("generate-pairs", help="phase 6: generate humanized/clean image pairs")
    p.add_argument("--output", type=Path, default=REPO_ROOT / "data" / "processed")
    p.set_defaults(func=cmd_generate_pairs)

    p = sub.add_parser("train", help="train the residual U-Net on cached reconstruction pairs")
    p.add_argument("--data", type=Path, default=REPO_ROOT / "data" / "processed")
    p.add_argument("--config", type=Path, default=DEFAULT_TRAIN)
    p.add_argument("--out", type=Path, required=True)
    p.set_defaults(func=cmd_train)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
