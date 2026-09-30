"""Phase 6: one cached reconstruction per resolved family per source.

Family is the outer loop, so one loaded VAE serves a whole family. The
manifest is appended and never rewritten: a (source_id, family) pair is
written once, whether it is already recorded or its PNG already exists.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from image_humanizer.image_io import file_sha256, load_png, save_png
from image_humanizer.reconstruct import load_vae, reconstruct
from image_humanizer.vae_registry import validate_registry


def generate_pairs(output: Path) -> int:
    """Append one reconstruction pair per resolved family per source."""
    manifest = output / "manifest.jsonl"
    rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    sources = [r for r in rows if "family" not in r]
    paired = {(r["source_id"], r["family"]) for r in rows if "family" in r}

    with manifest.open("a", encoding="utf-8") as f:
        for report in validate_registry():  # family outer: one VAE serves a family
            if not report.ok:
                for finding in report.findings:
                    print(f"{report.name}: {finding.code}: {finding.message}")
                continue
            model = load_vae(report.resolved)  # outside try: a load failure propagates
            try:
                for row in sources:
                    target = output / "groups" / row["source_id"] / f"{report.name}.png"
                    if (row["source_id"], report.name) in paired or target.exists():
                        continue  # already recorded, or already reconstructed: never redo
                    chw = load_png(output / row["clean_target"])
                    out = reconstruct(report.name, chw[None], model=model)
                    save_png(out[0], target)  # batch dropped, CHW is what save_png takes
                    f.write(
                        json.dumps(
                            {
                                "group_id": row["group_id"],
                                "source_id": row["source_id"],
                                "split": row["split"],
                                "family": report.name,
                                "clean_target": row["clean_target"],
                                "reconstruction": target.relative_to(output).as_posix(),
                                "reconstruction_sha256": file_sha256(target),
                            },
                            sort_keys=True,
                        )
                        + "\n"
                    )
                    paired.add((row["source_id"], report.name))
            finally:
                del model
                torch.cuda.empty_cache()
    return 0
