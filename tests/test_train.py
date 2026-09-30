"""Phase 9: the trainer's settled contract -- one run, three files, loadable weights.

CPU only: `torch.cuda.is_available` is patched off so the run never needs a GPU,
and the global seed is pinned because the trainer deliberately does not seed. Both
datasets are temp-dir stand-ins for what Phases 6-7 write -- real PNGs, a real
splits.json, a real manifest.jsonl -- so PairDataset takes them exactly as it takes
a cached corpus. No VAE, no generated pixel, no test split.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest
import torch
from safetensors.torch import load_file

from image_humanizer.dataset import PairDataset, SplitFile
from image_humanizer.image_io import save_png
from image_humanizer.model import ResidualUNet
from image_humanizer.train import train

FAMILIES = ("sd15", "sdxl")
# (original byte, reconstruction byte) per split per family. The reconstruction is
# darker than the original, as a VAE round-trip leaves it, and the ~40-byte gap is
# orders of magnitude above anything two 1e-3 AdamW steps move -- so "the loss went
# down" is a claim about training, not about 8-bit quantisation.
VALUES: dict[str, dict[str, tuple[int, int]]] = {
    "train": {"sd15": (200, 160), "sdxl": (210, 170)},
    "validation": {"sd15": (120, 84), "sdxl": (130, 94)},
}
CONFIG = """\
[model]
base_width = 8
channel_multipliers = [1, 2, 4, 8]

[optim]
lr = 0.001
batch_size = 2
grad_clip = 1.0
steps = 2
"""
EXPECTED_FILES = {"best.safetensors", "last.safetensors", "model_config.json"}
# A fixed batch with no seed and no fixture order behind it: the same bytes every run.
BATCH = torch.arange(2 * 3 * 8 * 8, dtype=torch.float32).reshape(2, 3, 8, 8).div_(255.0)


def _grey(byte: int) -> torch.Tensor:
    # 8x8 clears the model's reflect-pad floor (H, W >= 5) and is divisible by 8.
    return torch.full((3, 8, 8), byte / 255.0)


def _write_processed(root: Path, values: dict[str, dict[str, tuple[int, int]]]) -> Path:
    """PNGs plus the two metadata files PairDataset reads. One group per split."""
    processed = root / "processed"
    rows: list[dict[str, Any]] = []
    group_to_split: dict[str, str] = {}
    for split, families in values.items():
        group = f"g_{split}"
        group_to_split[group] = split
        for family, (original, reconstruction) in families.items():
            sid = f"s_{split}_{family}"
            home = processed / "groups" / sid
            save_png(_grey(original), home / "original.png")
            save_png(_grey(reconstruction), home / f"{family}.png")
            rows.append(
                {
                    "group_id": group,
                    "source_id": sid,
                    "split": split,
                    "family": family,
                    "clean_target": f"groups/{sid}/original.png",
                    "reconstruction": f"groups/{sid}/{family}.png",
                }
            )
    (processed / "splits.json").write_text(
        SplitFile(canon_version="1", group_to_split=group_to_split).to_json() + "\n",
        encoding="utf-8",
    )
    (processed / "manifest.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8"
    )
    return processed


def _train(name: str, values: dict[str, dict[str, tuple[int, int]]], tmp_path_factory) -> SimpleNamespace:
    root = tmp_path_factory.mktemp(name)
    processed = _write_processed(root, values)
    config = root / "train.toml"
    config.write_text(CONFIG, encoding="utf-8")
    out = root / "run"

    torch.manual_seed(0)
    with patch.object(torch.cuda, "is_available", return_value=False):
        code = train(processed, config, out)

    return SimpleNamespace(
        code=code,
        out=out,
        train=PairDataset(processed, "train"),
        validation=PairDataset(processed, "validation"),
    )


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    return _train("phase9", VALUES, tmp_path_factory)


@pytest.fixture(scope="module")
def floor_run(tmp_path_factory):
    # Identical validation pairs. Validation loss is then exactly the size of the
    # correction the model applies, and that correction grows every step by
    # construction (each step moves it toward the train target, on the same
    # pixels), so the lowest-validation state is the first one. That makes the
    # selection observable from the two checkpoints on disk, with nothing
    # patched inside the trainer.
    values = {"train": VALUES["train"], "validation": {"sd15": (160, 160), "sdxl": (170, 170)}}
    return _train("phase9_floor", values, tmp_path_factory)


def _reloaded(path: Path) -> ResidualUNet:
    model = ResidualUNet(8)
    model.load_state_dict(load_file(path))  # strict: proves key and shape coverage
    return model.eval()


def _charbonnier(model: ResidualUNet | None, dataset: PairDataset) -> float:
    """The run's objective, recomputed from disk. `None` is a freshly initialized
    model: its head is zeros, so it emits the reconstruction untouched."""
    with torch.inference_mode():
        total = 0.0
        for i in range(len(dataset)):
            item = dataset[i]
            recon = item["reconstruction"][None]
            original = item["original"][None]
            pred = recon if model is None else model(recon)
            total += float(torch.sqrt((pred - original) ** 2 + 1e-6).mean())
        return total / len(dataset)


def test_one_run_returns_zero_and_writes_exactly_three_files(run):
    assert run.code == 0
    assert {path.name for path in run.out.iterdir()} == EXPECTED_FILES


def test_model_config_is_the_model_that_was_built(run):
    assert json.loads((run.out / "model_config.json").read_text(encoding="utf-8")) == {
        "base_width": 8,
        "channel_multipliers": [1, 2, 4, 8],
        "groups": 8,
    }


def test_last_reloads_into_a_fresh_model_and_repeats_its_own_output(run):
    first = _reloaded(run.out / "last.safetensors")
    second = _reloaded(run.out / "last.safetensors")

    with torch.inference_mode():
        out_first = first(BATCH)
        assert torch.equal(out_first, second(BATCH)), "same saved state, different output"
        assert out_first.shape == BATCH.shape
        # The kept checkpoint is the same architecture at a different point in the
        # run, so Phase 10 loads it exactly the same way.
        assert _reloaded(run.out / "best.safetensors")(BATCH).shape == BATCH.shape


def test_selection_reads_validation_not_the_train_loss(floor_run):
    best = _reloaded(floor_run.out / "best.safetensors")
    last = _reloaded(floor_run.out / "last.safetensors")

    assert _charbonnier(best, floor_run.validation) < _charbonnier(last, floor_run.validation), (
        "best is not the lowest-validation state"
    )
    # The same two states, the opposite verdict on train: the kept checkpoint is the
    # worse one, so its selection cannot have come from the training loss.
    assert _charbonnier(best, floor_run.train) > _charbonnier(last, floor_run.train)
    # And two real steps did lower the training objective below what any freshly
    # initialized (zero-head) model scores on the same pairs.
    assert _charbonnier(last, floor_run.train) < _charbonnier(None, floor_run.train)
