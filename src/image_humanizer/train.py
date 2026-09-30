"""Phase 9: straight-line training of the residual U-Net on cached pairs.

One loop, one loss, fp32 only: AdamW, gradient clipping, exact Charbonnier,
validation over the whole validation split after every step, `best.safetensors`
whenever validation is strictly below the prior best, `last.safetensors` at the
end. The train loader is cycled by recreating its iterator, so `steps` is a step
count and not an epoch count. The test split is never read. Deliberately absent:
scheduler, AMP, accumulation, resume, seeding, and any config object -- add them
when a run proves it needs them.
"""

from __future__ import annotations

import json
import tomllib
from pathlib import Path

import torch
from safetensors.torch import save_file
from torch.utils.data import DataLoader

from image_humanizer.dataset import PairDataset
from image_humanizer.model import ResidualUNet

GROUPS = 8


def train(data: Path, config: Path, out: Path) -> int:
    """Train for `[optim] steps` steps and write checkpoints plus model_config.json."""
    with Path(config).open("rb") as f:
        raw = tomllib.load(f)
    optim = raw["optim"]
    lr = float(optim["lr"])
    batch_size = int(optim["batch_size"])
    grad_clip = float(optim["grad_clip"])
    steps = int(optim["steps"])
    for name, value in (("batch_size", batch_size), ("steps", steps), ("lr", lr)):
        if value <= 0:
            raise ValueError(f"[optim] {name} must be positive, got {value}")
    if grad_clip < 0:
        raise ValueError(f"[optim] grad_clip must be nonnegative, got {grad_clip}")

    base_width = int(raw["model"]["base_width"])
    channel_multipliers = [int(m) for m in raw["model"]["channel_multipliers"]]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ResidualUNet(base_width, tuple(channel_multipliers), GROUPS).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    train_loader = DataLoader(PairDataset(data, "train"), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(PairDataset(data, "validation"), batch_size=batch_size, shuffle=False)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)

    best = float("inf")
    batches = iter(train_loader)
    for step in range(1, steps + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        try:
            batch = next(batches)
        except StopIteration:  # cycle: steps counts steps, not epochs
            batches = iter(train_loader)
            batch = next(batches)
        pred = model(batch["reconstruction"].to(device))
        loss = torch.sqrt((pred - batch["original"].to(device)) ** 2 + 1e-6).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        model.eval()
        weighted = 0.0
        seen = 0
        with torch.no_grad():
            for val in val_loader:
                pred = model(val["reconstruction"].to(device))
                count = pred.shape[0]
                weighted += float(torch.sqrt((pred - val["original"].to(device)) ** 2 + 1e-6).mean()) * count
                seen += count
        val_loss = weighted / seen

        if val_loss < best:
            best = val_loss
            save_file({k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()}, out / "best.safetensors")
        print(f"step {step}/{steps}  train {loss.item():.6f}  val {val_loss:.6f}")

    save_file({k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()}, out / "last.safetensors")
    (out / "model_config.json").write_text(
        json.dumps(
            {"base_width": base_width, "channel_multipliers": channel_multipliers, "groups": GROUPS},
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return 0
