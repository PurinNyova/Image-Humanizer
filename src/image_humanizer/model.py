"""Phase 8: residual pixel-space U-Net.

One 4-level residual U-Net mapping a VAE reconstruction to the clean
original. The residual head is zero-initialized, so `model(x) == x` exactly
at init. Forward pads H/W up to a multiple of 8 by reflection, runs the net,
adds the residual to the padded input, and crops back. No clamping: that
belongs to inference, not here. No attention, conditioning, pretrained
weights, or VAE import.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def _block(channels: int, groups: int) -> nn.Sequential:
    """GroupNorm -> SiLU -> Conv3x3, same width in and out."""
    return nn.Sequential(
        nn.GroupNorm(groups, channels),
        nn.SiLU(),
        nn.Conv2d(channels, channels, 3, padding=1),
    )


class ResidualUNet(nn.Module):
    def __init__(
        self,
        base_width: int = 64,
        channel_multipliers: tuple[int, ...] = (1, 2, 4, 8),
        groups: int = 8,
    ) -> None:
        super().__init__()
        widths = [base_width * m for m in channel_multipliers]
        if len(widths) != 4 or any(w <= 0 for w in widths):
            raise ValueError(
                f"expected exactly 4 positive widths, got {widths} "
                f"from base_width={base_width}, channel_multipliers={tuple(channel_multipliers)}"
            )
        if groups < 1 or any(w % groups for w in widths):
            raise ValueError(
                f"each width must be divisible by groups={groups}, got {widths}"
            )

        self.stem = nn.Conv2d(3, widths[0], 3, padding=1)
        self.enc = nn.ModuleList(
            [nn.Sequential(_block(w, groups), _block(w, groups)) for w in widths]
        )
        self.down = nn.ModuleList(
            [nn.Conv2d(widths[i], widths[i + 1], 3, stride=2, padding=1) for i in range(3)]
        )
        # Decoder walks levels 2, 1, 0: upsample, project to the skip width, add it.
        self.up = nn.ModuleList(
            [nn.Conv2d(widths[i + 1], widths[i], 3, padding=1) for i in (2, 1, 0)]
        )
        self.dec = nn.ModuleList(
            [nn.Sequential(_block(widths[i], groups), _block(widths[i], groups)) for i in (2, 1, 0)]
        )
        self.head = nn.Conv2d(widths[0], 3, 3, padding=1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4 or x.shape[1] != 3 or x.shape[2] == 0 or x.shape[3] == 0:
            raise ValueError(f"expected float32 Bx3xHxW with non-empty H, W, got {tuple(x.shape)}")
        if x.dtype is not torch.float32:
            raise ValueError(f"expected float32 input, got {x.dtype}")

        h, w = x.shape[2], x.shape[3]
        pad_h, pad_w = -h % 8, -w % 8
        for name, size, pad in (("H", h, pad_h), ("W", w, pad_w)):
            if pad and pad >= size:
                raise ValueError(
                    f"reflect padding {pad} on {name}={size} is invalid; padding must be "
                    f"smaller than that axis (use H, W >= 5, or pad the image first)"
                )
        padded = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")

        hfeat = self.stem(padded)
        skips: list[torch.Tensor] = []
        for i in range(4):
            hfeat = self.enc[i](hfeat)
            skips.append(hfeat)
            if i < 3:
                hfeat = self.down[i](hfeat)
        for j, i in enumerate((2, 1, 0)):
            hfeat = F.interpolate(hfeat, scale_factor=2, mode="bilinear", align_corners=False)
            hfeat = self.dec[j](self.up[j](hfeat) + skips[i])

        return (padded + self.head(hfeat))[:, :, :h, :w]
