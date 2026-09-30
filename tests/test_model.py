"""Phase 8: residual U-Net identity, geometry, and learnability.

CPU only, no checkpoint, no VAE. The head is zero-initialised, so every
forward at init must reproduce its input bitwise -- that is what makes the
network start as the identity and is the property all of training rests on.
Geometry is checked at an odd size (reflect pad then crop back) and above one
megapixel, which is where a fixed-stride or fixed-shape implementation would
fall over.
"""

from __future__ import annotations

import torch

from image_humanizer.model import ResidualUNet


def test_default_model_is_the_identity_at_init():
    # Exact, not approximate: the residual head is zeros, so `padded + 0`
    # returns the input bit-for-bit.
    torch.manual_seed(0)
    x = torch.randn(2, 3, 8, 8)

    model = ResidualUNet()
    with torch.no_grad():
        assert torch.equal(model(x), x)


def test_odd_geometry_comes_back_at_the_exact_input_shape():
    # 37 and 53 both need 3 rows/cols of reflection; a size-rounded output
    # would be 40x56 and silently wrong for every downstream crop.
    torch.manual_seed(1)
    x = torch.randn(2, 3, 37, 53)

    model = ResidualUNet()
    with torch.no_grad():
        y = model(x)
    assert y.shape == (2, 3, 37, 53)
    assert torch.equal(y, x)


def test_above_one_megapixel_runs_and_stays_the_identity():
    # 1,001,000 pixels: proves the net is shape-agnostic and that padding and
    # cropping round-trip at scale. Narrow base_width keeps it cheap on CPU --
    # this is a shape/capability check, not a claim about the default model's
    # memory use.
    torch.manual_seed(2)
    x = torch.randn(1, 3, 1001, 1000)

    model = ResidualUNet(base_width=8).eval()
    with torch.inference_mode():
        y = model(x)
    assert y.shape == (1, 3, 1001, 1000)
    assert torch.equal(y, x)


def test_one_adamw_step_lowers_the_charbonnier_loss():
    torch.manual_seed(3)
    model = ResidualUNet(base_width=8)
    x = torch.randn(2, 3, 16, 16)
    y = torch.randn(2, 3, 16, 16)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    before = torch.sqrt((model(x) - y) ** 2 + 1e-6).mean().item()

    optimizer.zero_grad()
    torch.sqrt((model(x) - y) ** 2 + 1e-6).mean().backward()
    optimizer.step()

    after = torch.sqrt((model(x) - y) ** 2 + 1e-6).mean().item()
    assert after < before, f"{after} !< {before}"
