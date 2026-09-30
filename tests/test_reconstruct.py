"""Phase 5: one direct reconstruction path.

Every weightless test drives a tiny fake VAE, so nothing here reads a checkpoint.
The one real test resolves `sd15` and skips with the registry's finding text
until the weights are placed locally.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from image_humanizer.reconstruct import reconstruct
from image_humanizer.vae_registry import resolve


class _Posterior:
    """latent_dist stand-in. `mode` only: `.sample()` existing but fatal proves
    the posterior is never sampled."""

    def __init__(self, latent: torch.Tensor) -> None:
        self._latent = latent

    def mode(self) -> torch.Tensor:
        return self._latent

    def sample(self) -> torch.Tensor:
        raise AssertionError("posterior.mode(), never posterior.sample()")


class _FakeVAE:
    """Records what encode() was handed, so the normalize-and-pad path is checked."""

    def __init__(self) -> None:
        self.seen: list[tuple[int, ...]] = []

    def encode(self, x: torch.Tensor) -> SimpleNamespace:
        assert x.ndim == 4 and x.dtype is torch.float32, f"expected BCHW float32, got {tuple(x.shape)}"
        assert x.shape[2] % 8 == 0 and x.shape[3] % 8 == 0, "padded to spatial_multiple 8"
        assert -1.001 <= float(x.min()) and float(x.max()) <= 1.001, "normalized to [-1, 1]"
        self.seen.append(tuple(x.shape))
        return SimpleNamespace(latent_dist=_Posterior(x[:, :1].expand(-1, 4, -1, -1)))

    def decode(self, latent: torch.Tensor) -> SimpleNamespace:
        return SimpleNamespace(sample=latent[:, :3] * 0.5)  # DecoderOutput.sample is a field


def test_reconstruct_keeps_geometry_and_is_deterministic_with_a_fake_vae():
    vae = _FakeVAE()
    rgb = torch.rand(2, 3, 5, 11)  # neither side a multiple of 8, so both get padded

    out = reconstruct("sd15", rgb, model=vae)
    again = reconstruct("sd15", rgb, model=vae)

    assert out.shape == rgb.shape
    assert out.dtype is torch.float32
    assert torch.isfinite(out).all()
    assert 0.0 <= float(out.min()) and float(out.max()) <= 1.0
    assert torch.equal(out, again)
    # Padded up to the spatial multiple, both calls, and cropped back on the way out.
    assert vae.seen == [(2, 3, 8, 16), (2, 3, 8, 16)]


def test_only_sd15_is_implemented():
    # model= is a fake and nothing local resolves, so this is the family gate,
    # not a resolution failure.
    with pytest.raises(NotImplementedError):
        reconstruct("sdxl", torch.rand(1, 3, 8, 8), model=_FakeVAE())


def test_sd15_reconstruct():
    report = resolve("sd15")
    if not report.ok:
        first = report.findings[0]
        pytest.skip(f"{first.code}: {first.message}")

    rgb = torch.rand(1, 3, 64, 64)
    out = reconstruct("sd15", rgb, model=None)
    again = reconstruct("sd15", rgb, model=None)

    assert out.shape == rgb.shape
    assert out.dtype is torch.float32
    assert torch.isfinite(out).all()
    assert 0.0 <= float(out.min()) and float(out.max()) <= 1.0
    assert torch.equal(out, again)
