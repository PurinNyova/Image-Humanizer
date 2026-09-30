"""Phase 5/6: direct local-only VAE loading and reconstruction.

Two functions, no framework, no cache. `load_vae(resolved)` picks the diffusers
class by `resolved.adapter`; `reconstruct(family, rgb_01, model=...)` is the one
reconstruction path Phase 6 reuses for every family. Every byte comes from paths
`vae_registry.resolve()` already named on disk: `local_files_only=True`, never a
repo id, never a download.

Never applied here: `scaling_factor`, `shift_factor`, or `denoiser_scale_factor`
(SD 1.5's 0.18215 is a denoiser convention, not a reconstruction scale), and
never `latent_dist.sample()` -- direct reconstruction must be deterministic, so
the posterior mode is the value. No tiling, no VAE slicing, no resize: a
geometry that does not come back identical is an error, not something to fix.
"""

from __future__ import annotations

import json

import torch
from torch.nn import functional as F

from image_humanizer import vae_registry
from image_humanizer.vae_registry import Resolved

_RESOLVED = "_image_humanizer_resolved"
_QWEN = "diffusers_autoencoder_kl_qwen_image"


def load_vae(resolved: Resolved) -> torch.nn.Module:
    """Load one `Resolved` as float32, frozen, eval, with `resolved` attached
    for `reconstruct`'s geometry. Local files only."""
    adapter = resolved.adapter
    if adapter == "diffusers_autoencoder_kl":
        from diffusers import AutoencoderKL

        model = AutoencoderKL.from_pretrained(resolved.directory, local_files_only=True)
    elif adapter == _QWEN:
        from diffusers import AutoencoderKLQwenImage

        model = AutoencoderKLQwenImage.from_pretrained(resolved.directory, local_files_only=True)
    elif adapter == "diffusers_autoencoder_kl_wan":
        from diffusers import AutoencoderKLWan

        model = AutoencoderKLWan.from_pretrained(resolved.directory, local_files_only=True)
    elif adapter == "diffusers_autoencoder_kl_hunyuan_video":
        model = _load_hunyuan(resolved)
    elif adapter == "diffusers_autoencoder_kl_ltx_video":
        from diffusers import AutoencoderKLLTXVideo

        model = AutoencoderKLLTXVideo.from_pretrained(resolved.directory, local_files_only=True)
    else:
        raise NotImplementedError(
            f"no loader for adapter {adapter!r} (VAE {resolved.name!r}); load_vae covers only "
            "diffusers_autoencoder_kl, diffusers_autoencoder_kl_qwen_image, "
            "diffusers_autoencoder_kl_wan, diffusers_autoencoder_kl_hunyuan_video and "
            "diffusers_autoencoder_kl_ltx_video."
        )

    model.to(torch.float32).requires_grad_(False).eval()
    setattr(model, _RESOLVED, resolved)
    return model


def _load_hunyuan(resolved: Resolved) -> torch.nn.Module:
    """Hunyuan's checkpoint is a registry-named `.pt` and its config declares the
    unexported `AutoencoderKLCausal3D`, so neither half is loadable by
    `from_pretrained` alone: build from the config, then load the local file."""
    from diffusers import AutoencoderKLHunyuanVideo

    model = AutoencoderKLHunyuanVideo.from_config(json.loads(resolved.config.read_text(encoding="utf-8")))
    state = torch.load(resolved.checkpoint, map_location="cpu", weights_only=True)
    for wrapper in ("state_dict", "model"):
        if isinstance(state, dict) and isinstance(state.get(wrapper), dict):
            state = state[wrapper]
            break
    model.load_state_dict(state, strict=True)
    return model


def reconstruct(family: str, rgb_01: torch.Tensor, model: torch.nn.Module | None = None) -> torch.Tensor:
    """float32 B x 3 x H x W in [0,1] -> the same shape, dtype and range."""
    if model is None:
        report = vae_registry.resolve(family)
        if not report.ok:
            finding = report.findings[0]
            # vae_registry.Finding is a frozen dataclass, not an Exception, so it
            # cannot be raised; its code and message are the error text verbatim.
            raise RuntimeError(f"{finding.code}: {finding.message}")
        model = load_vae(report.resolved)

    resolved = getattr(model, _RESOLVED, None)
    if resolved is not None:
        multiple = resolved.spatial_multiple
        # Qwen declares temporal=false in the registry but its class is a video
        # VAE: 5D in, 5D out, regardless of what the metadata says.
        needs_5d = bool(resolved.temporal) or resolved.adapter == _QWEN
    elif family == "sd15":
        multiple, needs_5d = 8, False  # native_image, no attached Resolved
    else:
        raise NotImplementedError(
            f"reconstruct(family={family!r}) on a model without an attached Resolved is only "
            "defined for 'sd15' (spatial_multiple=8, native_image). Build the family with "
            "load_vae(vae_registry.resolve(name).resolved) so its geometry travels with it."
        )

    if rgb_01.dim() != 4 or rgb_01.shape[1] != 3:
        raise ValueError(f"rgb_01 must be B x 3 x H x W, got {tuple(rgb_01.shape)}")
    if rgb_01.dtype != torch.float32:
        raise ValueError(f"rgb_01 must be float32, got {rgb_01.dtype}")
    if not bool(torch.isfinite(rgb_01).all()):
        raise ValueError("rgb_01 holds NaN or Inf")
    if float(rgb_01.min()) < 0.0 or float(rgb_01.max()) > 1.0:
        raise ValueError(f"rgb_01 must be in [0, 1], got [{float(rgb_01.min())}, {float(rgb_01.max())}]")

    b, c, h, w = rgb_01.shape
    pad_h, pad_w = -h % multiple, -w % multiple
    x = F.pad(rgb_01, (0, pad_w, 0, pad_h), mode="replicate") if (pad_h or pad_w) else rgb_01
    x = x.mul(2.0).sub_(1.0)  # [0,1] -> the models' [-1,1]
    if needs_5d:
        x = x.unsqueeze(2)  # B 3 1 H W
    with torch.inference_mode():
        latents = model.encode(x).latent_dist.mode()
        out = model.decode(latents).sample
        if needs_5d:
            out = out[:, :, 0]  # the one frame we asked for
        out = out.div_(2.0).add_(0.5).clamp_(0.0, 1.0)
    out = out[:, :, :h, :w]  # remove only the padding we added

    if tuple(out.shape) != (b, c, h, w):
        raise ValueError(
            f"VAE {family!r} returned {tuple(out.shape)} for a {b} x {c} x {h} x {w} input; geometry "
            "must match exactly, so this is a wrong model, never a resize."
        )
    if out.dtype != torch.float32:
        raise ValueError(f"VAE {family!r} reconstruction must be float32, got {out.dtype}")
    if not bool(torch.isfinite(out).all()):
        raise ValueError(f"VAE {family!r} reconstruction contains NaN/Inf")
    return out
