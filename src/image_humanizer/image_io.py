"""Canonical image pipeline. Phase 2. Decisions D1-D4, D6 in docs/phase0_decisions.md."""

from __future__ import annotations

import hashlib
import io
import tomllib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageCms, ImageOps

CANON_VERSION = "1"
PHASH_SIZE = 8
PHASH_HIGHFREQ = 4
PHASH_BANDS = 4
_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}


@dataclass(frozen=True)
class CanonConfig:
    canon_version: str
    resolution: int
    alpha_background: tuple[float, float, float]
    resize_filter: str
    skip_resize_if_square: bool
    phash_max_distance: int
    parent_max_fraction: float
    splits: dict[str, float]


def load_canon_config(path: Path) -> CanonConfig:
    with Path(path).open("rb") as f:
        raw = tomllib.load(f)
    splits = {k: float(v) for k, v in raw["splits"].items()}
    if abs(sum(splits.values()) - 1.0) > 1e-9:
        raise ValueError(f"split fractions must sum to 1.0, got {sum(splits.values())}")
    parent_max_fraction = float(raw["grouping"].get("parent_max_fraction", 0.2))
    if not 0.0 <= parent_max_fraction <= 1.0:
        raise ValueError(f"parent_max_fraction must be in [0, 1], got {parent_max_fraction}")
    return CanonConfig(
        canon_version=str(raw["canon_version"]),
        resolution=int(raw["resolution"]),
        alpha_background=tuple(float(c) for c in raw["alpha_background"]),
        resize_filter=str(raw["resize_filter"]),
        skip_resize_if_square=bool(raw["skip_resize_if_square"]),
        phash_max_distance=int(raw["grouping"]["phash_max_distance"]),
        parent_max_fraction=parent_max_fraction,
        splits=splits,
    )


@dataclass(frozen=True)
class Canonical:
    """One decoded, EXIF-corrected, sRGB, cropped source."""

    source_id: str
    tensor: torch.Tensor  # 3 x R x R, float32, [0, 1]
    phash: int
    size: tuple[int, int]  # (width, height) after canonicalization
    mode: str
    resized: bool
    icc_converted: bool
    composited: bool

    @property
    def phash_hex(self) -> str:
        return f"{self.phash:016x}"


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _to_srgb(path: Path, im: Image.Image) -> tuple[Image.Image, bool]:
    if im.mode in {"I", "F"} or im.mode.startswith("I;16"):
        # Pillow's convert() clips these to 8 bits silently; refuse instead.
        raise ValueError(
            f"{path}: {im.mode} high-bit grayscale is unsupported; "
            "normalize it to 8-bit L before ingest"
        )
    icc = im.info.get("icc_profile")
    if icc is not None:
        src = ImageCms.ImageCmsProfile(io.BytesIO(icc))
        dst = ImageCms.createProfile("sRGB")
        if im.mode in {"RGBA", "LA"}:
            # LA cannot be transformed directly; move luminance to RGB, keep alpha.
            alpha = im.getchannel("A")
            rgb = ImageCms.profileToProfile(im.convert("RGB"), src, dst, outputMode="RGB")
            return Image.merge("RGBA", (*rgb.split(), alpha)), True
        return ImageCms.profileToProfile(im, src, dst, outputMode="RGB"), True
    if im.info.get("transparency") is not None and im.mode in {"P", "RGB"}:
        return im.convert("RGBA"), True
    if im.mode not in {"RGB", "RGBA", "L", "LA"}:
        return im.convert("RGB"), True
    if im.mode == "L":
        return im.convert("RGB"), True
    if im.mode == "LA":
        return im.convert("RGBA"), True
    return im, False


def _composite_over_white(im: Image.Image, background: tuple[float, float, float]) -> Image.Image:
    bg = tuple(int(round(c * 255)) for c in background)
    base = Image.new("RGB", im.size, bg)
    return Image.alpha_composite(base.convert("RGBA"), im.convert("RGBA")).convert("RGB")


def _dct2(x: np.ndarray) -> np.ndarray:
    """Separable DCT-II, unnormalized (the scale factor is irrelevant to a hash)."""
    n = x.shape[0]
    k = np.arange(n).reshape(-1, 1)
    i = np.arange(n).reshape(1, -1)
    basis = np.cos(np.pi * (2 * i + 1) * k / (2 * n))
    return basis @ x @ basis.T


def compute_phash(gray_2d: np.ndarray) -> int:
    """64-bit pHash from an H x W float array in [0, 1]."""
    n = PHASH_SIZE * PHASH_HIGHFREQ
    small = Image.fromarray((gray_2d * 255).round().astype(np.uint8), mode="L").resize(
        (n, n), Image.Resampling.LANCZOS
    )
    low = _dct2(np.asarray(small, dtype=np.float64))[:PHASH_SIZE, :PHASH_SIZE]
    bits = low > np.median(low.flat[1:])  # drop the DC term
    return int("".join("1" if b else "0" for b in bits.flat), 2)


def canonicalize(path: Path, cfg: CanonConfig) -> Canonical:
    """Decode once, then apply D2 then D3 exactly once each."""
    with Image.open(path) as raw:
        raw.load()
        mode = raw.mode
        exif_oriented = ImageOps.exif_transpose(raw)
        icc_converted = False
        if exif_oriented.mode == "CMYK":
            # Pillow's CMYK->RGB is a blind guess; refuse rather than guess.
            if exif_oriented.info.get("icc_profile"):
                exif_oriented, icc_converted = _to_srgb(path, exif_oriented)
            else:
                raise ValueError(f"{path}: CMYK without an ICC profile, refusing to guess the conversion")
        else:
            exif_oriented, icc_converted = _to_srgb(path, exif_oriented)
        composited = False
        if exif_oriented.mode in {"RGBA", "LA"}:
            exif_oriented = _composite_over_white(exif_oriented, cfg.alpha_background)
            composited = True

    res = cfg.resolution
    resized = False
    if not (cfg.skip_resize_if_square and exif_oriented.size == (res, res)):
        short = min(exif_oriented.size)
        if short != res:
            scale = res / short
            new = (max(res, round(exif_oriented.width * scale)), max(res, round(exif_oriented.height * scale)))
            filt = getattr(Image.Resampling, cfg.resize_filter.upper())
            exif_oriented = exif_oriented.resize(new, filt)
            resized = True

    w, h = exif_oriented.size
    left, top = (w - res) // 2, (h - res) // 2
    exif_oriented = exif_oriented.crop((left, top, left + res, top + res))

    arr = np.asarray(exif_oriented, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(np.ascontiguousarray(arr.transpose(2, 0, 1)))
    digest = hashlib.sha256(tensor.numpy().tobytes() + cfg.canon_version.encode()).hexdigest()
    phash = compute_phash(arr.mean(axis=2))
    return Canonical(
        source_id=digest,
        tensor=tensor,
        phash=phash,
        size=exif_oriented.size,
        mode=mode,
        resized=resized,
        icc_converted=icc_converted,
        composited=composited,
    )


def iter_source_files(root: Path) -> list[Path]:
    return sorted(p for p in Path(root).rglob("*") if p.suffix.lower() in _IMAGE_SUFFIXES and p.is_file())


def load_png(path: Path) -> torch.Tensor:
    """Inverse of save_png: 3 x H x W float32 in [0, 1]."""
    with Image.open(path) as im:
        arr = np.asarray(im.convert("RGB"), dtype=np.uint8)
    return torch.from_numpy(np.ascontiguousarray(arr.transpose(2, 0, 1))).float().div_(255.0)


def save_png(tensor: torch.Tensor, path: Path) -> None:
    arr = (tensor.clamp(0, 1) * 255).round().to(torch.uint8).permute(1, 2, 0).numpy()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr, mode="RGB").save(path, format="PNG", compress_level=6)
