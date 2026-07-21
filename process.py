#!/usr/bin/env python3
"""
process.py — Post-process the scraped strain images, iPhone-Photos style.

For every image in ./in it applies a chain of adjustments (all optional,
all flags), optionally rotates it, downscales so neither side exceeds a cap,
and writes the result (WebP) into ./out with the same filename.

The colour/tone adjustments mirror the sliders in Apple's Photos "Edit" panel.
Each one takes a value in the iPhone range -100..+100 where 0 = no change,
except --saturation which keeps its original Pillow factor form (1.0 = off) for
back-compat. So the tool doubles as a general image-batch editor.

iPhone-style adjustments (default 0 = off):
    --exposure     overall lightness, multiplicative     (-100..100)
    --brightness   overall lightness, additive           (-100..100)
    --contrast     tonal contrast                        (-100..100)
    --highlights   recover / boost the bright regions    (-100..100)
    --shadows      lift / deepen the dark regions        (-100..100)
    --black-point  crush (+) or lift (-) the blacks       (-100..100)
    --vibrance     smart saturation (protects skin/sat)  (-100..100)
    --saturation-adj  plain saturation, iPhone scale     (-100..100)
    --warmth       colour temperature, warm(+)/cool(-)   (-100..100)
    --tint         green(-)/magenta(+) shift             (-100..100)
    --sharpness    edge sharpening                       (-100..100)
    --vignette     darken(+) / lighten(-) the corners    (-100..100)

Auto-enhance:
    --magic-wand   one-shot auto-optimise: white balance + auto levels +
                   vibrance + sharpen, computed per image. Runs before any
                   manual sliders, so you can wand-then-tweak.

Legacy / geometry / output flags:
    --saturation   Pillow saturation factor (1.05 = +5%, default; 1.0 = off)
    --rotate       "off" | "random" | <degrees>          (default: off)
    --max-size     cap the longest side in px            (default: 512; 0 = off)
    --quality      WebP quality 1-100                     (default: 90)
    --seed         reproducible random rotation angles (with --rotate random)
    --in / --out   source / destination dirs             (default: in/out)

Rotation is on-demand only: nothing is turned unless you pass --rotate.

Examples:
    python process.py --magic-wand                     # auto-optimise, no rotation
    python process.py --magic-wand --warmth 15         # auto, then a warm tweak
    python process.py --warmth 25 --vignette 30        # warm, moody
    python process.py --vibrance 40 --contrast 15
    python process.py --rotate random --seed 42        # opt in to random rotation
    python process.py --saturation 1.0                 # resize-only passthrough
    python process.py --overwrite                      # redo results that exist

By default a result that already exists is skipped (never overwritten). Pass
--overwrite to re-process and replace it.

Dependencies:
    pip install Pillow numpy
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance

# WebP is the scraped format, but accept common image types just in case.
SUPPORTED_EXT = {".webp", ".png", ".jpg", ".jpeg", ".bmp", ".tiff"}


# --------------------------------------------------------------------------- #
# Rotation
# --------------------------------------------------------------------------- #
def resolve_angle(rotate: str, rng: random.Random) -> float:
    """Turn the --rotate flag value into a concrete angle in degrees.

    "random" -> a random angle in [0, 360);  "off" -> 0;  "<number>" -> that number.
    """
    r = rotate.strip().lower()
    if r in ("off", "none", "no", "0"):
        return 0.0
    if r == "random":
        return rng.uniform(0, 360)
    try:
        return float(rotate)
    except ValueError:
        raise ValueError(f"--rotate must be 'random', 'off', or a number; got {rotate!r}")


# --------------------------------------------------------------------------- #
# iPhone-style colour / tone adjustments
#
# These operate on a float RGB array in [0, 1]. The alpha channel is carried
# through untouched. Every function is a no-op when its slider value is 0, so
# chaining them only costs work for the adjustments you actually asked for.
# --------------------------------------------------------------------------- #
def _s(v: float) -> float:
    """Map an iPhone slider (-100..100) to a signed unit strength (-1..1)."""
    return max(-100.0, min(100.0, v)) / 100.0


def adj_exposure(rgb: np.ndarray, v: float) -> np.ndarray:
    # Photographic stops: +100 ~ +1 stop (x2), -100 ~ -1 stop (x0.5).
    return rgb * (2.0 ** _s(v))


def adj_brightness(rgb: np.ndarray, v: float) -> np.ndarray:
    # Additive lift, up to +/-0.5 of full scale.
    return rgb + 0.5 * _s(v)


def adj_contrast(rgb: np.ndarray, v: float) -> np.ndarray:
    # Pivot around mid-grey; +100 doubles contrast, -100 flattens toward grey.
    f = 1.0 + _s(v)
    return (rgb - 0.5) * f + 0.5


def adj_highlights(rgb: np.ndarray, v: float) -> np.ndarray:
    # Weight the change toward bright pixels (luminance^2 mask).
    lum = _luma(rgb)
    mask = (lum ** 2)[..., None]
    return rgb + 0.5 * _s(v) * mask


def adj_shadows(rgb: np.ndarray, v: float) -> np.ndarray:
    # Weight the change toward dark pixels ((1-luminance)^2 mask).
    lum = _luma(rgb)
    mask = ((1.0 - lum) ** 2)[..., None]
    return rgb + 0.5 * _s(v) * mask


def adj_black_point(rgb: np.ndarray, v: float) -> np.ndarray:
    # Positive crushes blacks (raise the black floor); negative lifts them.
    s = _s(v)
    if s >= 0:
        lo = 0.25 * s               # new black floor
        return (rgb - lo) / max(1e-6, 1.0 - lo)
    lift = -0.25 * s
    return rgb * (1.0 - lift) + lift


def adj_saturation(rgb: np.ndarray, v: float) -> np.ndarray:
    # Plain saturation on the iPhone scale: mix toward/away from grey.
    lum = _luma(rgb)[..., None]
    return lum + (rgb - lum) * (1.0 + _s(v))


def adj_vibrance(rgb: np.ndarray, v: float) -> np.ndarray:
    # Smart saturation: boost low-saturation pixels more than already-vivid ones.
    lum = _luma(rgb)[..., None]
    sat = np.abs(rgb - lum).max(axis=-1, keepdims=True)   # 0 = grey, ~1 = vivid
    factor = 1.0 + _s(v) * (1.0 - sat)                     # less push where sat is high
    return lum + (rgb - lum) * factor


def adj_warmth(rgb: np.ndarray, v: float) -> np.ndarray:
    # Colour temperature: warm (+) pushes red up / blue down; cool (-) the reverse.
    s = _s(v) * 0.15
    out = rgb.copy()
    out[..., 0] += s      # R
    out[..., 2] -= s      # B
    return out


def adj_tint(rgb: np.ndarray, v: float) -> np.ndarray:
    # Green/magenta axis: + toward magenta (R,B up / G down), - toward green.
    s = _s(v) * 0.15
    out = rgb.copy()
    out[..., 0] += s * 0.5   # R
    out[..., 1] -= s         # G
    out[..., 2] += s * 0.5   # B
    return out


def adj_vignette(rgb: np.ndarray, v: float) -> np.ndarray:
    # Radial darkening (+) or lightening (-) toward the corners.
    h, w = rgb.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
    dist = np.sqrt(((xx - cx) / cx) ** 2 + ((yy - cy) / cy) ** 2)
    dist = np.clip(dist / np.sqrt(2.0), 0.0, 1.0)          # 0 center .. 1 corner
    s = _s(v)
    mask = 1.0 - s * (dist ** 2)                            # darken corners when s>0
    return rgb * mask[..., None]


def _luma(rgb: np.ndarray) -> np.ndarray:
    """Rec. 601 luma of an (H,W,3) float array -> (H,W)."""
    return rgb[..., 0] * 0.299 + rgb[..., 1] * 0.587 + rgb[..., 2] * 0.114


# The ordered pipeline: (attr name on args, function). Order roughly follows
# the way Photos stacks tone before colour before effects.
ADJUSTMENTS = [
    ("exposure", adj_exposure),
    ("brightness", adj_brightness),
    ("contrast", adj_contrast),
    ("highlights", adj_highlights),
    ("shadows", adj_shadows),
    ("black_point", adj_black_point),
    ("saturation_adj", adj_saturation),
    ("vibrance", adj_vibrance),
    ("warmth", adj_warmth),
    ("tint", adj_tint),
    ("vignette", adj_vignette),
]


def magic_wand(im: Image.Image) -> Image.Image:
    """Auto-optimise an image, like the "magic wand" in Photos.

    Per-image and fully automatic: gray-world white balance, a percentile
    contrast/levels stretch, then a gentle vibrance + sharpness lift. All stats
    are computed from the *opaque* pixels only, so the transparent background
    these strain cut-outs carry doesn't skew the result.
    """
    arr = np.asarray(im.convert("RGBA"), dtype=np.float32) / 255.0
    rgb, alpha = arr[..., :3], arr[..., 3:]
    opaque = alpha[..., 0] > 0.5           # mask of real (non-transparent) pixels
    if not opaque.any():                   # nothing to work with; leave it alone
        return im
    sample = rgb[opaque]                   # (N, 3) of just the visible pixels

    # 1. Gray-world white balance: nudge each channel toward the common mean so
    #    a colour cast is neutralised. Damped so we correct, not overcorrect.
    means = sample.mean(axis=0)
    target = float(means.mean())
    gains = np.where(means > 1e-4, target / means, 1.0)
    gains = 1.0 + 0.6 * (gains - 1.0)      # 60% of the full correction
    gains = np.clip(gains, 0.7, 1.4)
    rgb = rgb * gains

    # 2. Auto levels: stretch luma between robust percentiles (ignores a few
    #    stray dark/bright pixels) so the tonal range fills [0, 1].
    lum = sample @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    lo, hi = np.percentile(lum, [0.5, 99.5])
    if hi - lo > 1e-3:
        rgb = (rgb - lo) / (hi - lo)

    rgb = np.clip(rgb, 0.0, 1.0)
    out = np.concatenate([rgb, alpha], axis=-1)
    im = Image.fromarray((out * 255.0 + 0.5).astype(np.uint8), mode="RGBA")

    # 3. Gentle finishing lift: a little vibrance and sharpening, as the wand does.
    im = ImageEnhance.Color(im).enhance(1.12)
    im = ImageEnhance.Sharpness(im).enhance(1.20)
    return im


def apply_iphone_adjustments(im: Image.Image, args) -> Image.Image:
    """Run the numpy-based colour/tone slider chain, then Pillow's sharpness."""
    active = [(name, fn) for name, fn in ADJUSTMENTS if getattr(args, name) != 0]
    if active:
        arr = np.asarray(im.convert("RGBA"), dtype=np.float32) / 255.0
        rgb, alpha = arr[..., :3], arr[..., 3:]
        for name, fn in active:
            rgb = fn(rgb, getattr(args, name))
        rgb = np.clip(rgb, 0.0, 1.0)
        out = np.concatenate([rgb, alpha], axis=-1)
        im = Image.fromarray((out * 255.0 + 0.5).astype(np.uint8), mode="RGBA")

    # Sharpness is cheap and clean via Pillow (ImageEnhance factor: 1.0 = off).
    if args.sharpness != 0:
        im = ImageEnhance.Sharpness(im).enhance(1.0 + _s(args.sharpness))
    return im


# --------------------------------------------------------------------------- #
# Per-image pipeline
# --------------------------------------------------------------------------- #
def transform_image(im: Image.Image, args, angle: float) -> Image.Image:
    """Apply the full adjust + rotate + downscale pipeline to an in-memory image.

    Returns a new RGBA image. This is the single source of truth shared by the
    CLI (process_image) and the GUI preview, so both always agree.
    """
    im = im.convert("RGBA")

    # 1. Magic-wand auto-optimise first, so any manual sliders layer on top.
    if getattr(args, "magic_wand", False):
        im = magic_wand(im)

    # 2. Legacy Pillow saturation factor (kept for back-compat, 1.0 = off).
    if args.saturation != 1.0:
        im = ImageEnhance.Color(im).enhance(args.saturation)

    # 3. iPhone-style colour/tone sliders + sharpness.
    im = apply_iphone_adjustments(im, args)

    # 4. Rotate (skip when angle is 0). expand=True keeps the whole image;
    #    fillcolor alpha=0 makes exposed corners transparent (no black border).
    if angle % 360 != 0:
        im = im.rotate(angle, resample=Image.BICUBIC, expand=True,
                       fillcolor=(0, 0, 0, 0))

    # 5. Cap the longest side at max_size (aspect ratio preserved; never
    #    upscales; max_size <= 0 disables the cap). Last, so rotation corners
    #    count toward the final dimensions.
    if args.max_size > 0 and max(im.size) > args.max_size:
        im.thumbnail((args.max_size, args.max_size), Image.LANCZOS)

    return im


def process_image(src: Path, dest: Path, args, angle: float) -> None:
    """Adjust + rotate + downscale a single image and write it as WebP."""
    with Image.open(src) as im:
        out = transform_image(im, args, angle)
        dest.parent.mkdir(parents=True, exist_ok=True)
        out.save(dest, format="WEBP", quality=args.quality, method=6)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--in", dest="in_dir", default="in", help="source directory")
    ap.add_argument("--out", dest="out_dir", default="out", help="output directory")

    # One-shot auto-enhance (the "magic wand"). Runs before any manual sliders.
    ap.add_argument("--magic-wand", action="store_true",
                    help="auto-optimise each image (white balance + levels + "
                         "vibrance + sharpen); no rotation unless --rotate is set")

    # iPhone-style sliders (-100..100, 0 = off).
    g = ap.add_argument_group("iPhone-style adjustments (-100..100, 0 = off)")
    for name, help_txt in [
        ("exposure", "overall lightness (multiplicative)"),
        ("brightness", "overall lightness (additive)"),
        ("contrast", "tonal contrast"),
        ("highlights", "recover/boost bright regions"),
        ("shadows", "lift/deepen dark regions"),
        ("black-point", "crush(+)/lift(-) blacks"),
        ("saturation-adj", "plain saturation (iPhone scale)"),
        ("vibrance", "smart saturation (protects vivid/skin)"),
        ("warmth", "colour temperature warm(+)/cool(-)"),
        ("tint", "green(-)/magenta(+) shift"),
        ("sharpness", "edge sharpening"),
        ("vignette", "darken(+)/lighten(-) corners"),
    ]:
        g.add_argument(f"--{name}", type=float, default=0.0, metavar="N", help=help_txt)

    # Legacy / geometry / output.
    ap.add_argument("--saturation", type=float, default=1.05,
                    help="legacy Pillow saturation factor (1.05 = +5%%, default; 1.0 = off)")
    ap.add_argument("--rotate", default="off",
                    help="'off' (default), 'random', or a fixed angle in degrees")
    ap.add_argument("--max-size", type=int, default=512,
                    help="cap the longest side in px (default 512; 0 = no cap)")
    ap.add_argument("--quality", type=int, default=90, help="WebP quality 1-100")
    ap.add_argument("--seed", type=int, default=None,
                    help="random seed for reproducible rotation angles")
    ap.add_argument("--overwrite", action="store_true",
                    help="re-process images whose result already exists (default: skip)")
    return ap


def default_args() -> argparse.Namespace:
    """A Namespace pre-filled with every flag's default value.

    Lets other code (e.g. the GUI) get a valid args object and then override
    only the fields it cares about, without duplicating the flag definitions.
    """
    return build_parser().parse_args([])


def main() -> int:
    args = build_parser().parse_args()

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)

    if not in_dir.is_dir():
        print(f"Source directory not found: {in_dir}", file=sys.stderr)
        return 1

    images = sorted(p for p in in_dir.iterdir()
                    if p.is_file() and p.suffix.lower() in SUPPORTED_EXT)
    if not images:
        print(f"No images found in {in_dir}", file=sys.stderr)
        return 1

    rng = random.Random(args.seed)  # seeded (or system-random if seed is None)

    try:
        resolve_angle(args.rotate, rng)  # validate --rotate once, up front
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return 1

    ok = fail = exists = 0
    for src in images:
        dest = out_dir / (src.stem + ".webp")
        # Guard: don't overwrite a result we already produced (unless --overwrite).
        # Note random rotation still advances the RNG for skipped items so a
        # seeded run stays reproducible regardless of what already exists.
        angle = resolve_angle(args.rotate, rng)
        if dest.exists() and not args.overwrite:
            print(f"  skip {src.name}  (already have {dest.name})")
            exists += 1
            continue
        try:
            process_image(src, dest, args, angle)
            print(f"  ok  {src.name}  ->  {dest.name}   (rotated {angle:6.2f}°)")
            ok += 1
        except Exception as exc:  # noqa: BLE001 - keep going on any single failure
            print(f"  FAIL {src.name}: {exc}", file=sys.stderr)
            fail += 1

    print(f"\nDone. {ok} processed, {exists} already existed, {fail} failed.  "
          f"Output: {out_dir.resolve()}")
    return 0 if fail == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
