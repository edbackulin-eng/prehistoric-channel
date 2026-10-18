"""The channel's single look, applied to every frame before it reaches ffmpeg.

Archive space photography arrives in wildly different grades: Apollo film scans are
warm and faded, modern orbital imagery is clinical and flat. Putting them in one
video without a common grade is the fastest way to look like a slideshow. Doing it
here in Pillow rather than in an ffmpeg filter also costs nothing at render time.
"""
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

# Render at 1.4x the output so a Ken Burns push never runs out of pixels.
WORK_W, WORK_H = 2688, 1512
OUT_W, OUT_H = 1920, 1080

# Cold shadows, warm highlights: the standard cinematic split, kept restrained.
SHADOW_TINT = (0.96, 1.00, 1.09)
HIGHLIGHT_TINT = (1.05, 1.005, 0.95)
CONTRAST = 1.12
SATURATION = 1.18
BLACK_LIFT = 5 / 255          # a lifted black point reads as film, not video


def cover_crop(im, width=WORK_W, height=WORK_H):
    """Fill the frame, centred, without distorting the aspect ratio."""
    scale = max(width / im.width, height / im.height)
    im = im.resize((max(width, round(im.width * scale)),
                    max(height, round(im.height * scale))), Image.LANCZOS)
    left, top = (im.width - width) // 2, (im.height - height) // 2
    return im.crop((left, top, left + width, top + height))


def _load(path, width, height):
    """Open a still, letting the JPEG decoder do the first halving itself.

    draft() picks a DCT-domain scale factor during decode, so an 8000x8000
    archive scan never becomes a full-size Python object. It only applies to
    JPEGs and only in powers of two, so cover_crop still does the exact fit.
    """
    im = Image.open(path)
    try:
        im.draft("RGB", (width, height))
    except (AttributeError, ValueError):
        pass
    return im.convert("RGB")


def _split_tone(arr):
    """Tint shadows and highlights in opposite directions, weighted by luminance."""
    luma = arr @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
    w = luma[..., None]
    shadow = np.array(SHADOW_TINT, dtype=np.float32)
    highlight = np.array(HIGHLIGHT_TINT, dtype=np.float32)
    return arr * (shadow * (1 - w) + highlight * w)


def fit_with_blur_fill(im, width=WORK_W, height=WORK_H):
    """Fit a small still inside the frame and fill the margins with a blurred copy.

    Much of the best archive material is small - NASA's own "high-resolution"
    Earthrise release is 952x952 - and blowing it up to fill 1512 lines turns it
    to mush. Fitting it at a sane scale over a blurred enlargement of itself keeps
    the subject crisp and still fills a 16:9 frame, which is what documentary
    editors do with archive footage for exactly this reason.
    """
    backdrop = cover_crop(im, width, height).filter(ImageFilter.GaussianBlur(42))
    backdrop = ImageEnhance.Brightness(backdrop).enhance(0.55)
    # Allow a mild upscale (never more than 1.5x) so small stills still read big.
    scale = min(width / im.width, height / im.height, 1.5)
    w, h = max(1, round(im.width * scale)), max(1, round(im.height * scale))
    front = im.resize((w, h), Image.LANCZOS)
    backdrop.paste(front, ((width - w) // 2, (height - h) // 2))
    return backdrop


# How far a source may stray from 16:9 and still simply be filled to the frame.
ASPECT_TOLERANCE = 0.02
# Beyond this upscale a non-16:9 still looks soft, so it gets the blurred plate
# instead of being stretched up to fill.
MAX_UPSCALE = 1.6


def _needs_blur_plate(im, width, height):
    """Only an oddly-shaped still that is also too small gets a blurred plate.

    Deciding this on size alone was wrong: FLUX frames arrive at 1360x768, which
    is already 16:9 but smaller than the working canvas, so every generated shot
    - two thirds of the video - was being inset on a blurred plate and showed a
    soft dark border at the edges of frame. Anything already 16:9 is simply
    scaled up to fill, whatever its pixel size.
    """
    aspect = im.width / im.height
    target = width / height
    if abs(aspect - target) <= ASPECT_TOLERANCE:
        return False
    upscale = max(width / im.width, height / im.height)
    return upscale > MAX_UPSCALE


# Frames that were inset on a blurred plate rather than filling the frame.
# Recorded here because this is the only place that knows; trying to detect it
# afterwards from the pixels flagged flat skies and infographic cards as faults.
plated = set()


def apply(path, width=WORK_W, height=WORK_H):
    """Load, crop to the working size, and grade. Returns a PIL image."""
    raw = _load(path, width, height)
    if _needs_blur_plate(raw, width, height):
        plated.add(str(path))
        im = fit_with_blur_fill(raw, width, height)
    else:
        im = cover_crop(raw, width, height)
    im = ImageEnhance.Contrast(im).enhance(CONTRAST)
    im = ImageEnhance.Color(im).enhance(SATURATION)
    arr = np.asarray(im, dtype=np.float32) / 255.0
    arr = _split_tone(arr)
    arr = BLACK_LIFT + arr * (1.0 - BLACK_LIFT)
    return Image.fromarray(np.clip(arr * 255.0, 0, 255).astype(np.uint8))
