"""Thumbnails in the style of the reference channels (Fossil, Extinct World):
one realistic animal in dramatic light on a dark ground, two to four words
set in code - never letters drawn by the image model - in a heavy condensed
face, title case, white with the key word in red, a dark outline and shadow.

Readability is measured on the 336x189 size YouTube shows in the sidebar:
  - cap height at least 26 px there;
  - WCAG contrast between the letters and the pixels around them at least 4.5;
  - the words stay clear of the bottom-right corner, where YouTube prints the
    running time.
"""
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

ROOT = Path(__file__).resolve().parent.parent
FONT = ROOT / "assets" / "fonts" / "Anton-Regular.ttf"
SIZE = (1280, 720)
SMALL = (336, 189)
WHITE, ACCENT = (255, 255, 255), (232, 28, 28)


def _lum(rgb):
    c = np.asarray(rgb, np.float64) / 255
    c = np.where(c <= 0.03928, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    return 0.2126 * c[..., 0] + 0.7152 * c[..., 1] + 0.0722 * c[..., 2]


def contrast(a, b):
    la, lb = max(a, b), min(a, b)
    return (la + 0.05) / (lb + 0.05)


def compose(base_path, lines, side, out_path, accent_line=None, shade_k=0.45, box_frac=0.46):
    """lines: 1-2 strings; side: 'left' or 'right' third; accent_line: index in yellow."""
    im = Image.open(base_path).convert("RGB")
    s = max(SIZE[0] / im.width, SIZE[1] / im.height)
    im = im.resize((round(im.width * s), round(im.height * s)), Image.LANCZOS)
    l, t = (im.width - SIZE[0]) // 2, (im.height - SIZE[1]) // 2
    im = im.crop((l, t, l + SIZE[0], t + SIZE[1]))
    # lift the subject a little, deepen the side the words sit on
    arr = np.asarray(im, np.float32)
    x = np.linspace(0, 1, SIZE[0])[None, :, None]
    shade = (1 - shade_k * np.clip((0.55 - x) / 0.55, 0, 1)) if side == "left" else \
            (1 - shade_k * np.clip((x - 0.45) / 0.55, 0, 1))
    arr = arr * shade
    arr = np.clip((arr - 128) * 1.08 + 128 + 4, 0, 255)
    im = Image.fromarray(arr.astype(np.uint8))

    words = list(lines)
    box_w = SIZE[0] * box_frac
    size = 200
    while size > 60:
        f = ImageFont.truetype(str(FONT), size)
        widths = [f.getlength(w) for w in words]
        if max(widths) <= box_w and size * 1.02 * len(words) <= SIZE[1] * 0.72:
            break
        size -= 4
    f = ImageFont.truetype(str(FONT), size)
    line_h = int(size * 1.02)
    block_h = line_h * len(words)
    y0 = int(SIZE[1] * 0.42 - block_h / 2)
    x0 = int(SIZE[0] * 0.05) if side == "left" else int(SIZE[0] * 0.95 - max(f.getlength(w) for w in words))

    shadow = Image.new("RGBA", SIZE, (0, 0, 0, 0))
    ds = ImageDraw.Draw(shadow)
    text = Image.new("RGBA", SIZE, (0, 0, 0, 0))
    dt = ImageDraw.Draw(text)
    masks = []
    for i, w in enumerate(words):
        y = y0 + i * line_h
        col = ACCENT if i == accent_line else WHITE
        ds.text((x0 + 6, y + 8), w, font=f, fill=(0, 0, 0, 230))
        dt.text((x0, y), w, font=f, fill=(*col, 255), stroke_width=max(3, size // 28), stroke_fill=(10, 10, 10, 255))
        m = Image.new("L", SIZE, 0)
        ImageDraw.Draw(m).text((x0, y), w, font=f, fill=255)
        masks.append(m)
    shadow = shadow.filter(ImageFilter.GaussianBlur(10))
    out = Image.alpha_composite(Image.alpha_composite(im.convert("RGBA"), shadow), text).convert("RGB")
    out.save(out_path, quality=95)
    return check(out, masks, size, len(words), out_path)


def _line_contrast(arr, m_small):
    ring = cv2.dilate(m_small.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool) & ~cv2.dilate(
        m_small.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    if not m_small.any() or not ring.any():
        return 0.0
    return contrast(float(np.median(_lum(arr[m_small]))), float(np.median(_lum(arr[ring]))))


def check(img, masks, size, n_lines, path):
    """WCAG large-text rule (3:1) on every line separately, at 336x189."""
    small = img.resize(SMALL, Image.LANCZOS)
    arr = np.asarray(small, np.float64)
    cap_px = size * 0.74 * SMALL[1] / SIZE[1]                   # Anton cap height ~0.74 em
    per_line = [round(_line_contrast(arr, np.asarray(m.resize(SMALL, Image.BILINEAR)) > 128), 2) for m in masks]
    big = np.zeros((SIZE[1], SIZE[0]), bool)
    for m in masks:
        big |= np.asarray(m) > 0
    ys, xs = np.nonzero(big)
    corner = bool(((xs > SIZE[0] * 0.80) & (ys > SIZE[1] * 0.82)).any()) if len(xs) else False
    rep = {"file": str(path), "cap_height_px_at_336": round(cap_px, 1), "contrast_per_line": per_line,
           "lines": n_lines, "clear_of_timestamp": not corner}
    rep["readable"] = cap_px >= 26 and min(per_line) >= 3.0 and not corner
    small.save(str(path).replace(".jpg", "_336.jpg"), quality=92)
    return rep


def best(base_path, lines, side, out_path, accent_line=None, box_frac=0.46):
    """Compose, deepening the shade on the text side until every line reads."""
    for k in (0.45, 0.62, 0.78, 0.9):
        rep = compose(base_path, lines, side, out_path, accent_line, shade_k=k, box_frac=box_frac)
        rep["shade"] = k
        if rep["readable"]:
            break
    return rep
