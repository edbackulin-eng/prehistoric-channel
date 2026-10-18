"""Thumbnails in the style of the reference channels: one realistic animal in
dramatic light, two to four words set in code - never letters drawn by the
image model - in a heavy condensed face.

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
WHITE, ACCENT = (255, 255, 255), (255, 201, 74)


def _lum(rgb):
    c = np.asarray(rgb, np.float64) / 255
    c = np.where(c <= 0.03928, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    return 0.2126 * c[..., 0] + 0.7152 * c[..., 1] + 0.0722 * c[..., 2]


def contrast(a, b):
    la, lb = max(a, b), min(a, b)
    return (la + 0.05) / (lb + 0.05)


def compose(base_path, lines, side, out_path, accent_line=None):
    """lines: 1-2 strings; side: 'left' or 'right' third; accent_line: index in yellow."""
    im = Image.open(base_path).convert("RGB")
    s = max(SIZE[0] / im.width, SIZE[1] / im.height)
    im = im.resize((round(im.width * s), round(im.height * s)), Image.LANCZOS)
    l, t = (im.width - SIZE[0]) // 2, (im.height - SIZE[1]) // 2
    im = im.crop((l, t, l + SIZE[0], t + SIZE[1]))
    # lift the subject a little, deepen the side the words sit on
    arr = np.asarray(im, np.float32)
    x = np.linspace(0, 1, SIZE[0])[None, :, None]
    shade = (1 - 0.45 * np.clip((0.55 - x) / 0.55, 0, 1)) if side == "left" else \
            (1 - 0.45 * np.clip((x - 0.45) / 0.55, 0, 1))
    arr = arr * shade
    arr = np.clip((arr - 128) * 1.08 + 128 + 4, 0, 255)
    im = Image.fromarray(arr.astype(np.uint8))

    words = [w.upper() for w in lines]
    box_w = SIZE[0] * 0.46
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
    mask = Image.new("L", SIZE, 0)
    dm = ImageDraw.Draw(mask)
    for i, w in enumerate(words):
        y = y0 + i * line_h
        col = ACCENT if i == accent_line else WHITE
        ds.text((x0 + 6, y + 8), w, font=f, fill=(0, 0, 0, 230))
        dt.text((x0, y), w, font=f, fill=(*col, 255), stroke_width=max(3, size // 28), stroke_fill=(10, 10, 10, 255))
        dm.text((x0, y), w, font=f, fill=255)
    shadow = shadow.filter(ImageFilter.GaussianBlur(10))
    out = Image.alpha_composite(Image.alpha_composite(im.convert("RGBA"), shadow), text).convert("RGB")
    out.save(out_path, quality=95)
    return check(out, mask, size, len(words), out_path)


def check(img, mask, size, n_lines, path):
    small = img.resize(SMALL, Image.LANCZOS)
    m_small = np.asarray(mask.resize(SMALL, Image.BILINEAR)) > 128
    cap_px = size * 0.74 * SMALL[1] / SIZE[1]                   # Anton cap height ~0.74 em
    arr = np.asarray(small, np.float64)
    ring = cv2.dilate(m_small.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool) & ~cv2.dilate(
        m_small.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    l_text = float(np.median(_lum(arr[m_small]))) if m_small.any() else 0
    l_bg = float(np.median(_lum(arr[ring]))) if ring.any() else 0
    big = np.asarray(mask) > 0
    ys, xs = np.nonzero(big)
    corner = bool(((xs > SIZE[0] * 0.80) & (ys > SIZE[1] * 0.82)).any()) if len(xs) else False
    rep = {"file": str(path), "cap_height_px_at_336": round(cap_px, 1), "contrast": round(contrast(l_text, l_bg), 2),
           "lines": n_lines, "clear_of_timestamp": not corner}
    rep["readable"] = rep["cap_height_px_at_336"] >= 26 and rep["contrast"] >= 4.5 and not corner
    small.save(str(path).replace(".jpg", "_336.jpg"), quality=92)
    return rep
