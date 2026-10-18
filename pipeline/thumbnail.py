"""Closed Tabs thumbnails: someone at 3 a.m. pointing at the impossible thing.

The recipe, every episode:
  - FLUX draws a shocked person, 20-35, in half profile on the left, pointing at
    a monitor on the right that faces the camera squarely - a flat, frontal
    screen, so what is painted into it sits in it exactly. The person changes
    every time (seeded from the episode). The monitor is generated blank.
  - Gemini gives the rough corners of that blank screen; code then finds the lit
    blank panel itself in the pixels (refine_quad) and paints the episode's
    screen into exactly that shape. A candidate whose screen cannot be found is
    ranked below every one whose screen was.
  - On the screen: the heading and one whole short sentence, never a cut-off
    one, and beneath them an image tile with a single glowing detail. The red
    circle goes around that detail, inside the tile, so it never covers words.
  - Two or three words in Anton, white with one word red, heavy black outline -
    words that add to the title rather than repeat it.
  - A light VHS layer, a REC dot and a 03:14 AM stamp.

Six to eight candidates are drawn. Gemini rejects hands without five fingers,
broken faces and eyes on the lens; the words must still read at 336x189, the size
a thumbnail is actually seen at in search results; the best survivor wins.
"""
import json
import random
import re
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from core import fonts
from core.logger import get_logger

log = get_logger("thumbnail")

W, H = 1280, 720
SMALL = (336, 189)
CANDIDATES = 8

PEOPLE = [
    "a young woman with short dark hair in a grey hoodie",
    "a young man with curly hair and round glasses in a black t-shirt",
    "a woman in her late twenties with long red hair in a denim jacket",
    "a man around thirty with a short beard in a flannel shirt",
    "a young woman with braided hair in an oversized sweater",
    "a young man with a buzz cut in a dark green bomber jacket",
    "a woman in her thirties with a messy bun and headphones around her neck",
    "a young man with long straight hair tied back, in a white hoodie",
    "a woman in her early twenties with a fringe and a striped shirt",
    "a man in his late twenties with wavy hair and a leather jacket",
]


def base_prompts(seed, n=CANDIDATES):
    rng = random.Random(seed)
    who = rng.sample(PEOPLE, n)
    return [
        f"{p}, shocked, eyes wide and mouth open, seen in profile on the left third of the "
        f"frame, pointing a finger at a large flat computer monitor on the right half of the "
        f"frame, the monitor faces the camera squarely, shot straight on, its screen a flat "
        f"rectangle parallel to the image, the screen is blank, evenly lit pale grey-white, "
        f"completely empty, the person looks at the monitor and not at the camera, a bedroom "
        f"at night lit by teal and magenta neon light and the glow of the screen, well "
        f"exposed, colourful, cinematic, photorealistic, 35mm film still, "
        f"no text, no letters, no logos, five fingers on each hand"
        for p in who]


# --- the screen content ----------------------------------------------------------

def screen_content(heading, line, size=(1200, 900)):
    """What is on the monitor: a page with a heading, one whole sentence, and an
    image tile whose single glowing detail is circled. Text and circle never meet:
    the circle lives inside the tile, below every word."""
    w, h = size
    im = Image.new("RGB", size, (20, 24, 38))
    d = ImageDraw.Draw(im)
    d.rectangle((0, 0, w, int(h * 0.08)), fill=(36, 42, 62))
    bar = int(h * 0.08)
    for i, c in enumerate(((235, 90, 80), (240, 190, 70), (90, 200, 110))):
        r = bar * 0.16
        cx0 = bar * 0.5 + i * bar * 0.45
        d.ellipse((cx0 - r, bar / 2 - r, cx0 + r, bar / 2 + r), fill=c)
    pad = int(w * 0.06)
    y = bar + int(h * 0.05)
    hs, hl = fonts.fit(heading, "sans_bold", w - 2 * pad, h * 0.20, int(h * 0.13))
    for t in hl:
        d.text((pad, y), t, font=fonts.font("sans_bold", hs), fill=(232, 238, 244))
        y += hs * 1.15
    y += h * 0.02
    ls, ll = fonts.fit(line, "sans", w - 2 * pad, h * 0.19, int(h * 0.08))
    for t in ll:
        d.text((pad, y), t, font=fonts.font("sans", ls), fill=(150, 220, 210))
        y += ls * 1.2
    y += h * 0.04
    tile = (pad, int(y), w - pad, h - pad)
    d.rectangle(tile, fill=(10, 12, 22), outline=(60, 70, 100), width=3)
    # the detail: a small glowing figure off-centre in the tile, as if hidden in it
    tw, th = tile[2] - tile[0], tile[3] - tile[1]
    cx, cy = tile[0] + tw * 0.66, tile[1] + th * 0.55
    rr = min(tw, th) * 0.22
    glow = Image.new("RGB", size, (0, 0, 0))
    gd = ImageDraw.Draw(glow)
    gd.ellipse((cx - rr * 0.5, cy - rr * 0.5, cx + rr * 0.5, cy + rr * 0.5), fill=(120, 255, 220))
    glow = glow.filter(ImageFilter.GaussianBlur(rr * 0.35))
    im = Image.fromarray(np.clip(np.asarray(im, np.int16) + np.asarray(glow, np.int16), 0, 255).astype(np.uint8))
    d = ImageDraw.Draw(im)
    d.ellipse((cx - rr * 0.16, cy - rr * 0.16, cx + rr * 0.16, cy + rr * 0.16), fill=(230, 255, 248))
    # the circle stays inside the tile
    R_ = min(rr * 1.25, (tile[3] - cy) - 10, (cy - tile[1]) - 10)
    d.ellipse((cx - R_ * 1.25, cy - R_, cx + R_ * 1.25, cy + R_), outline=(232, 44, 52),
              width=max(8, int(h * 0.016)))
    return im


def screen_line(script, most_chars=60):
    """One whole short sentence from the script for the screen: never a cut-off one."""
    import re as _re
    text = " ".join(l.lstrip("<> ").strip() for l in script.splitlines()
                    if l.strip() and not l.strip().startswith("["))
    sents = [x.strip() for x in _re.split(r"(?<=[.!?])\s+", text)]
    good = [x for x in sents if 18 <= len(x) <= most_chars and x[:1].isupper() and x[-1] in ".!?"]
    return good[0] if good else ""


def refine_quad(base, quad):
    """The blank lit panel itself, found in the pixels near Gemini's rough corners.

    The screen was generated as an evenly lit pale panel: bright, nearly grey and
    smooth. The largest such region overlapping the rough quad is the screen; its
    four extreme points are the corners. Returns a quad in 0-1 or None."""
    from scipy import ndimage
    a = np.asarray(base.convert("RGB"), np.float32)
    v = a.max(axis=2)
    sat = (v - a.min(axis=2)) / (v + 1e-3)
    grey = a.mean(axis=2)
    smooth = ndimage.uniform_filter((grey - ndimage.uniform_filter(grey, 9)) ** 2, 9) ** 0.5
    mask = (v > np.percentile(v, 80)) & (v > 120) & (sat < 0.35) & (smooth < 6)
    mask = ndimage.binary_opening(mask, iterations=2)
    mask = ndimage.binary_closing(mask, iterations=3)
    lab, n = ndimage.label(mask)
    if not n:
        return None
    h, w = mask.shape
    if quad and valid_quad(quad):
        q = np.array(quad, float) * [w, h]
        x0, y0 = q.min(axis=0)
        x1, y1 = q.max(axis=0)
        rough_area = (x1 - x0) * (y1 - y0)
    else:
        x0, y0, x1, y1 = w * 0.45, 0, w, h * 0.9          # the right half, where it is asked for
        rough_area = None
    best, best_n = None, 0
    for k in range(1, n + 1):
        ys, xs = np.nonzero(lab[int(y0):int(y1) + 1, int(x0):int(x1) + 1] == k)
        if len(xs) > best_n:
            best, best_n = k, len(xs)
    if not best:
        return None
    ys, xs = np.nonzero(lab == best)
    area = len(xs)
    if area < 0.04 * w * h:
        return None
    if rough_area and not 0.45 * rough_area <= area <= 1.6 * rough_area:
        return None
    # A screen is a convex quadrilateral that its pixels nearly fill.
    tl = (xs + ys).argmin(); br = (xs + ys).argmax()
    tr = (xs - ys).argmax(); bl = (xs - ys).argmin()
    pts = np.array([[xs[tl], ys[tl]], [xs[tr], ys[tr]], [xs[br], ys[br]], [xs[bl], ys[bl]]], float)
    quad_area = 0.5 * abs(np.dot(pts[:, 0], np.roll(pts[:, 1], 1)) - np.dot(pts[:, 1], np.roll(pts[:, 0], 1)))
    if quad_area <= 0 or area / quad_area < 0.85:
        return None
    # pull in by 1% toward the centre so the paint stays inside the bezel
    c = pts.mean(axis=0)
    pts = c + (pts - c) * 0.995
    out = [[round(float(x / w), 4), round(float(y / h), 4)] for x, y in pts]
    return out if valid_quad(out) else None


def _perspective_coeffs(src, dst):
    """Coefficients for Image.transform(PERSPECTIVE) mapping dst quad onto src quad."""
    a = []
    for (x, y), (u, v) in zip(dst, src):
        a.append([x, y, 1, 0, 0, 0, -u * x, -u * y])
        a.append([0, 0, 0, x, y, 1, -v * x, -v * y])
    b = np.array(src, np.float64).reshape(8)
    return np.linalg.solve(np.array(a, np.float64), b).tolist()


def paste_screen(base, quad, content):
    """Warp `content` into the quad (four corners, clockwise from top left, 0-1)."""
    dst = [(x * W, y * H) for x, y in quad]
    cw, ch = content.size
    src = [(0, 0), (cw, 0), (cw, ch), (0, ch)]
    coeffs = _perspective_coeffs(src, dst)
    warped = content.transform((W, H), Image.PERSPECTIVE, coeffs, Image.BICUBIC)
    mask = Image.new("L", (W, H), 0)
    ImageDraw.Draw(mask).polygon(dst, fill=255)
    mask = mask.filter(ImageFilter.GaussianBlur(1.2))
    out = base.copy()
    out.paste(warped, (0, 0), mask)
    # the screen glows: a soft bloom of its own light around the quad
    glow = Image.new("RGB", (W, H), (0, 0, 0))
    glow.paste(warped, (0, 0), mask)
    glow = glow.filter(ImageFilter.GaussianBlur(26))
    return Image.blend(out, Image.fromarray(np.clip(np.asarray(out, np.int16) +
                                                    np.asarray(glow, np.int16) // 3, 0, 255)
                                            .astype(np.uint8)), 0.8)


def valid_quad(q):
    if not (isinstance(q, list) and len(q) == 4):
        return False
    try:
        pts = np.array(q, float)
    except (TypeError, ValueError):
        return False
    if pts.shape != (4, 2) or (pts < 0).any() or (pts > 1).any():
        return False
    # convex and big enough to read
    cross = []
    for i in range(4):
        a, b, c = pts[i], pts[(i + 1) % 4], pts[(i + 2) % 4]
        cross.append(np.cross(b - a, c - b))
    area = 0.5 * abs(np.dot(pts[:, 0], np.roll(pts[:, 1], 1)) - np.dot(pts[:, 1], np.roll(pts[:, 0], 1)))
    return (all(c > 0 for c in cross) or all(c < 0 for c in cross)) and area > 0.04


# --- words, VHS, checks ---------------------------------------------------------

def _words(text):
    words = [w for w in re.findall(r"[\w'?!.-]+", text.upper())][:3]
    return words or ["CLOSED", "TABS"]


def draw_words(im, text, face=None):
    """Anton, white with the last word red, black outline, on the left half - above
    or below the face, whichever leaves it clear. Returns its box and size."""
    words = _words(text)
    d = ImageDraw.Draw(im)
    box_w, box_h = W * 0.56, H * 0.40
    phrase = " ".join(words)
    size, lines = fonts.fit(phrase, "title", box_w, box_h, 190, min_size=60, spacing=1.0)
    f = fonts.font("title", size)
    block_h = size * len(lines)
    x0, y0 = 48, 40
    if face and isinstance(face, list) and len(face) == 4:
        fy0, fy1 = face[1] * H, face[3] * H
        if fy0 < 40 + block_h + 20:                 # the top would cover the face
            y0 = int(H - block_h - 50) if fy1 < H - block_h - 70 else 40
    stroke = max(6, size // 14)
    red_word = words[-1]
    y = y0
    for line in lines:
        x = x0
        for word in line.split():
            col = (232, 44, 52) if word == red_word else (255, 255, 255)
            d.text((x, y), word, font=f, fill=col, stroke_width=stroke, stroke_fill=(0, 0, 0))
            x += f.getlength(word + " ")
        y += size * 1.0
    return (x0, y0, x0 + box_w, y), size


def vhs(im, seed=0):
    rng = np.random.default_rng(seed)
    a = np.asarray(im, np.int16).copy()
    a[:, 3:, 0] = a[:, :-3, 0]                         # red channel slip
    a[:, :-3, 2] = a[:, 3:, 2]                         # blue channel slip
    a += rng.integers(-9, 10, a.shape[:2], dtype=np.int16)[..., None]
    a[::3] = (a[::3] * 0.88).astype(np.int16)          # scanlines
    out = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))
    d = ImageDraw.Draw(out)
    f = fonts.font("label", 54)
    d.ellipse((W - 250, 44, W - 222, 72), fill=(232, 30, 30))
    d.text((W - 208, 34), "REC", font=f, fill=(255, 255, 255), stroke_width=2, stroke_fill=(0, 0, 0))
    stamp = "03:14 AM"
    d.text((W - 48 - f.getlength(stamp), H - 92), stamp, font=f, fill=(255, 255, 255),
           stroke_width=2, stroke_fill=(0, 0, 0))
    return out


def readable_small(im, text_size):
    """Do the words survive at 336x189? Cap height >= 16 px and real contrast."""
    scale = SMALL[1] / H
    cap_px = text_size * 0.72 * scale
    small = im.resize(SMALL, Image.LANCZOS).convert("L")
    region = np.asarray(small)[:int(SMALL[1] * 0.45), :int(SMALL[0] * 0.58)].astype(float)
    spread = np.percentile(region, 95) - np.percentile(region, 5)
    return bool(cap_px >= 16 and spread >= 120), {"cap_px": round(float(cap_px), 1), "spread": round(float(spread), 1)}


def compose(base_path, quad, words, heading, line, seed, face=None):
    """Returns (image, text size, whether the screen was found in the pixels)."""
    from pipeline import grade
    base = Image.open(base_path).convert("RGB")
    s_ = max(W / base.width, H / base.height)
    base = base.resize((round(base.width * s_), round(base.height * s_)), Image.LANCZOS)
    x, y = (base.width - W) // 2, (base.height - H) // 2
    base = base.crop((x, y, x + W, y + H))
    # The rough quad from Gemini is in the original frame; the crop above is a
    # centred fill, so map it before refining.
    if quad and valid_quad(quad):
        bw, bh = base.width + 2 * x, base.height + 2 * y
        quad = [[(qx * bw - x) / W, (qy * bh - y) / H] for qx, qy in quad]
    exact = refine_quad(base, quad)
    base = grade.grade(base)
    if exact:
        q = np.array(exact) * [W, H]
        qw = np.linalg.norm(q[1] - q[0])
        qh = np.linalg.norm(q[3] - q[0])
        cw = 1200
        content = screen_content(heading, line, (cw, max(400, int(cw * qh / max(qw, 1)))))
        base = paste_screen(base, exact, content)
    else:
        panel = screen_content(heading, line, (720, 540)).resize((560, 420), Image.LANCZOS)
        base.paste(panel, (W - 600, 220))
    _, size = draw_words(base, words, face)
    return vhs(base, seed), size, bool(exact)


def choose(bases, verdicts, words, heading, line, out_dir, seed):
    """Compose every approved base, keep those that read small, return the best.

    Returns (path of the winner, report). Falls back to the best-scored base when
    none passes everything, and says so - a thumbnail is never left out."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report, ranked = [], []
    for k, b in enumerate(bases):
        v = verdicts.get(Path(b), {})
        im, size, exact = compose(b, v.get("screen"), words, heading, line, seed + k, v.get("face"))
        path = out_dir / f"thumb_{k}.jpg"
        im.save(path, quality=92)
        readable, m = readable_small(im, size)
        row = {"file": path.name, "ok": v.get("ok", False), "defect": v.get("defect"),
               "readable": readable, "screen_found": exact, **m, "score": v.get("score", 0)}
        report.append(row)
        if row["ok"] and readable:
            ranked.append((exact, row["score"], path))
    if ranked:
        best = max(ranked, key=lambda r: (r[0], r[1]))[2]
    else:
        best = out_dir / max(report, key=lambda r: (r["screen_found"], r["readable"], r["score"]))["file"]
        log.warning("no candidate passed every check; using %s", best.name)
    final = out_dir / "thumbnail.jpg"
    Image.open(best).save(final, quality=92, optimize=True)
    if final.stat().st_size > 2_000_000:
        Image.open(best).save(final, quality=80, optimize=True)
    return final, report
