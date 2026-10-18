"""Procedurally drawn cards: timers, numbers and diagrams.

These carry 15-25% of a video (measured in docs/kt3-source-budget.md), so they are
not decoration - they are a third image source alongside the archive and FLUX.
Drawing them rather than generating them means the numbers are always right, the
look never drifts, and they cost no GPU time.

Everything is rendered at the same working size as a graded photo so the Ken Burns
stage treats a card exactly like any other frame.
"""
import math
import re
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont

from core.grade import WORK_H, WORK_W
from core.logger import get_logger

log = get_logger("infographics")

FONTS = Path(__file__).resolve().parent.parent / "assets" / "fonts"

BG = (9, 12, 20)
INK = (236, 240, 248)
ACCENT = (255, 184, 28)
DIM = (120, 132, 154)
GRID = (26, 32, 46)


def _font(name, size):
    return ImageFont.truetype(str(FONTS / name), size)


def _canvas(vignette=True):
    im = Image.new("RGB", (WORK_W, WORK_H), BG)
    if vignette:
        # A faint radial lift stops a flat fill reading as a dead frame on video.
        glow = Image.new("L", (WORK_W, WORK_H), 0)
        ImageDraw.Draw(glow).ellipse(
            (WORK_W * 0.1, -WORK_H * 0.35, WORK_W * 0.9, WORK_H * 1.2), fill=42)
        glow = glow.filter(ImageFilter.GaussianBlur(220))
        im = Image.composite(Image.new("RGB", im.size, (26, 33, 48)), im, glow)
    return im


def _grid(im, step=112):
    """A faint technical grid: reads as instrumentation, not as empty space."""
    d = ImageDraw.Draw(im)
    for x in range(0, WORK_W, step):
        d.line([(x, 0), (x, WORK_H)], fill=GRID, width=2)
    for y in range(0, WORK_H, step):
        d.line([(0, y), (WORK_W, y)], fill=GRID, width=2)
    return im


# Cards are pushed slightly during the shot, so type has to sit inside a safe
# box rather than running to the edge of the canvas.
SAFE_WIDTH = 0.76


def _fit(draw, text, font_name, start_size, max_fraction=SAFE_WIDTH):
    """Largest size at which `text` fits inside the safe width."""
    size = start_size
    limit = WORK_W * max_fraction
    while size > 60:
        font = _font(font_name, size)
        if draw.textlength(text, font=font) <= limit:
            return font
        size -= 15
    return _font(font_name, 60)


def timer_card(label, sublabel="", progress=1.0):
    """The event clock: "T+8 MIN", "DAY 19"."""
    im = _grid(_canvas())
    d = ImageDraw.Draw(im)
    text = label.upper()
    f = _fit(d, text, "Anton-Regular.ttf", 430)
    d.text((WORK_W / 2, WORK_H / 2 - 40), text, font=f,
           fill=_mix(INK, progress), anchor="mm", stroke_width=0)
    # An accent rule under the clock, the same device as the thumbnails. It
    # wipes out from the centre as the reveal runs.
    w = d.textlength(text, font=f) * min(1.0, progress * 1.2)
    y = WORK_H / 2 + 190
    if w > 1:
        d.rectangle([(WORK_W - w) / 2, y, (WORK_W + w) / 2, y + 14], fill=ACCENT)
    if sublabel:
        d.text((WORK_W / 2, y + 90), sublabel.upper(),
               font=_font("BebasNeue-Regular.ttf", 96),
               fill=_mix(DIM, progress), anchor="ma")
    return im


_NUMBER_IN_TEXT = re.compile(r"(-?[\d][\d,]*\.?\d*)")


def _count_up(value, progress):
    """Ramp the first number in `value` from zero to its full figure.

    A card that simply appears is static; a figure that climbs reads as data
    arriving, which is what these cards are for. Non-numeric headlines are
    returned unchanged and just fade in.
    """
    text = str(value)
    match = _NUMBER_IN_TEXT.search(text)
    if not match or progress >= 1.0:
        return text
    # A formula is not a quantity: ramping the 1 in "Inverse Cube (1/r^3)" turns
    # it into "(0/r^3)", which reads as a rendering fault.
    if any(ch in text for ch in "^/=*<>"):
        return text
    raw = match.group(1)
    try:
        target = float(raw.replace(",", ""))
    except ValueError:
        return text
    # Below this there are no meaningful intermediate steps - "1 Million Years"
    # would just sit on "0" until the final frame.
    if abs(target) < 2:
        return text
    current = target * progress
    decimals = len(raw.split(".")[1]) if "." in raw else 0
    if "," in raw:
        shown = f"{current:,.{decimals}f}"
    else:
        shown = f"{current:.{decimals}f}"
    return text[:match.start()] + shown + text[match.end():]


def _mix(colour, progress):
    """Fade a colour up from the background as the reveal runs."""
    return tuple(round(BG[i] + (colour[i] - BG[i]) * progress) for i in range(3))


def number_card(value, caption, footnote="", progress=1.0):
    """One enormous figure with its unit: "8x", "16 m", "3 billion"."""
    im = _grid(_canvas())
    d = ImageDraw.Draw(im)
    shown = _count_up(value, progress)
    # Size the type to the FINAL string so the card does not jiggle while counting.
    f = _fit(d, str(value), "Anton-Regular.ttf", 500)
    d.text((WORK_W / 2, WORK_H / 2 - 90), shown, font=f,
           fill=_mix(ACCENT, progress), anchor="mm")
    d.text((WORK_W / 2, WORK_H / 2 + 230), caption.upper(),
           font=_font("BebasNeue-Regular.ttf", 128),
           fill=_mix(INK, progress), anchor="ma")
    if footnote:
        d.text((WORK_W / 2, WORK_H - 150), footnote,
               font=_font("BebasNeue-Regular.ttf", 72),
               fill=_mix(DIM, progress), anchor="ma")
    return im


def distance_diagram(fraction, label, roche_fraction=None):
    """Earth, the Moon's current position along its track, and the Roche line.

    `fraction` is how far along the original orbit the Moon still is, 1.0 being
    its real distance and 0 being Earth's centre.
    """
    im = _canvas(vignette=False)
    d = ImageDraw.Draw(im)
    cx, cy = 300, WORK_H / 2
    track_end = WORK_W - 220

    d.line([(cx, cy), (track_end, cy)], fill=GRID, width=6)
    # Earth
    d.ellipse([cx - 110, cy - 110, cx + 110, cy + 110], fill=(44, 92, 150))
    d.ellipse([cx - 110, cy - 110, cx + 110, cy + 110], outline=(96, 150, 214), width=5)
    d.text((cx, cy + 165), "EARTH", font=_font("BebasNeue-Regular.ttf", 76),
           fill=DIM, anchor="ma")

    if roche_fraction is not None:
        rx = cx + (track_end - cx) * roche_fraction
        for y in range(int(cy - 300), int(cy + 300), 36):
            d.line([(rx, y), (rx, y + 20)], fill=ACCENT, width=5)
        d.text((rx, cy - 360), "ROCHE LIMIT", font=_font("BebasNeue-Regular.ttf", 68),
               fill=ACCENT, anchor="ma")

    mx = cx + (track_end - cx) * fraction
    r = 54
    d.ellipse([mx - r, cy - r, mx + r, cy + r], fill=(186, 190, 198))
    d.text((mx, cy + 120), "MOON", font=_font("BebasNeue-Regular.ttf", 68),
           fill=DIM, anchor="ma")

    d.text((WORK_W / 2, 120), label.upper(), font=_font("Anton-Regular.ttf", 132),
           fill=INK, anchor="ma")
    return im


def tide_diagram(multiplier, label):
    """Earth with its two tidal bulges, exaggerated by `multiplier`."""
    im = _canvas()
    d = ImageDraw.Draw(im)
    cx, cy = WORK_W / 2, WORK_H / 2 + 40
    r = 330
    stretch = 1.0 + min(0.55, 0.06 * math.log1p(multiplier) * 3)

    # Ocean envelope: an ellipse stretched along the Earth-Moon line.
    d.ellipse([cx - r * stretch, cy - r / stretch,
               cx + r * stretch, cy + r / stretch], fill=(30, 66, 112))
    d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(48, 96, 154))
    d.ellipse([cx - r, cy - r, cx + r, cy + r], outline=(104, 158, 220), width=6)

    d.text((WORK_W / 2, 110), label.upper(), font=_font("Anton-Regular.ttf", 128),
           fill=INK, anchor="ma")
    d.text((WORK_W / 2, WORK_H - 180), f"{multiplier}x NORMAL TIDAL FORCE",
           font=_font("BebasNeue-Regular.ttf", 104), fill=ACCENT, anchor="ma")
    return im


REVEAL_S = 0.8          # how long the figure takes to arrive
REVEAL_FPS = 30


def render_reveal(kind, kwargs, out_dir, stem, fps=REVEAL_FPS, seconds=REVEAL_S):
    """Write the reveal as a numbered frame sequence, plus the settled card.

    Cards hold still in the cut - panning across type is what was cropping the
    numbers - so the motion lives inside the card instead of being applied to it.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    builder = {"timer": timer_card, "number": number_card}[kind]
    frames = max(2, int(round(seconds * fps)))
    for i in range(frames):
        # Ease-out: fast at first, settling into the final value.
        linear = (i + 1) / frames
        progress = 1 - (1 - linear) ** 3
        builder(progress=progress, **kwargs).save(
            out_dir / f"{stem}_{i:04d}.png")
    final = out_dir / f"{stem}_final.png"
    builder(progress=1.0, **kwargs).save(final)
    return frames, final


def render_all(specs, out_dir):
    """Render a list of card specs. Each spec is (kind, kwargs)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    kinds = {"timer": timer_card, "number": number_card,
             "distance": distance_diagram, "tide": tide_diagram}
    paths = []
    for i, (kind, kwargs) in enumerate(specs):
        im = kinds[kind](**kwargs)
        p = out_dir / f"info_{i:03d}_{kind}.png"
        im.save(p)
        paths.append(p)
    log.info("rendered %d infographic cards", len(paths))
    return paths
