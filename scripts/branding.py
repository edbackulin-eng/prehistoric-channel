"""Draw the channel avatar (800x800) and banner (2560x1440), and optionally apply
the description, keywords and banner through the API.

    python scripts/branding.py              # draw both into branding/
    python scripts/branding.py --apply      # also set description, keywords, banner

The banner keeps everything that matters inside YouTube's 1546x423 safe area,
the part every device shows. The avatar cannot be set through the API; upload
branding/avatar.png in YouTube Studio.
"""
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from core import fonts  # noqa: E402

OUT = ROOT / "branding"
CYAN = (90, 220, 200)
RED = (232, 44, 52)

DESCRIPTION = (
    "Real internet mysteries, told properly. Puzzle hunts that recruited strangers, "
    "broadcasts nobody can explain, channels that posted thousands of videos and "
    "vanished, lost media, and the folklore the internet made up about itself.\n\n"
    "Every episode is written from at least two open sources, listed in its "
    "description. No real tragedies, no victims, no private people - just the "
    "tabs nobody closed.\n\n"
    "New episodes Monday, Tuesday, Thursday, Friday and Saturday."
)
KEYWORDS = ('"internet mysteries" "unsolved mysteries" ARG "alternate reality game" '
            '"lost media" "internet folklore" "creepy internet" "Cicada 3301" '
            '"internet history" documentary')


def _glow_bg(w, h, seed=0):
    rng = np.random.default_rng(seed)
    y, x = np.ogrid[:h, :w]
    g = np.exp(-(((x - w * 0.62) / (w * 0.45)) ** 2 + ((y - h * 0.5) / (h * 0.55)) ** 2))[..., None]
    base = np.array((7, 9, 13), np.float32) + g * np.array(CYAN, np.float32) * 0.16
    base += rng.normal(0, 4, (h, w, 1))
    a = np.clip(base, 0, 255).astype(np.uint8)
    a[::4] = (a[::4] * 0.85).astype(np.uint8)
    return Image.fromarray(a)


def _tab(d, x, y, w, h, title, closed=True):
    d.rounded_rectangle((x, y, x + w, y + h + 30), 26, fill=(26, 32, 44), outline=(52, 62, 80), width=4)
    room = w - h * 0.45 - (h * 1.15 if closed else h * 0.45)     # never under the x
    size, _ = fonts.fit(title, "label", room, h * 0.7, int(h * 0.62), min_size=10)
    f = fonts.font("label", size)
    d.text((x + h * 0.45, y + (h - size) / 2), title, font=f, fill=(200, 210, 218))
    if closed:
        cx, cy, r = x + w - h * 0.6, y + h * 0.5, h * 0.22
        d.line((cx - r, cy - r, cx + r, cy + r), fill=RED, width=max(6, int(h * 0.09)))
        d.line((cx - r, cy + r, cx + r, cy - r), fill=RED, width=max(6, int(h * 0.09)))


def avatar(path):
    s = 800
    im = _glow_bg(s, s, 1)
    d = ImageDraw.Draw(im)
    _tab(d, 90, 250, 620, 150, "CT", closed=True)
    d.rounded_rectangle((90, 380, 710, 600), 18, fill=(14, 18, 26), outline=(52, 62, 80), width=4)
    d.rectangle((130, 440, 520, 456), fill=(60, 72, 90))
    d.rectangle((130, 490, 640, 506), fill=(60, 72, 90))
    d.rectangle((130, 540, 420, 556), fill=(60, 72, 90))
    im.save(path)
    return path


def banner(path):
    w, h = 2560, 1440
    im = _glow_bg(w, h, 2)
    d = ImageDraw.Draw(im)
    # a row of tabs across the top, all but one closed
    titles = ["CICADA 3301", "UVB-76", "KRYPTOS", "POLYBIUS", "11B-X-1371", "THE BACKROOMS"]
    for k, t in enumerate(titles):
        _tab(d, 120 + k * 400, 300, 380, 90, t, closed=(k != 2))
    # the safe area: 1546x423 centred
    sx0, sy0 = (w - 1546) // 2, (h - 423) // 2
    size, _ = fonts.fit("CLOSED TABS", "title", 1300, 240, 230)
    f = fonts.font("title", size)
    tw = f.getlength("CLOSED TABS")
    d.text(((w - tw) / 2, sy0 + 40), "CLOSED TABS", font=f, fill=(255, 255, 255),
           stroke_width=8, stroke_fill=(0, 0, 0))
    tag = "REAL INTERNET MYSTERIES  -  MON TUE THU FRI SAT"
    ts, _ = fonts.fit(tag, "label", 1400, 90, 80)
    tf = fonts.font("label", ts)
    d.text(((w - tf.getlength(tag)) / 2, sy0 + 40 + size * 1.12 + 24), tag, font=tf, fill=CYAN)
    # REC sits inside the safe area, below the tab row, never on a tab
    d.ellipse((sx0 + 10, sy0 - 40, sx0 + 44, sy0 - 6), fill=(232, 30, 30))
    d.text((sx0 + 60, sy0 - 52), "REC", font=fonts.font("label", 60), fill=(255, 255, 255))
    d.text((sx0 + 1546 - 300, sy0 + 423 + 50), "03:14 AM", font=fonts.font("label", 60),
           fill=(255, 255, 255))
    im.save(path, optimize=True)
    return path


def main():
    OUT.mkdir(exist_ok=True)
    a, b = avatar(OUT / "avatar.png"), banner(OUT / "banner.png")
    print("drew", a, b)
    if "--apply" in sys.argv:
        from pipeline import youtube
        cid = youtube.set_branding(DESCRIPTION, KEYWORDS, b)
        print("branding applied to", cid)
    return 0


if __name__ == "__main__":
    sys.exit(main())
