"""Animations drawn by code: interfaces, documents, counters, annotations.

Every visible word in these shots is drawn here, with the font fitted to its box,
so nothing is ever a generated smear of fake letters. Each function writes a
1920x1080, 30 fps clip and returns the moments that should carry a sound
(key, click, blip, tick, thump), which the sound stage turns into effects.

The look is the channel's lit night: deep blue-violet, teal and magenta light,
a warm amber accent for the thing that matters. No real logos, no real site chrome.
"""
import math
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from core import fonts

W, H, FPS = 1920, 1080, 30
# Lit night, not black: a deep blue-violet room with teal and magenta light in it.
BG = (24, 28, 46)
PANEL = (34, 40, 60)
PANEL_HI = (50, 58, 84)
TEXT = (214, 224, 232)
DIM = (120, 132, 146)
SCREEN = (90, 220, 200)          # the cold screen colour
RED = (232, 44, 52)
WARM = (255, 196, 92)
MAGENTA = (236, 72, 160)


# --- plumbing --------------------------------------------------------------

class Clip:
    """Feed PIL frames to ffmpeg; collect sound cues."""

    def __init__(self, out_path, duration):
        self.out = Path(out_path)
        self.out.parent.mkdir(parents=True, exist_ok=True)
        self.n = max(1, round(duration * FPS))
        self.cues = []
        self.proc = subprocess.Popen(
            ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo",
             "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
             "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p",
             str(self.out)], stdin=subprocess.PIPE)

    def frame(self, im):
        self.proc.stdin.write(im.convert("RGB").tobytes())

    def cue(self, t, kind):
        self.cues.append({"t": round(t, 3), "kind": kind})

    def close(self):
        self.proc.stdin.close()
        if self.proc.wait() != 0:
            raise RuntimeError(f"ffmpeg failed writing {self.out}")
        return {"path": str(self.out), "cues": self.cues}


def ease(x):
    x = min(1.0, max(0.0, x))
    return x * x * (3 - 2 * x)


def ease_out(x):
    x = min(1.0, max(0.0, x))
    return 1 - (1 - x) ** 3


_BACKDROPS = {}


def backdrop(glow=SCREEN):
    """A lit night room behind the panel: deep blue-violet, a teal pool of light on
    the left, a magenta one on the right, the panel's own glow in the middle.
    Cached per colour."""
    if glow not in _BACKDROPS:
        y, x = np.ogrid[:H, :W]
        d = np.sqrt(((x - W / 2) / (W / 2)) ** 2 + ((y - H / 2) / (H / 2)) ** 2)
        vig = np.clip(1.0 - 0.30 * d ** 2, 0.62, 1.0)[..., None]
        base = np.zeros((H, W, 3), np.float32) + np.array(BG, np.float32)
        teal = np.exp(-(((x - W * 0.12) / 620) ** 2 + ((y - H * 0.85) / 520) ** 2))[..., None]
        mag = np.exp(-(((x - W * 0.90) / 600) ** 2 + ((y - H * 0.15) / 480) ** 2))[..., None]
        g = np.exp(-(((x - W / 2) / 760) ** 2 + ((y - H / 2) / 460) ** 2))[..., None]
        base += teal * np.array(SCREEN, np.float32) * 0.42
        base += mag * np.array(MAGENTA, np.float32) * 0.36
        base += g * np.array(glow, np.float32) * 0.22
        # a faint grain so the backdrop is a room, not a flat fill
        rng = np.random.default_rng(len(_BACKDROPS))
        base += rng.normal(0, 2.2, (H, W, 1)).astype(np.float32)
        _BACKDROPS[glow] = Image.fromarray(np.clip(base * vig, 0, 255).astype(np.uint8))
    return _BACKDROPS[glow].copy()


def text_box(draw, xy, box, text, kind="sans", max_size=46, fill=TEXT, spacing=1.25):
    """Draw text fitted to a box. Returns the size it was drawn at."""
    x, y = xy
    bw, bh = box
    size, lines = fonts.fit(text, kind, bw, bh, max_size, spacing=spacing)
    f = fonts.font(kind, size)
    for i, line in enumerate(lines):
        draw.text((x, y + i * size * spacing), line, font=f, fill=fill)
    return size


def _cover(path_or_img):
    from pipeline import grade
    im = path_or_img if isinstance(path_or_img, Image.Image) else Image.open(path_or_img)
    from pipeline import motion
    im = grade.grade(motion.trim_bands(im.convert("RGB")))
    s = max(W / im.width, H / im.height)
    im = im.resize((math.ceil(im.width * s), math.ceil(im.height * s)), Image.LANCZOS)
    x, y = (im.width - W) // 2, (im.height - H) // 2
    return im.crop((x, y, x + W, y + H))


# --- a generic browser window ------------------------------------------------

def _window(im, title, address, typed_address, cursor_on):
    d = ImageDraw.Draw(im)
    x0, y0, x1, y1 = 230, 130, W - 230, H - 130
    d.rounded_rectangle((x0, y0, x1, y1), 18, fill=PANEL, outline=PANEL_HI, width=2)
    d.rounded_rectangle((x0, y0, x1, y0 + 64), 18, fill=PANEL_HI)
    d.rectangle((x0, y0 + 40, x1, y0 + 64), fill=PANEL_HI)
    for i, c in enumerate(((235, 90, 80), (240, 190, 70), (90, 200, 110))):
        d.ellipse((x0 + 24 + i * 30, y0 + 22, x0 + 42 + i * 30, y0 + 40), fill=c)
    d.rounded_rectangle((x0 + 140, y0 + 12, x0 + 560, y0 + 56), 10, fill=PANEL)
    text_box(d, (x0 + 160, y0 + 20), (380, 30), title, "sans", 22, DIM)
    d.rounded_rectangle((x0 + 30, y0 + 84, x1 - 30, y0 + 136), 12, fill=BG)
    shown = address[:typed_address]
    f = fonts.font("mono", 28)
    d.text((x0 + 56, y0 + 94), shown, font=f, fill=TEXT)
    if cursor_on:
        cx = x0 + 56 + f.getlength(shown) + 4
        d.rectangle((cx, y0 + 96, cx + 3, y0 + 126), fill=SCREEN)
    return (x0 + 70, y0 + 180, x1 - 70, y1 - 50)


def browser(out_path, duration, address, heading, body, title="New tab", type_cps=34):
    """Address typed, then the page heading and body typed."""
    clip = Clip(out_path, duration)
    addr_t = min(duration * 0.25, len(address) / 22)
    body_t0 = addr_t + 0.3
    page = f"{heading}\n{body}"
    # Typing always finishes by 85% of the clip, however long the text is.
    type_cps = max(type_cps, len(page) / max(0.5, duration * 0.85 - body_t0))
    for i in range(clip.n):
        t = i / FPS
        im = backdrop()
        typed = len(address) if t >= addr_t else int(len(address) * t / max(addr_t, 1e-3))
        blink = int(t * 2) % 2 == 0
        area = _window(im, title, address, typed, t < addr_t or (blink and t < body_t0))
        d = ImageDraw.Draw(im)
        if t >= body_t0:
            shown = page[:int((t - body_t0) * type_cps)]
            head, _, rest = shown.partition("\n")
            text_box(d, (area[0], area[1] + 20), (area[2] - area[0], 140), head, "sans_bold", 92, TEXT)
            if rest:
                text_box(d, (area[0], area[1] + 190),
                         (area[2] - area[0], area[3] - area[1] - 200), rest, "sans", 68,
                         (196, 206, 214))
        clip.frame(im)
    for k in range(len(address)):
        clip.cue(k * addr_t / max(1, len(address)), "key")
    for k in range(0, len(page), 3):
        if body_t0 + k / type_cps < duration:
            clip.cue(body_t0 + k / type_cps, "key")
    return clip.close()


# --- a cursor gliding to a target and clicking --------------------------------

def _arrow(d, x, y, scale=1.0):
    pts = [(0, 0), (0, 34), (9, 26), (16, 42), (22, 39), (15, 24), (27, 24)]
    d.polygon([(x + px * scale, y + py * scale) for px, py in pts],
              fill=(250, 250, 250), outline=(10, 10, 10))


def cursor_click(out_path, duration, background, start=(0.25, 0.75), target=(0.62, 0.42)):
    clip = Clip(out_path, duration)
    base = _cover(background)
    click_t = duration * 0.62
    for i in range(clip.n):
        t = i / FPS
        k = ease(t / click_t)
        x = (start[0] + (target[0] - start[0]) * k) * W
        y = (start[1] + (target[1] - start[1]) * k) * H
        im = base.copy()
        if t >= click_t:
            r = 10 + 90 * ease_out((t - click_t) / 0.5)
            a = max(0.0, 1 - (t - click_t) / 0.5)
            if a > 0:
                ov = Image.new("RGBA", im.size)
                ImageDraw.Draw(ov).ellipse((x - r, y - r, x + r, y + r),
                                           outline=(*SCREEN, int(220 * a)), width=5)
                im = Image.alpha_composite(im.convert("RGBA"), ov).convert("RGB")
        _arrow(ImageDraw.Draw(im), x, y, 1.6)
        clip.frame(im)
    clip.cue(click_t, "click")
    return clip.close()


# --- chat and notifications --------------------------------------------------

def chat(out_path, duration, messages, header="Thread"):
    """Bubbles appear one by one, alternating sides. messages: [(who, text)]."""
    clip = Clip(out_path, duration)
    step = duration * 0.85 / max(1, len(messages))
    # The panel is sized to what it holds and centred, so one or two messages do
    # not sit at the top of an empty window.
    heights = []
    for _, msg in messages:
        size, lines = fonts.fit(msg, "sans", 960 - 70, 240, 54)
        heights.append(int(len(lines) * size * 1.25 + 90))
    content = 140 + sum(heights) + 26 * max(0, len(heights) - 1) + 40
    top = max(70, (H - content) // 2)
    for i in range(clip.n):
        t = i / FPS
        im = backdrop()
        d = ImageDraw.Draw(im)
        d.rounded_rectangle((330, top, W - 330, top + content), 22, fill=PANEL)
        text_box(d, (380, top + 30), (W - 760, 60), header, "sans_bold", 44, DIM)
        y = top + 140
        for k, (who, msg) in enumerate(messages):
            t0 = k * step
            if t < t0:
                break
            p = ease_out((t - t0) / 0.28)
            right = k % 2 == 1
            bw = 960
            size, lines = fonts.fit(msg, "sans", bw - 70, 240, 54)
            bh = int(len(lines) * size * 1.25 + 90)
            x0 = (W - 380 - bw) if right else 380
            off = int((1 - p) * 40)
            d.rounded_rectangle((x0, y + off, x0 + bw, y + off + bh), 20,
                                fill=(32, 64, 70) if right else PANEL_HI)
            d.text((x0 + 30, y + off + 14), who, font=fonts.font("sans_bold", 30),
                   fill=SCREEN if right else WARM)
            for j, line in enumerate(lines):
                d.text((x0 + 30, y + off + 58 + j * size * 1.25), line,
                       font=fonts.font("sans", size), fill=TEXT)
            y += bh + 26
        clip.frame(im)
    for k in range(len(messages)):
        clip.cue(k * step, "blip")
    return clip.close()


def notification(out_path, duration, background, app, text):
    """A notification card slides in over a still. The card grows to hold the whole
    sentence: nothing is ever cut to fit."""
    clip = Clip(out_path, duration)
    base = _cover(background)
    t0 = duration * 0.2
    size, lines = fonts.fit(text, "sans", 900, 220, 50, min_size=30)
    card_h = int(130 + len(lines) * size * 1.25 + 30)
    for i in range(clip.n):
        t = i / FPS
        im = base.copy()
        p = ease_out((t - t0) / 0.35)
        if p > 0:
            ov = Image.new("RGBA", im.size)
            d = ImageDraw.Draw(ov)
            x1 = W - 70
            x0 = x1 - 1080
            y0 = int(-card_h - 30 + (70 + card_h + 30) * p)
            d.rounded_rectangle((x0, y0, x1, y0 + card_h), 30, fill=(24, 28, 44, 240),
                                outline=(*PANEL_HI, 255), width=2)
            d.ellipse((x0 + 36, y0 + 38, x0 + 108, y0 + 110), fill=(*SCREEN, 255))
            d.text((x0 + 136, y0 + 36), app, font=fonts.font("sans_bold", 44), fill=TEXT)
            for j, line in enumerate(lines):
                d.text((x0 + 136, y0 + 104 + j * size * 1.25), line,
                       font=fonts.font("sans", size), fill=(214, 222, 230))
            im = Image.alpha_composite(im.convert("RGBA"), ov).convert("RGB")
        clip.frame(im)
    clip.cue(t0, "blip")
    return clip.close()


# --- a counter --------------------------------------------------------------

def counter(out_path, duration, label, end, start=0, prefix="", suffix=""):
    clip = Clip(out_path, duration)
    run = duration * 0.7
    last = None
    for i in range(clip.n):
        t = i / FPS
        v = start + (end - start) * ease_out(t / run)
        value = f"{prefix}{int(round(v)):,}{suffix}"
        im = backdrop(WARM)
        d = ImageDraw.Draw(im)
        size, _ = fonts.fit(f"{prefix}{end:,}{suffix}", "title", 1500, 380, 320)
        f = fonts.font("title", size)
        bx = f.getbbox(value)
        top = 250
        d.text(((W - (bx[2] - bx[0])) / 2 - bx[0], top - bx[1]), value, font=f, fill=TEXT)
        lab = label.upper()
        lsize, _ = fonts.fit(lab, "label", W - 520, 110, 90)
        lf = fonts.font("label", lsize)
        lb = lf.getbbox(lab)
        d.text(((W - (lb[2] - lb[0])) / 2 - lb[0], top + (bx[3] - bx[1]) + 50), lab,
               font=lf, fill=DIM)
        clip.frame(im)
        if value != last and t < run and i % 3 == 0:
            clip.cue(t, "tick")
        last = value
    return clip.close()


# --- a timeline --------------------------------------------------------------

def timeline(out_path, duration, events):
    """A line draws left to right; each event pops in. events: [(when, text)]."""
    clip = Clip(out_path, duration)
    n = len(events)
    xs = [260 + (W - 520) * (k / max(1, n - 1)) for k in range(n)] if n > 1 else [W / 2]
    for i in range(clip.n):
        t = i / FPS
        im = backdrop()
        d = ImageDraw.Draw(im)
        p = ease(t / (duration * 0.8))
        d.line((220, H / 2, 220 + (W - 440) * p, H / 2), fill=DIM, width=6)
        for k, (when, what) in enumerate(events):
            if (xs[k] - 220) / (W - 440) > p:
                continue
            d.ellipse((xs[k] - 16, H / 2 - 16, xs[k] + 16, H / 2 + 16),
                      fill=RED if k == n - 1 else SCREEN)
            box_y = H / 2 - 250 if k % 2 == 0 else H / 2 + 50
            text_box(d, (xs[k] - 210, box_y - 20), (420, 100), when, "label", 104, WARM)
            text_box(d, (xs[k] - 210, box_y + 90), (420, 150), what, "sans", 44, TEXT)
        clip.frame(im)
    for k in range(n):
        clip.cue(duration * 0.8 * (xs[k] - 220) / (W - 440), "blip")
    return clip.close()


# --- a red circle and underline drawn onto a still ---------------------------

def annotate(out_path, duration, background, circle=(0.5, 0.5, 0.18), underline=None):
    """circle: (cx, cy, r) relative; underline: (x0, x1, y) relative."""
    clip = Clip(out_path, duration)
    base = _cover(background)
    draw_t = 0.9
    t0 = duration * 0.25
    cx, cy, r = circle[0] * W, circle[1] * H, circle[2] * H
    for i in range(clip.n):
        t = i / FPS
        z = 1.0 + 0.04 * ease(t / duration)
        im = base.resize((int(W * z), int(H * z)), Image.BILINEAR)
        ox, oy = (im.width - W) // 2, (im.height - H) // 2
        im = im.crop((ox, oy, ox + W, oy + H))
        d = ImageDraw.Draw(im)
        p = ease((t - t0) / draw_t)
        if p > 0:
            pts = []
            for k in range(int(140 * p) + 1):
                a = -1.2 + 2 * math.pi * 1.08 * k / 140
                rr = r * (1 + 0.04 * math.sin(a * 3))
                pts.append((cx + rr * 1.25 * math.cos(a), cy + rr * math.sin(a)))
            if len(pts) > 1:
                d.line(pts, fill=RED, width=10, joint="curve")
        if underline and p >= 1:
            q = ease((t - t0 - draw_t) / 0.5)
            x0, x1, y = underline[0] * W, underline[1] * W, underline[2] * H
            d.line((x0, y, x0 + (x1 - x0) * q, y + 4), fill=RED, width=9)
        clip.frame(im)
    clip.cue(t0, "thump")
    return clip.close()


# --- a document with censor bars ---------------------------------------------

def censor(out_path, duration, heading, lines, redact):
    """Lines appear; then black bars slide over the words in `redact`."""
    clip = Clip(out_path, duration)
    appear = duration * 0.35
    bar_t0 = appear + 0.2
    for i in range(clip.n):
        t = i / FPS
        im = backdrop((200, 200, 190))
        d = ImageDraw.Draw(im)
        d.rectangle((300, 60, W - 300, H - 60), fill=(226, 222, 210))
        text_box(d, (380, 130), (W - 760, 110), heading.upper(), "label", 96, (40, 40, 40))
        y = 300
        shown = int(len(lines) * ease(t / appear) + 0.999)
        for line in lines[:shown]:
            size, wrapped = fonts.fit(line, "mono", W - 760, 160, 58)
            f = fonts.font("mono", size)
            for j, part in enumerate(wrapped):
                yy = y + j * size * 1.3
                d.text((380, yy), part, font=f, fill=(30, 30, 30))
                for m, word in enumerate(redact):
                    pos = part.find(word)
                    if pos < 0:
                        continue
                    x0 = 380 + f.getlength(part[:pos])
                    x1 = x0 + f.getlength(word)
                    p = ease((t - bar_t0 - 0.25 * m) / 0.35)
                    if p > 0:
                        d.rectangle((x0 - 6, yy - 2, x0 - 6 + (x1 - x0 + 12) * p,
                                     yy + size * 1.1), fill=(8, 8, 8))
            y += len(wrapped) * size * 1.3 + 24
        clip.frame(im)
    for m in range(len(redact)):
        clip.cue(bar_t0 + 0.25 * m, "thump")
    return clip.close()


# --- a pull quote ----------------------------------------------------------------

def quote(out_path, duration, text, accent=None):
    """A line of the narration set large, word by word, one word in red."""
    clip = Clip(out_path, duration)
    words = text.split()
    # the longest word, and among equals the later one: the stress of a line
    # falls at its end ("things VANISH", not "THINGS vanish")
    accent = accent or max(reversed(words), key=lambda w: len(w.strip(".,!?;:\"'")), default="")
    size, lines = fonts.fit(text.upper(), "title", W * 0.8, H * 0.62, 150, spacing=1.08)
    f = fonts.font("title", size)
    reveal = duration * 0.55
    for i in range(clip.n):
        t = i / FPS
        im = backdrop()
        d = ImageDraw.Draw(im)
        shown = int(len(words) * ease_out(t / reveal) + 0.999)
        k = 0
        y = (H - len(lines) * size * 1.08) / 2
        for line in lines:
            parts = line.split()
            x = (W - f.getlength(line)) / 2
            for p in parts:
                if k < shown:
                    red = p.strip(".,!?;:\"'").lower() == accent.strip(".,!?;:\"'").lower()
                    d.text((x, y), p, font=f, fill=RED if red else TEXT,
                           stroke_width=max(3, size // 26), stroke_fill=(0, 0, 0))
                x += f.getlength(p + " ")
                k += 1
            y += size * 1.08
        clip.frame(im)
    for k in range(len(words)):
        clip.cue(reveal * k / max(1, len(words)), "tick")
    return clip.close()


# --- a world map with the places the narration names -------------------------

_MAP = {}


def _map_base():
    """The land in equirectangular projection, lit teal on the night backdrop."""
    if "base" not in _MAP:
        from pipeline import geo
        im = backdrop(SCREEN).convert("RGBA")
        land = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        d = ImageDraw.Draw(land)
        x0, y0, mw, mh = 120, 150, W - 240, (W - 240) / 2      # 2:1 world
        for ring in geo.land():
            pts = [(x0 + (lon + 180) / 360 * mw, y0 + (90 - lat) / 180 * mh) for lon, lat in ring]
            if len(pts) > 2:
                d.polygon(pts, fill=(44, 92, 104, 255), outline=(110, 210, 200, 255))
        # a faint graticule every 30 degrees
        for lon in range(-180, 181, 30):
            x = x0 + (lon + 180) / 360 * mw
            d.line((x, y0, x, y0 + mh), fill=(120, 140, 180, 40), width=1)
        for lat in range(-60, 91, 30):
            y = y0 + (90 - lat) / 180 * mh
            d.line((x0, y, x0 + mw, y), fill=(120, 140, 180, 40), width=1)
        _MAP["base"] = Image.alpha_composite(im, land).convert("RGB")
        _MAP["frame"] = (x0, y0, mw, mh)
    return _MAP["base"].copy(), _MAP["frame"]


def worldmap(out_path, duration, places, title=""):
    """Dots appear one by one on the places named, each with a pulse and a label.
    places: [(name, lat, lon)]."""
    clip = Clip(out_path, duration)
    base, (x0, y0, mw, mh) = _map_base()
    n = max(1, len(places))
    step = duration * 0.7 / n
    pos = [(x0 + (lon + 180) / 360 * mw, y0 + (90 - lat) / 180 * mh) for _, lat, lon in places]
    # Each label takes the first of four corners around its dot that covers no
    # other dot and no label already placed.
    f = fonts.font("sans_bold", 40)
    boxes = []
    for (name, _, _), (px, py) in zip(places, pos):
        lw = f.getlength(name)
        choice = None
        for dx, dy in ((24, -66), (24, 20), (-lw - 48, -66), (-lw - 48, 20), (24, -120), (24, 74)):
            lx = min(W - 60 - lw, max(60, px + dx))
            b = (lx - 12, py + dy - 6, lx + lw + 12, py + dy + 50)
            hits_dot = any(b[0] - 16 < qx < b[2] + 16 and b[1] - 16 < qy < b[3] + 16 for qx, qy in pos)
            hits_box = any(not (b[2] < c[0] or b[0] > c[2] or b[3] < c[1] or b[1] > c[3]) for c in boxes)
            if not hits_dot and not hits_box:
                choice = b
                break
        boxes.append(choice or (min(W - 60 - lw, px + 24) - 12, py - 72, min(W - 60 - lw, px + 24) + lw + 12, py - 16))
    for i in range(clip.n):
        t = i / FPS
        z = 1.0 + 0.03 * ease(t / duration)
        im = base.resize((int(W * z), int(H * z)), Image.BILINEAR)
        ox, oy = (im.width - W) // 2, (im.height - H) // 2
        im = im.crop((ox, oy, ox + W, oy + H))
        d = ImageDraw.Draw(im)
        if title:
            text_box(d, (120, 50), (W - 240, 80), title.upper(), "label", 72, WARM)

        def zx(x, y):
            return (x - W / 2) * z + W / 2, (y - H / 2) * z + H / 2
        for k, ((name, _, _), (px, py), b) in enumerate(zip(places, pos, boxes)):
            t0 = 0.3 + k * step
            if t < t0:
                continue
            px, py = zx(px, py)
            p = ease_out((t - t0) / 0.3)
            pulse = ((t - t0) % 1.4) / 1.4
            rr = 14 + 46 * pulse
            d.ellipse((px - rr, py - rr, px + rr, py + rr), outline=MAGENTA, width=max(1, int(4 * (1 - pulse))))
            r = 13 * p
            d.ellipse((px - r, py - r, px + r, py + r), fill=RED, outline=(255, 255, 255), width=3)
            bx0, by0 = zx(b[0], b[1])
            bx1, by1 = zx(b[2], b[3])
            d.rounded_rectangle((bx0, by0, bx1, by1), 10, fill=(20, 22, 36))
            d.text((bx0 + 12, by0 + 6), name, font=f, fill=TEXT)
        clip.frame(im)
    for k in range(len(places)):
        clip.cue(0.3 + k * step, "blip")
    return clip.close()
