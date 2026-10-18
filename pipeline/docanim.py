"""Shots drawn in code: the map, the timeline of eras, name captions, the title.

Map: NASA's Blue Marble Next Generation (public domain), 21600x10800, graded
down a little so the pins read, with a slow eased push from a wide view to the
place. Pins pulse; labels set in Montserrat with wide tracking.

Timeline: 500 million years to today, the geological periods as bands, and a
bar for each surviving lineage drawing in from its first appearance - each
date from the episode's sources.json - crossing the red line of the asteroid.

Captions and the title are transparent PNGs that ffmpeg fades over the shot.
"""
import math
import urllib.request
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

ROOT = Path(__file__).resolve().parent.parent
FONTS = ROOT / "assets" / "fonts"
BLUE_MARBLE = ROOT / "cache" / "maps" / "bluemarble_21600.jpg"
BLUE_MARBLE_URL = ("https://eoimages.gsfc.nasa.gov/images/imagerecords/73000/73909/"
                   "world.topo.bathy.200412.3x21600x10800.jpg")
Image.MAX_IMAGE_PIXELS = None

PLACES = {
    "chicxulub": {"lat": 21.3, "lon": -89.5, "name": "CHICXULUB", "sub": "Yucatán Peninsula, Mexico",
                  "view": [(-88, 18, 80), (-89.5, 21.0, 26)], "impact": True},
    "east_london": {"lat": -33.02, "lon": 27.91, "name": "EAST LONDON", "sub": "South Africa",
                    "view": [(24, -26, 62), (27.5, -31.5, 20)]},
    "comoros": {"lat": -12.2, "lon": 44.3, "name": "COMORO ISLANDS", "sub": "where the fishermen knew it",
                "view": None},
    "sulawesi": {"lat": 1.5, "lon": 124.8, "name": "SULAWESI", "sub": "Indonesia - a second species",
                 "view": None},
}
VIEWS = {  # several places: one view that holds them all
    ("east_london", "comoros"): [(34, -23, 56), (36, -22.5, 46)],
    ("comoros", "sulawesi"): [(83, -6, 122), (86, -5, 104)],
}


def font(name, size, weight=None):
    f = ImageFont.truetype(str(FONTS / name), size)
    if weight is not None:
        try:
            f.set_variation_by_axes([weight])
        except Exception:                                        # noqa: BLE001
            pass
    return f


def tracked(draw, xy, text, fnt, fill, tracking=0.12, anchor="l"):
    """Text with letter spacing; returns its width."""
    x, y = xy
    widths = [draw.textlength(ch, font=fnt) for ch in text]
    total = sum(widths) + tracking * fnt.size * (len(text) - 1)
    if anchor == "c":
        x -= total / 2
    for ch, w in zip(text, widths):
        draw.text((x, y), ch, font=fnt, fill=fill)
        x += w + tracking * fnt.size
    return total


# ------------------------------------------------------------------- map
_base = {}


def _blue_marble():
    if "img" not in _base:
        if not BLUE_MARBLE.exists():
            BLUE_MARBLE.parent.mkdir(parents=True, exist_ok=True)
            req = urllib.request.Request(BLUE_MARBLE_URL, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=600) as r:
                BLUE_MARBLE.write_bytes(r.read())
        im = cv2.imread(str(BLUE_MARBLE), cv2.IMREAD_COLOR)[:, :, ::-1]
        _base["img"] = np.ascontiguousarray(im)
    return _base["img"]


def _xy(lon, lat, w, h):
    return (lon + 180) / 360 * w, (90 - lat) / 180 * h


def map_frames(places, seconds, out_w=3840, out_h=2160, fps=30, caption=None):
    base = _blue_marble()
    bh, bw = base.shape[:2]
    key = tuple(places)
    view = VIEWS.get(key) or PLACES[places[0]]["view"]
    (lon0, lat0, wd0), (lon1, lat1, wd1) = view
    # crop the region the whole move covers, once, and grade it
    pad = 1.15
    span_w = max(wd0, wd1) * pad
    span_h = span_w * out_h / out_w
    cx, cy = (lon0 + lon1) / 2, (lat0 + lat1) / 2
    lonA, lonB = cx - span_w / 2 - abs(lon1 - lon0), cx + span_w / 2 + abs(lon1 - lon0)
    latA, latB = cy + span_h / 2 + abs(lat1 - lat0), cy - span_h / 2 - abs(lat1 - lat0)
    xa, ya = _xy(lonA, latA, bw, bh)
    xb, yb = _xy(lonB, latB, bw, bh)
    xa, ya, xb, yb = int(max(0, xa)), int(max(0, ya)), int(min(bw, xb)), int(min(bh, yb))
    deg_per_px = 360 / bw
    # pre-shrink the region to the density the closest moment of the move needs
    dens = min(1.0, out_w / (min(wd0, wd1) / deg_per_px) * 1.05)
    region = base[ya:yb, xa:xb]
    if dens < 1.0:
        region = cv2.resize(region, (int(region.shape[1] * dens), int(region.shape[0] * dens)),
                            interpolation=cv2.INTER_AREA)
    region = region.astype(np.float32)
    # grade: a touch darker and cooler, so the white pins and labels lead
    region = region * 0.82
    lum = region.mean(axis=2, keepdims=True)
    region = lum + (region - lum) * 0.85
    region = np.clip(region, 0, 255).astype(np.uint8)
    rh, rw = region.shape[:2]
    deg_per_px = 360 / bw

    yy, xx = np.mgrid[0:out_h, 0:out_w].astype(np.float32)
    vign = 1 - 0.35 * (((xx / out_w - 0.5) ** 2 + (yy / out_h - 0.5) ** 2) / 0.5) ** 1.2
    vign = cv2.merge([(vign * 255).clip(0, 255).astype(np.uint8)] * 3)

    s = out_w / 3840
    f_name = font("Montserrat[wght].ttf", int(84 * s), 600)
    f_sub = font("Montserrat[wght].ttf", int(50 * s), 400)
    f_cap = font("Montserrat[wght].ttf", int(64 * s), 400)
    n = int(round(seconds * fps))
    for k in range(n):
        t = k / max(1, n - 1)
        e = 0.5 - 0.5 * math.cos(math.pi * t)
        lon = lon0 + (lon1 - lon0) * e
        lat = lat0 + (lat1 - lat0) * e
        wd = wd0 * (wd1 / wd0) ** e
        # output px per source px
        scale = out_w / (wd / deg_per_px) / dens
        sx, sy = _xy(lon, lat, bw, bh)
        sx, sy = (sx - xa) * dens, (sy - ya) * dens
        m = np.array([[scale, 0, out_w / 2 - scale * sx], [0, scale, out_h / 2 - scale * sy]])
        frame = cv2.warpAffine(region, m, (out_w, out_h), flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_REFLECT)
        frame = cv2.multiply(frame, vign, scale=1 / 255)
        im = Image.fromarray(frame)
        over = Image.new("RGBA", im.size, (0, 0, 0, 0))
        d = ImageDraw.Draw(over)
        for pi, pk in enumerate(places):
            p = PLACES[pk]
            appear = 0.6 + pi * 1.6
            tt = k / fps - appear
            if tt < 0:
                continue
            a = min(1.0, tt / 0.5)
            px, py = _xy(p["lon"], p["lat"], bw, bh)
            ox, oy = m @ np.array([(px - xa) * dens, (py - ya) * dens, 1.0])
            r = 15 * s
            for ring in range(2):                                   # pulses
                ph = ((tt + ring * 1.0) % 2.0) / 2.0
                rr = r + ph * 70 * s
                d.ellipse([ox - rr, oy - rr, ox + rr, oy + rr], outline=(255, 255, 255, int(200 * (1 - ph) * a)),
                          width=max(2, int(3 * s)))
            col = (255, 120, 60, int(255 * a)) if p.get("impact") else (255, 255, 255, int(255 * a))
            d.ellipse([ox - r, oy - r, ox + r, oy + r], fill=col)
            wlab = max(d.textlength(p["name"], font=f_name) * 1.15, d.textlength(p["sub"], font=f_sub) * 1.07)
            left = ox + 70 * s + wlab > out_w * 0.94
            lx = ox - 70 * s - wlab if left else ox + 70 * s
            ly = min(max(oy - 70 * s, out_h * 0.06), out_h * 0.84)
            d.line([ox - r if left else ox + r, oy, (lx + wlab + 10 * s) if left else lx - 10 * s, oy],
                   fill=(255, 255, 255, int(160 * a)), width=max(1, int(2 * s)))
            tracked(d, (lx, ly), p["name"], f_name, (255, 255, 255, int(255 * a)), 0.14)
            tracked(d, (lx, ly + 100 * s), p["sub"], f_sub, (225, 230, 235, int(230 * a)), 0.06)
        if len(places) > 1:                                         # a route between the places
            pts = []
            for pk in places:
                px, py = _xy(PLACES[pk]["lon"], PLACES[pk]["lat"], bw, bh)
                pts.append(m @ np.array([(px - xa) * dens, (py - ya) * dens, 1.0]))
            prog = min(1.0, max(0.0, (k / fps - 1.4) / 1.6))
            if prog > 0:
                (x0, y0), (x1, y1) = pts[0], pts[1]
                steps = 60
                for i in range(int(steps * prog)):
                    if i % 2:
                        continue
                    u0, u1 = i / steps, (i + 1) / steps
                    bend = lambda u: -math.sin(math.pi * u) * 0.18 * math.hypot(x1 - x0, y1 - y0)
                    pa = (x0 + (x1 - x0) * u0, y0 + (y1 - y0) * u0 + bend(u0))
                    pb = (x0 + (x1 - x0) * u1, y0 + (y1 - y0) * u1 + bend(u1))
                    d.line([pa, pb], fill=(255, 255, 255, 190), width=max(2, int(3 * s)))
        if caption:
            a = min(1.0, max(0.0, (k / fps - 0.3) / 0.8))
            tracked(d, (out_w / 2, out_h * 0.86), caption, f_cap, (255, 255, 255, int(235 * a)), 0.3, anchor="c")
        yield _composite(frame, over, s)


def _composite(frame, over, s):
    """Overlay (with a soft glow) onto frame, touching only the drawn area."""
    box = over.getbbox()
    if not box:
        return frame
    pad = int(30 * s)
    x0, y0 = max(0, box[0] - pad), max(0, box[1] - pad)
    x1, y1 = min(frame.shape[1], box[2] + pad), min(frame.shape[0], box[3] + pad)
    ov = np.asarray(over.crop((x0, y0, x1, y1)), np.float32)
    small = cv2.resize(ov, ((x1 - x0) // 4 or 1, (y1 - y0) // 4 or 1), interpolation=cv2.INTER_AREA)
    glow = cv2.resize(cv2.GaussianBlur(small, (0, 0), 1.5 * s), (x1 - x0, y1 - y0))
    base = frame[y0:y1, x0:x1].astype(np.float32)
    ga = glow[..., 3:4] / 255 / 3
    base = base * (1 - ga) + glow[..., :3] * ga
    oa = ov[..., 3:4] / 255
    base = base * (1 - oa) + ov[..., :3] * oa
    out = frame.copy()
    out[y0:y1, x0:x1] = base.clip(0, 255).astype(np.uint8)
    return out


# -------------------------------------------------------------- timeline
PERIODS = [("Cambrian", 539, 485), ("Ordovician", 485, 443), ("Silurian", 443, 419),
           ("Devonian", 419, 359), ("Carboniferous", 359, 299), ("Permian", 299, 252),
           ("Triassic", 252, 201), ("Jurassic", 201, 145), ("Cretaceous", 145, 66),
           ("Cenozoic", 66, 0)]
LINEAGES = [("NAUTILOIDS", 500), ("HORSESHOE CRABS", 445), ("COELACANTHS", 410),
            ("TUATARA'S ORDER", 244)]


def timeline_frames(seconds, out_w=3840, out_h=2160, fps=30, backdrop=None):
    s = out_w / 3840
    if backdrop is not None:
        bg = cv2.resize(backdrop, (out_w, out_h))
        bg = cv2.GaussianBlur(bg, (0, 0), 18 * s).astype(np.float32) * 0.35
    else:
        g = np.linspace(28, 12, out_h, dtype=np.float32)[:, None, None]
        bg = np.broadcast_to(g * np.array([0.55, 0.8, 1.0], np.float32), (out_h, out_w, 3)).copy()
    bg = bg.clip(0, 255).astype(np.uint8)
    x0, x1 = out_w * 0.08, out_w * 0.92
    ax_y = out_h * 0.24
    X = lambda ma: x0 + (539 - ma) / 539 * (x1 - x0)
    f_p = font("Montserrat[wght].ttf", int(36 * s), 500)
    f_l = font("Montserrat[wght].ttf", int(66 * s), 600)
    f_t = font("Montserrat[wght].ttf", int(44 * s), 400)
    f_k = font("Montserrat[wght].ttf", int(56 * s), 600)
    n = int(round(seconds * fps))
    for k in range(n):
        t = k / fps
        im = Image.fromarray(bg).convert("RGBA")
        over = Image.new("RGBA", im.size, (0, 0, 0, 0))
        d = ImageDraw.Draw(over)
        a0 = min(1.0, t / 0.8)
        for i, (name, a, b) in enumerate(PERIODS):                 # period bands
            band = (90, 130, 160) if i % 2 else (70, 105, 135)
            d.rectangle([X(a), ax_y - 72 * s, X(b), ax_y], fill=(*band, int(120 * a0)))
            if X(b) - X(a) > d.textlength(name.upper(), font=f_p) * 1.12 + 24 * s:
                tracked(d, ((X(a) + X(b)) / 2, ax_y - 58 * s), name.upper(), f_p,
                        (235, 240, 245, int(220 * a0)), 0.12, anchor="c")
        for ma in (500, 400, 300, 200, 100, 0):
            d.line([X(ma), ax_y, X(ma), ax_y + 14 * s], fill=(255, 255, 255, int(200 * a0)), width=int(2 * s))
            lab = "TODAY" if ma == 0 else f"{ma} MYA"
            tracked(d, (X(ma), ax_y + 22 * s), lab, f_t, (220, 225, 230, int(220 * a0)), 0.08, anchor="c")
        # the asteroid
        ka = min(1.0, max(0.0, (t - 0.8) / 0.6))
        d.line([X(66), ax_y - 70 * s, X(66), out_h * 0.90], fill=(255, 90, 60, int(230 * ka)), width=int(4 * s))
        tracked(d, (X(66) - 16 * s, out_h * 0.92), "ASTEROID  66 MYA", f_k, (255, 120, 90, int(240 * ka)), 0.1,
                anchor="c")
        for i, (name, ma) in enumerate(LINEAGES):                  # lineages drawing in
            st = 1.4 + i * 0.9
            p = min(1.0, max(0.0, (t - st) / 2.2))
            if p <= 0:
                continue
            e = 1 - (1 - p) ** 3
            y = ax_y + (330 + i * 300) * s
            xe = X(ma) + (X(0) - X(ma)) * e
            d.rounded_rectangle([X(ma), y, xe, y + 34 * s], radius=int(17 * s), fill=(240, 220, 170, 235))
            d.ellipse([X(ma) - 20 * s, y - 3 * s, X(ma) + 20 * s, y + 37 * s], fill=(255, 255, 255, 255))
            al = int(255 * min(1.0, p * 3))
            tracked(d, (X(ma), y - 92 * s), name, f_l, (255, 255, 255, al), 0.1)
            tracked(d, (X(ma) + d.textlength(name, font=f_l) * 1.14 + 50 * s, y - 80 * s),
                    f"{ma} MILLION YEARS", f_t, (240, 220, 170, al), 0.08)
        im = Image.alpha_composite(im, over)
        yield np.asarray(im.convert("RGB"))


# ------------------------------------------------- captions and the title
def caption_png(name, latin, path, out_w=3840, out_h=2160):
    s = out_w / 3840
    im = Image.new("RGBA", (out_w, out_h), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    x, y = out_w * 0.065, out_h * 0.78
    f_n = font("Montserrat[wght].ttf", int(64 * s), 600)
    f_l = font("CormorantGaramond-Italic[wght].ttf", int(62 * s), 500)
    shadow = Image.new("RGBA", im.size, (0, 0, 0, 0))
    ds = ImageDraw.Draw(shadow)
    for dd, fill in ((ds, (0, 0, 0, 170)), (d, (255, 255, 255, 255))):
        off = 3 * s if dd is ds else 0
        dd.line([x + off, y - 22 * s + off, x + 120 * s + off, y - 22 * s + off], fill=fill, width=int(4 * s))
        tracked(dd, (x + off, y + off), name.upper(), f_n, fill, 0.16)
        dd.text((x + off, y + 92 * s + off), latin, font=f_l, fill=fill if dd is ds else (235, 225, 200, 255))
    shadow = shadow.filter(ImageFilter.GaussianBlur(8 * s))
    Image.alpha_composite(shadow, im).save(path)
    return path


def title_png(title, sub, path, out_w=3840, out_h=2160):
    s = out_w / 3840
    im = Image.new("RGBA", (out_w, out_h), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    f_t = font("Montserrat[wght].ttf", int(190 * s), 300)
    f_s = font("Montserrat[wght].ttf", int(46 * s), 500)
    shadow = Image.new("RGBA", im.size, (0, 0, 0, 0))
    ds = ImageDraw.Draw(shadow)
    for dd, fill in ((ds, (0, 0, 0, 200)), (d, (255, 255, 255, 255))):
        w = tracked(dd, (out_w / 2, out_h * 0.42), title, f_t, fill, 0.22, anchor="c")
        dd.line([out_w / 2 - w * 0.25, out_h * 0.585, out_w / 2 + w * 0.25, out_h * 0.585], fill=fill,
                width=int(3 * s))
        tracked(dd, (out_w / 2, out_h * 0.61), sub, f_s, fill if dd is ds else (235, 225, 200, 255), 0.35,
                anchor="c")
    shadow = shadow.filter(ImageFilter.GaussianBlur(14 * s))
    Image.alpha_composite(shadow, im).save(path)
    return path
