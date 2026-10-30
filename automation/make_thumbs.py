"""Three German thumbnails per episode in the manner of the niche's top 30, and the pick.

    python automation/make_thumbs.py <slug>

What the most viewed thumbnails of the niche do (analysis/comp_dino/thumbs_top30.jpg):
bright, saturated colour; one animal huge in frame, head or eye, plainly
prehistoric; and a trick that makes you look twice. The three variants of an
episode (planned in episodes/<slug>/thumbs.json by automation/plan_episode.py)
each use one:
  scale       a tiny diver beside the giant, ringed in yellow
  split       the fossil on the left, the living or reconstructed animal on the right
  silhouette  a T. rex's dark silhouette looming behind the animal

Rules (config/channel.json "thumbnail"): two or three German words set in code,
white with one red accent word; the bottom-right 25 % x 20 % left empty (YouTube
prints the running time there); the words never cross the animal's head; every
line readable at 336x189 (pipeline/doc_thumb.py measures it). Living animals
are real photographs, extinct ones FLUX frames. Each base goes through the
species check; Gemini-flash then picks the variant that works best at small
size, and that one becomes thumbnail.jpg - the one that goes to YouTube.
"""
import json
import os
import sys
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from core.logger import get_logger  # noqa: E402
from pipeline import doc_thumb, speciescheck  # noqa: E402

log = get_logger("thumbs")
CFG = json.loads((ROOT / "config" / "channel.json").read_text(encoding="utf-8"))["thumbnail"]
SIZE = (1280, 720)
UA = {"User-Agent": "prehistoric-channel/1.0 (thumbnails)"}
FONT = ROOT / "assets" / "fonts" / "Anton-Regular.ttf"
YELLOW, RED, WHITE = (255, 214, 0), (232, 28, 28), (255, 255, 255)

# outlines, in a 0-1 box, drawn in code (no stock art)
DIVER = [(0.50, 0.00), (0.58, 0.05), (0.56, 0.14), (0.70, 0.22), (1.00, 0.20), (1.00, 0.27), (0.70, 0.31),
         (0.62, 0.40), (0.66, 0.62), (0.80, 0.95), (0.70, 1.00), (0.55, 0.70), (0.45, 0.70), (0.30, 1.00),
         (0.20, 0.95), (0.34, 0.62), (0.38, 0.40), (0.30, 0.31), (0.00, 0.27), (0.00, 0.20), (0.30, 0.22),
         (0.44, 0.14), (0.42, 0.05)]
TREX = [(0.00, 0.55), (0.18, 0.45), (0.32, 0.42), (0.42, 0.38), (0.50, 0.30), (0.56, 0.18), (0.64, 0.10),
        (0.80, 0.08), (0.95, 0.12), (1.00, 0.20), (0.97, 0.26), (0.86, 0.27), (0.92, 0.32), (0.86, 0.35),
        (0.74, 0.31), (0.70, 0.40), (0.73, 0.46), (0.68, 0.47), (0.64, 0.42), (0.60, 0.52), (0.62, 0.70),
        (0.70, 0.95), (0.60, 1.00), (0.52, 0.74), (0.46, 0.66), (0.40, 0.80), (0.44, 1.00), (0.34, 1.00),
        (0.33, 0.70), (0.25, 0.60), (0.10, 0.60)]


def _key(name):
    return os.environ.get(name, "").strip()


def pixabay_photo(query, dest, n=0):
    q = urllib.parse.urlencode({"key": _key("PIXABAY_API_KEY"), "q": query, "image_type": "photo",
                                "per_page": 20, "safesearch": "true", "order": "popular"})
    hits = json.load(urllib.request.urlopen(urllib.request.Request(f"https://pixabay.com/api/?{q}", headers=UA),
                                            timeout=60))["hits"]
    hits = [h for h in hits if h["imageWidth"] >= 1600]
    if n >= len(hits):
        return None
    h = hits[n]
    dest.write_bytes(urllib.request.urlopen(urllib.request.Request(h["largeImageURL"], headers=UA),
                                            timeout=120).read())
    return {"source": "Pixabay", "title": h["tags"], "author": h["user"], "page": h["pageURL"],
            "license": "Pixabay Content License"}


def commons_photo(query, dest, n=0):
    from pipeline import footage
    rows = [r for r in footage.commons(query, limit=30, media="image") if (r.get("width") or 0) >= 1200]
    if n >= len(rows):
        return None
    r = rows[n]
    footage.download(r["url"], dest, min_bytes=20_000)
    return {"source": "Wikimedia Commons", "title": r["title"], "author": r["author"], "page": r["page"],
            "license": r["license"]}


def cover(im, size=SIZE):
    s = max(size[0] / im.width, size[1] / im.height)
    im = im.resize((round(im.width * s), round(im.height * s)), Image.LANCZOS)
    left, top = (im.width - size[0]) // 2, (im.height - size[1]) // 2
    return im.crop((left, top, left + size[0], top + size[1]))


def vivid(im):
    """The niche's look: saturated, contrasty, bright."""
    im = ImageEnhance.Color(im).enhance(1.35)
    im = ImageEnhance.Contrast(im).enhance(1.18)
    return ImageEnhance.Brightness(im).enhance(1.06)


def subject_box(im):
    """Where the animal is (rembg IS-Net); its upper third is taken as the head."""
    try:
        from rembg import new_session, remove
        m = np.asarray(remove(im, session=new_session("isnet-general-use"), only_mask=True)) > 128
        ys, xs = np.nonzero(m)
        if len(xs) > 500:
            x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
            return (x0, y0, x1, y1), (x0, y0, x1, y0 + (y1 - y0) // 3)
    except Exception as e:                                           # noqa: BLE001
        log.warning("subject mask failed: %s", e)
    w, h = im.size
    return (w // 4, h // 6, w * 3 // 4, h * 5 // 6), (w // 4, h // 6, w * 3 // 4, h // 2)


def poly(draw, pts, box, fill):
    x0, y0, x1, y1 = box
    draw.polygon([(x0 + (x1 - x0) * x, y0 + (y1 - y0) * y) for x, y in pts], fill=fill)


def hook_scale(im, body):
    d = ImageDraw.Draw(im)
    x0, y0, x1, y1 = body
    cx = x0 - 70 if x0 > 160 else x1 + 70
    cy = (y0 + y1) // 2
    bw, bh = 34, 50
    poly(d, DIVER, (cx - bw // 2, cy - bh // 2, cx + bw // 2, cy + bh // 2), (20, 20, 25))
    for r, wdt in ((52, 7), (55, 3)):
        d.ellipse([cx - r, cy - r, cx + r, cy + r], outline=YELLOW, width=wdt)
    return im


def hook_silhouette(im, body):
    sil = Image.new("RGBA", im.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(sil)
    w, h = im.size
    left = body[0] > w / 2                      # behind, on the side away from the animal
    box = (int(w * 0.02), int(h * 0.05), int(w * 0.62), int(h * 0.98)) if left else \
          (int(w * 0.38), int(h * 0.05), int(w * 0.98), int(h * 0.98))
    pts = TREX if not left else [(1 - x, y) for x, y in TREX]
    poly(d, pts, box, (10, 8, 8, 215))
    glow = sil.filter(ImageFilter.GaussianBlur(18))
    red = Image.new("RGBA", im.size, (200, 30, 20, 0))
    red.putalpha(glow.getchannel("A").point(lambda a: int(a * 0.5)))
    base = Image.alpha_composite(im.convert("RGBA"), red)
    return Image.alpha_composite(base, sil).convert("RGB")


def hook_split(im, fossil):
    w, h = im.size
    out = im.copy()
    if fossil is not None:
        f = cover(fossil, (w // 2, h))
        out.paste(f, (0, 0))
    d = ImageDraw.Draw(out)
    d.rectangle([w // 2 - 4, 0, w // 2 + 4, h], fill=WHITE)
    d.polygon([(w // 2 - 70, h * 0.50), (w // 2 + 40, h * 0.44), (w // 2 + 40, h * 0.40), (w // 2 + 90, h * 0.50),
               (w // 2 + 40, h * 0.60), (w // 2 + 40, h * 0.56), (w // 2 - 70, h * 0.56)], fill=RED)
    return out


def compose(base, words, accent, side, out, head):
    """Words on `side`, kept off the head and out of the bottom-right corner."""
    rep = doc_thumb.best(base, words, side, out, accent_line=accent, box_frac=0.44)
    rep["head_box"] = [int(v) for v in head]
    return rep


def make_variant(t, work, out_dir, raw):
    """One variant: base (photo or FLUX), species check, hook, words. Returns a report."""
    tried = []
    base_img, src = None, None
    if t.get("base") == "file":
        base_img, src = Image.open(t["file"]).convert("RGB"), {"source": "FLUX.1-schnell", "title": Path(t["file"]).stem}
    elif t.get("base") == "flux":
        for s in (0, 1):
            p = raw / f"{t['id']}__s{s}.jpg"
            if not p.exists():
                continue
            chk = speciescheck.check_thumbnail(p, t.get("animal"), t["words"])
            tried.append({"base": p.name, "check": chk})
            if chk.get("pass", True):
                base_img, src = Image.open(p).convert("RGB"), {"source": "FLUX.1-schnell", "title": t["id"]}
                break
    else:
        for n in range(4):
            dest = work / f"{t['id']}_{n}.jpg"
            try:
                src = pixabay_photo(t["query"], dest, n) or commons_photo(t["query"], dest, n)
            except Exception as e:                                       # noqa: BLE001
                tried.append({"query": t["query"], "error": str(e)[:160]})
                src = None
            if not src:
                break
            chk = speciescheck.check_thumbnail(dest, t.get("animal"), t["words"])
            tried.append({"base": src, "check": chk})
            if chk.get("pass", True):
                base_img = Image.open(dest).convert("RGB")
                break
    if base_img is None:
        return {"id": t["id"], "ok": False, "why": "no base passed", "tried": tried}
    im = vivid(cover(base_img))
    body, head = subject_box(im)
    if t["hook"] == "scale":
        im = hook_scale(im, body)
    elif t["hook"] == "silhouette":
        im = hook_silhouette(im, body)
    elif t["hook"] == "split":
        fossil = None
        try:
            fp = work / f"{t['id']}_fossil.jpg"
            if commons_photo(t.get("fossil_query") or "fossil", fp):
                fossil = vivid(Image.open(fp).convert("RGB"))
        except Exception as e:                                           # noqa: BLE001
            log.warning("fossil photo: %s", e)
        im = hook_split(im, fossil)
    base_path = work / f"{t['id']}_base.jpg"
    im.save(base_path, quality=95)
    # the words go where the head is not
    cx = (head[0] + head[2]) / 2
    side = "left" if cx > SIZE[0] / 2 else "right"
    out = out_dir / f"thumbnail_{t['id'].split('_', 1)[1]}.jpg"
    rep = compose(base_path, t["words"], t.get("accent", 1), side, out, head)
    rep.update({"id": t["id"], "hook": t["hook"], "words": t["words"], "side": side, "source": src,
                "tried": tried, "ok": rep["readable"]})
    return rep


JUDGE = """Du bist der strenge Bildredakteur eines deutschen Urzeit-Kanals. Prüfe dieses YouTube-Vorschaubild
(Titel der Folge: "{title}") in der kleinen Ansicht. Vorbild sind die meistgesehenen Vorschaubilder der Nische:
EIN Tier riesig im Bild (Kopf, Auge, offenes Maul - es füllt die Hälfte des Bildes oder mehr), sofort als
urzeitliches Tier oder lebendes Fossil zu erkennen, knallige gesättigte Farben (Türkis/Tiefblau oder warmes
Orange gegen dunklen Grund), starker Kontrast, oft ein Größenvergleich (winziger Taucher, Boot, gelber Kreis)
oder ein roter Pfeil/Kreis, und 2-3 große Wörter, in einer Sekunde lesbar.
VERBOTEN (dann sofort "pass": false): DNA, Moleküle, Diagramme, Schemata, Karten, abstrakte Muster,
Texturen, eine Landschaft/ein Meer ohne großes Tier, ein Tier winzig oder angeschnitten bis zur
Unkenntlichkeit, Menschen im Vordergrund, mehr als drei Wörter.
Antworte als JSON: {{"animal": "<welches Tier>", "animal_big_and_recognisable": true|false,
"prehistoric_or_living_fossil": true|false, "bright": true|false, "words_readable": true|false,
"forbidden_seen": ["..."], "pass": true|false, "why": "<kurz>"}}"""


def judge(path, title):
    """The Gemini judge: True/False and why. Without a key nothing passes (the QA needs a judged one)."""
    try:
        from core import gemini
        a = gemini.ask(JUDGE.format(title=title), images=[str(path)])
    except Exception as e:                                               # noqa: BLE001
        return False, f"judge unavailable: {str(e)[:120]}"
    ok = bool(a.get("pass")) and a.get("animal_big_and_recognisable") and a.get("prehistoric_or_living_fossil") \
        and a.get("words_readable") and not a.get("forbidden_seen")
    return bool(ok), a


def extra_variants(plan, raw, n_have):
    """More bases when no thumbnail passed: the episode's own FLUX animals (no landscapes), each
    with the next trick and the words of the planned variants."""
    tricks = ["scale", "silhouette", "split"]
    out = []
    frames = [f for f in sorted(raw.parent.glob("*.jpg")) if "__s" in f.name and not f.name.startswith("thumb_")
              and not any(w in f.name for w in ("landscape", "ocean", "sea_", "extinction", "impact"))]
    for i, f in enumerate(frames[:6]):
        t = dict(plan[i % len(plan)])
        t.update({"id": f"thumb_x{n_have + i}", "hook": tricks[i % 3], "base": "file", "file": str(f)})
        out.append(t)
    return out


CHOOSE = """Drei Vorschaubilder für dieselbe deutsche Urzeit-Doku (Titel: "{title}"). Welches würde in der
kleinen Ansicht (336x189) die meisten Klicks bekommen und ist trotzdem ehrlich? Kriterien: knallig, ein Tier
groß und sofort als urzeitlich erkennbar, der Bild-Trick funktioniert, Text in 1 Sekunde lesbar.
Antworte als JSON {{"best": <1|2|3>, "why": "<kurz>"}}"""


def main(slug):
    ep = ROOT / "episodes" / slug
    plan = json.loads((ep / "thumbs.json").read_text(encoding="utf-8"))
    meta = json.loads((ep / "meta.json").read_text(encoding="utf-8"))
    work = ROOT / "work" / slug / "thumbs"
    work.mkdir(parents=True, exist_ok=True)
    out_dir = ROOT / "work" / slug / "release"
    out_dir.mkdir(parents=True, exist_ok=True)
    raw = ROOT / "work" / slug / "inputs" / "thumb_bases"
    reps = [make_variant(t, work, out_dir, raw) for t in plan]
    for r in reps:
        if r.get("ok"):
            r["judge_pass"], r["judge"] = judge(out_dir / f"thumbnail_{r['id'].split('_', 1)[1]}_336.jpg",
                                                meta["title"])
    good = [r for r in reps if r.get("ok") and r.get("judge_pass")]
    if not good:                       # none passed the judge: new ones from the episode's own animals
        more = [make_variant(t, work, out_dir, raw / "x") for t in
                extra_variants(plan, ROOT / "gpu" / "results_recon" / "raw" / "x", len(reps))]
        for r in more:
            if r.get("ok"):
                r["judge_pass"], r["judge"] = judge(out_dir / f"thumbnail_{r['id'].split('_', 1)[1]}_336.jpg",
                                                    meta["title"])
        reps += more
        good = [r for r in reps if r.get("ok") and r.get("judge_pass")][:3]
    choice = {"best": None, "judge_pass": False}
    if good:
        files = [out_dir / f"thumbnail_{r['id'].split('_', 1)[1]}.jpg" for r in good]
        try:
            from core import gemini
            ans = gemini.ask(CHOOSE.format(title=meta["title"]), images=[str(f).replace(".jpg", "_336.jpg")
                                                                       for f in files])
            k = max(1, min(len(files), int(ans.get("best", 1))))
            choice = {"best": files[k - 1].name, "why": ans.get("why"), "judge_pass": True,
                      "judge": good[k - 1].get("judge")}
        except Exception as e:                                           # noqa: BLE001
            choice = {"best": files[0].name, "why": f"chooser unavailable ({e}); first judged",
                      "judge_pass": True, "judge": good[0].get("judge")}
        import shutil
        shutil.copy(out_dir / choice["best"], out_dir / "thumbnail.jpg")
    (out_dir / "thumbnails_report.json").write_text(json.dumps({"variants": reps, "choice": choice}, indent=1,
                                                               ensure_ascii=False, default=str), encoding="utf-8")
    log.info("thumbnails: %s, chosen %s", [(r["id"], r.get("ok"), r.get("judge_pass"),
                                             (r.get("judge") or {}).get("why") if isinstance(r.get("judge"), dict)
                                             else r.get("judge") or r.get("why")) for r in reps], choice)
    return 0 if good else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
