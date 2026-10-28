"""Three German thumbnails from real photographs, into test/.

    python automation/make_thumbs.py living-fossils

The living animals of this channel are never drawn by FLUX, thumbnails
included: each base is a real photograph (Pixabay, or Wikimedia Commons under
PD / CC0 / CC BY). The look is the one the thumb_croc test set: one animal,
close on its head or eye, dark ground, light from the side; the photo is graded
toward it (darker surround, vignette, contrast) and the words are set in code -
two or three German words, white with the key word in red (pipeline/doc_thumb.py).

Every base goes through the species check first (pipeline/speciescheck.py:
the right animal, and the style); a base that fails gives way to the next
candidate. Readability is measured at 336x189.
"""
import json
import os
import sys
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from pipeline import doc_thumb, speciescheck  # noqa: E402

UA = {"User-Agent": "prehistoric-channel/0.1 (thumbnails)"}
PIXABAY = "Pixabay Content License"


def px(pid, title):
    return {"source": "Pixabay", "pixabay_id": pid, "title": title, "license": PIXABAY,
            "license_url": "https://pixabay.com/service/license-summary/", "page": f"https://pixabay.com/photos/id-{pid}/"}


def commons(url, page, title, author, lic):
    return {"source": "Wikimedia Commons", "url": url, "page": page, "title": title, "author": author, "license": lic}


VARIANTS = [
    {"species": "crocodile", "words": ["Sie", "überlebten"], "box": 0.53, "vignette": 0.55, "candidates": [
        px(2814446, "crocodile head on black"), px(7428462, "crocodile in dark water"),
        px(6776169, "alligator eye")]},
    {"species": "latimeria", "words": ["Lebendes", "Fossil"], "box": 0.46, "vignette": 0.9, "candidates": [
        commons("https://upload.wikimedia.org/wikipedia/commons/c/c0/"
                "Latimeria_chalumnae_Mus%C3%A9e_d%27Histoire_Naturelle_Tournai_27122015_1.jpg",
                "https://commons.wikimedia.org/wiki/File:Latimeria_chalumnae_Mus%C3%A9e_d%27Histoire_Naturelle_"
                "Tournai_27122015_1.jpg", "Latimeria chalumnae, Musée d'Histoire Naturelle Tournai", "Vassil", "CC0"),
        commons("https://upload.wikimedia.org/wikipedia/commons/9/93/"
                "Latimeria_chalumnae_%28model%29_%28coelacanth%29_1_%2815695435636%29.jpg",
                "https://commons.wikimedia.org/wiki/File:Latimeria_chalumnae_(model)_(coelacanth)_1_(15695435636).jpg",
                "Latimeria chalumnae (model)", "James St. John", "CC BY 2.0")]},
    {"species": "tuatara", "words": ["Älter als", "Dinosaurier"], "box": 0.53, "vignette": 0.55, "candidates": [
        px(7596468, "tuatara"), px(2190866, "tuatara"),
        commons("https://upload.wikimedia.org/wikipedia/commons/1/1a/Sphenodon_punctatus_head.jpg",
                "https://commons.wikimedia.org/wiki/File:Sphenodon_punctatus_head.jpg",
                "Sphenodon punctatus head", "TimVickers", "Public domain")]},
]


def fetch(cand, dest):
    if cand["source"] == "Pixabay":
        key = os.environ.get("PIXABAY_API_KEY", "").strip()
        if not key:
            raise RuntimeError("PIXABAY_API_KEY is not set")
        q = urllib.parse.urlencode({"key": key, "id": cand["pixabay_id"]})
        hit = json.load(urllib.request.urlopen(urllib.request.Request(f"https://pixabay.com/api/?{q}", headers=UA),
                                               timeout=60))["hits"][0]
        url = hit.get("largeImageURL") or hit["webformatURL"]
        cand.update({"author": hit["user"], "page": hit["pageURL"]})
    else:
        url = cand["url"]
    dest.write_bytes(urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=120).read())
    return dest


def grade(src, dest, vignette=0.55):
    """Toward the thumb_croc look: darker surround, a vignette, firmer contrast."""
    im = Image.open(src).convert("RGB")
    im.thumbnail((2560, 2560))
    a = np.asarray(im, np.float32) / 255
    h, w = a.shape[:2]
    y, x = np.mgrid[0:h, 0:w]
    r = np.sqrt(((x - w / 2) / (w / 2)) ** 2 + ((y - h / 2) / (h / 2)) ** 2)
    a = a * (1 - vignette * np.clip(r - 0.35, 0, 1) ** 1.4)[..., None]
    # bright backdrops (a museum wall) roll off toward the dark ground of the style
    lum = a.mean(axis=2, keepdims=True)
    a = a * (1 - vignette * 0.6 * np.clip((lum - 0.55) / 0.35, 0, 1) * np.clip(r - 0.2, 0, 1)[..., None])
    a = np.clip((a - 0.5) * 1.15 + 0.5 - 0.06, 0, 1) ** 1.12
    Image.fromarray((a * 255).astype(np.uint8)).save(dest, quality=95)
    return dest


def text_side(path):
    """The words go on the darker half."""
    g = np.asarray(Image.open(path).convert("L"), np.float32)
    return "left" if g[:, : g.shape[1] // 2].mean() < g[:, g.shape[1] // 2:].mean() else "right"


def main(slug):
    out = ROOT / "test"
    out.mkdir(exist_ok=True)
    work = ROOT / "work" / slug / "thumbs"
    work.mkdir(parents=True, exist_ok=True)
    reps = []
    for i, v in enumerate(VARIANTS, 1):
        chosen, tried = None, []
        for k, cand in enumerate(v["candidates"]):
            try:
                raw = fetch(cand, work / f"t{i}_{k}_raw.jpg")
            except Exception as e:                                   # noqa: BLE001
                tried.append({"source": cand, "error": str(e)[:160]})
                continue
            base = grade(raw, work / f"t{i}_{k}.jpg", v.get("vignette", 0.55))
            chk = speciescheck.check_thumbnail(base, v["species"], v["words"])
            tried.append({"source": {kk: vv for kk, vv in cand.items()}, "check": chk})
            if chk.get("pass", True):
                chosen = (cand, base)
                break
        if chosen is None:
            reps.append({"variant": i, "words": v["words"], "readable": False, "error": "no base passed", "tried": tried})
            continue
        cand, base = chosen
        side = text_side(base)
        rep = doc_thumb.best(base, v["words"], side, out / f"thumbnail_{i}.jpg", accent_line=1,
                             box_frac=v.get("box", 0.46))
        rep.update({"variant": i, "words": v["words"], "species": v["species"], "side": side,
                    "source": {kk: vv for kk, vv in cand.items() if kk != "pixabay_id"}, "tried": tried})
        reps.append(rep)
        print({k: rep[k] for k in ("variant", "words", "readable", "side")})
    (out / "thumbnails_report.json").write_text(json.dumps(reps, indent=1, ensure_ascii=False), encoding="utf-8")
    return 0 if all(r.get("readable") for r in reps) else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
