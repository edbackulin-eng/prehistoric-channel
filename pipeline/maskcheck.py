"""Is the 2.5D foreground really the animal? A last look before the render.

The kernel already refuses masks that are the wrong size, in pieces, or that
its two segmentation models disagree about (gpu/kernels/recon_job.py). What it
cannot judge is meaning: a clean, confident mask of the rock beside the animal
passes every geometric check. So here each accepted mask is shown to Gemini
(flash-lite) as an overlay - the masked region in colour, everything else dimmed
- and a mask that is not the whole animal, or carries background with it, is
set aside. Its shot then gets a plain Ken Burns move instead of parallax
(pipeline/parallax.py treats a missing alpha that way).

Needs this channel's own key in GEMINI_API_KEY_PH. Without it the check is
skipped and says so: another channel's quota is never borrowed (18.10 that
exhausted What If's daily allowance).
"""
import base64
import io
import json
import os
import time
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image

MODEL = "gemini-flash-lite-latest"
URL = "https://generativelanguage.googleapis.com/v1beta/models/{m}:generateContent"
PROMPT = """The image is a photograph or illustration in which one region is shown in
full colour with a red tint, and everything else is dimmed. The red-tinted region is
meant to be exactly the main animal (living or fossil) - the whole animal, including
tail, limbs and fins - and nothing else.

Answer with JSON only:
{"animal": true|false,            // is the tinted region an animal (or a fossil animal)?
 "whole_animal": true|false,      // does it cover the whole animal, tail and limbs included?
 "background_included": true|false, // does it also cover ground, rock, water or sky?
 "note": "<a few words>"}"""


def _overlay(img_path, alpha_path, side=768):
    im = Image.open(img_path).convert("RGB")
    a = Image.open(alpha_path).convert("L").resize(im.size)
    im.thumbnail((side, side))
    a = np.asarray(a.resize(im.size), np.float32)[..., None] / 255
    x = np.asarray(im, np.float32)
    out = x * (0.25 + 0.75 * a) + np.array([255, 30, 30], np.float32) * 0.30 * a
    buf = io.BytesIO()
    Image.fromarray(out.clip(0, 255).astype(np.uint8)).save(buf, "JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode()


def _ask(key, b64):
    payload = {"contents": [{"parts": [{"text": PROMPT},
                                       {"inline_data": {"mime_type": "image/jpeg", "data": b64}}]}],
               "generationConfig": {"temperature": 0, "responseMimeType": "application/json"}}
    for attempt in range(4):
        try:
            req = urllib.request.Request(URL.format(m=MODEL), data=json.dumps(payload).encode(),
                                         headers={"Content-Type": "application/json", "x-goog-api-key": key})
            with urllib.request.urlopen(req, timeout=60) as r:
                d = json.loads(r.read())
            text = "".join(p.get("text", "") for p in d["candidates"][0]["content"]["parts"])
            return json.loads(text[text.find("{"): text.rfind("}") + 1])
        except Exception as e:                                   # noqa: BLE001
            if "429" in str(e) and attempt < 3:
                time.sleep(20 * (attempt + 1))
                continue
            return {"error": f"{type(e).__name__}: {str(e)[:160]}"}


def check(layers_dir, report_path=None):
    """Look at every foreground mask in layers_dir; set aside the ones that fail."""
    layers_dir = Path(layers_dir)
    key = os.environ.get("GEMINI_API_KEY_PH", "").strip()
    rep = {"model": MODEL, "masks": {}}
    alphas = sorted(layers_dir.glob("*_alpha.png"))
    if not key:
        rep["skipped"] = "GEMINI_API_KEY_PH is not set; geometric and two-model checks only"
    for ap in alphas:
        name = ap.name[: -len("_alpha.png")]
        img = layers_dir / f"{name}.jpg"
        if not key or not img.exists():
            rep["masks"][name] = {"verdict": "not checked"}
            continue
        v = _ask(key, _overlay(img, ap))
        keep = bool(v.get("animal")) and bool(v.get("whole_animal")) and not v.get("background_included")
        if "error" in v:
            keep = True                       # an unreachable checker does not throw work away
        v["kept"] = keep
        rep["masks"][name] = v
        if not keep:
            for suffix in ("_alpha.png", "_bg.jpg"):
                f = layers_dir / f"{name}{suffix}"
                if f.exists():
                    f.rename(f.with_name(f.name + ".rejected"))
        time.sleep(4)
    if report_path:
        Path(report_path).write_text(json.dumps(rep, indent=1), encoding="utf-8")
    return rep
