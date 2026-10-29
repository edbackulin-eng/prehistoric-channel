"""Gemini for this channel: its own key only (GEMINI_API_KEY_PH), JSON answers.

Another channel's key is never used (18.10 that exhausted What If's quota). A
model out of quota is set aside for the rest of the run and the next one is
tried, rather than waiting out 429s (28.10: that ran a job out of time).
"""
import base64
import io
import json
import os
import time
import urllib.error
import urllib.request

from core.logger import get_logger

log = get_logger("gemini")

URL = "https://generativelanguage.googleapis.com/v1beta/models/{m}:generateContent"
TEXT_MODELS = ("gemini-flash-latest", "gemini-flash-lite-latest")
_SPENT = set()


def key():
    k = os.environ.get("GEMINI_API_KEY_PH", "").strip()
    if not k:
        raise RuntimeError("GEMINI_API_KEY_PH is not set")
    return k


def image_part(path, side=768):
    from PIL import Image
    im = Image.open(path).convert("RGB")
    im.thumbnail((side, side))
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=85)
    return {"inline_data": {"mime_type": "image/jpeg", "data": base64.b64encode(buf.getvalue()).decode()}}


def ask(prompt, images=(), models=TEXT_MODELS, json_out=True, temperature=0.4, max_tokens=None, timeout=240):
    """The model's answer: parsed JSON when json_out, else text."""
    parts = [{"text": prompt}] + [image_part(p) for p in images]
    cfg = {"temperature": temperature}
    if json_out:
        cfg["responseMimeType"] = "application/json"
    if max_tokens:
        cfg["maxOutputTokens"] = max_tokens
    payload = json.dumps({"contents": [{"parts": parts}], "generationConfig": cfg}).encode()
    last = None
    for m in models:
        if m in _SPENT:
            continue
        for attempt in range(5):
            try:
                req = urllib.request.Request(URL.format(m=m), data=payload,
                                             headers={"Content-Type": "application/json", "x-goog-api-key": key()})
                with urllib.request.urlopen(req, timeout=timeout) as r:
                    d = json.loads(r.read())
                text = "".join(p.get("text", "") for p in d["candidates"][0]["content"]["parts"])
                log.info("gemini %s: %d chars", m, len(text))
                if not json_out:
                    return text
                return json.loads(text[min(i for i in (text.find("{"), text.find("[")) if i >= 0):])
            except urllib.error.HTTPError as e:
                body = e.read()[:600].decode("utf-8", "replace")
                last = f"{m}: HTTP {e.code} {body[:200]}"
                # a per-minute limit is waited out; only a spent day (or a missing
                # model) sends the call to the next, lighter model
                if e.code == 429 and "PerDay" not in body and attempt < 2:
                    time.sleep(35 * (attempt + 1))
                    continue
                if e.code in (429, 404):
                    _SPENT.add(m)
                    log.warning("%s set aside for this run: %s", m, body[:160])
                    break
                # 5xx ("high demand", 29.10 a 503) passes in a minute or two: wait it out
                time.sleep(20 * (attempt + 1) if e.code >= 500 else 5 * (attempt + 1))
            except Exception as e:                                   # noqa: BLE001
                last = f"{m}: {type(e).__name__}: {str(e)[:200]}"
                time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"Gemini gave no answer: {last}")
