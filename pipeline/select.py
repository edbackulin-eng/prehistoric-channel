"""Choose the real footage: look at every candidate before it is allowed in.

For each candidate three frames are pulled from its preview (10 %, 50 %, 90 %)
and shown to Gemini, which answers per clip: what the subject actually is,
whether it matches the scene it was searched for, and whether a person, a
logo, a watermark or on-screen text appears anywhere. Tags on stock sites are
written by uploaders and are often wrong; the frames are not.

Perceptual hashes of the same frames catch one clip uploaded twice under two
ids, and stop two scenes from reusing the same shot.
"""
import base64
import io
import json
import subprocess
import time
import urllib.request
from pathlib import Path

import imagehash
from PIL import Image

from core import llm
from core.logger import get_logger

log = get_logger("select")

MODEL = "gemini-flash-latest"
FRAMES = (0.1, 0.5, 0.9)

SYSTEM = """You review candidate stock clips for a natural-history documentary in the
style of the BBC. For every clip you get three frames from its start, middle and end.
Be strict: a clip with any visible person, body part, diver, boat crew, logo,
watermark, caption, subtitle, timestamp or website address is rejected outright,
however small. Aquarium glass, tanks, zoo enclosures, fences or human structures
also count against a clip (say so in "setting"). Reply with valid JSON only."""

PROMPT = """The scene these clips were searched for: "{need}" - {brief}

For each clip, in order, return
{{"results":[{{"clip":<index>,"subject":"<what is really shown, a few words>",
"match":0-5, "people":true|false, "branding":true|false, "text":true|false,
"setting":"wild|aquarium|zoo|studio|other", "quality":0-5,
"note":"<few words>"}}]}}
match: 5 = exactly the scene asked for, 3 = usable stand-in, 0 = unrelated.
quality: sharpness, exposure, stability, cinematic look."""


def frames(src, duration, out_dir, stem):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for k, f in enumerate(FRAMES):
        p = out_dir / f"{stem}_{k}.jpg"
        for _ in range(3):
            if p.exists():
                break
            t = max(0.0, (duration or 10) * f)
            try:
                ua = ["-user_agent", "Mozilla/5.0"] if str(src).startswith("http") else []
                subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                                *ua, "-ss", f"{t:.2f}", "-i", str(src),
                                "-frames:v", "1", "-vf", "scale=640:-2", "-q:v", "4", str(p)],
                               capture_output=True, timeout=240)
            except subprocess.TimeoutExpired:
                pass
        if p.exists():
            paths.append(p)
    return paths


def phash(paths):
    return [str(imagehash.phash(Image.open(p))) for p in paths]


def similar(h1, h2, limit=10):
    return any(imagehash.hex_to_hash(a) - imagehash.hex_to_hash(b) <= limit for a in h1 for b in h2)


def _b64(path):
    im = Image.open(path).convert("RGB")
    im.thumbnail((640, 640))
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=82)
    return base64.b64encode(buf.getvalue()).decode()


def judge(clips, need, brief):
    """clips: [{frames: [paths]}]. Returns the verdict list, one per clip."""
    key = llm.get_key(llm.GEMINI_KEY_NAME)
    parts = [{"text": PROMPT.format(need=need, brief=brief)}]
    for i, c in enumerate(clips):
        parts.append({"text": f"clip {i}:"})
        for p in c["frames"]:
            parts.append({"inline_data": {"mime_type": "image/jpeg", "data": _b64(p)}})
    payload = {"contents": [{"role": "user", "parts": parts}],
               "systemInstruction": {"parts": [{"text": SYSTEM}]},
               "generationConfig": {"temperature": 0.0, "maxOutputTokens": 4096,
                                    "responseMimeType": "application/json",
                                    "thinkingConfig": {"thinkingBudget": 0}}}
    url = llm.GEMINI_URL.format(model=MODEL)
    for attempt in range(8):
        try:
            req = urllib.request.Request(url, data=json.dumps(payload).encode(), method="POST",
                                         headers={"Content-Type": "application/json",
                                                  "x-goog-api-key": key})
            with urllib.request.urlopen(req, timeout=240) as r:
                data = json.loads(r.read().decode())
            text = "".join(p.get("text", "") for p in data["candidates"][0]["content"]["parts"])
            return llm.extract_json(text).get("results", [])
        except Exception as e:                                   # noqa: BLE001
            log.info("gemini retry %d: %s", attempt + 1, e)
            time.sleep(15 * (attempt + 1))
    raise RuntimeError("gemini judging failed")


def accepted(v, min_match=3, min_quality=3, allow=()):
    """`allow` relaxes a rule for a beat whose subject needs it: old books carry
    print ("text"), a fossil is photographed on a table ("studio")."""
    settings = ("wild", "other") + (("studio",) if "studio" in allow else ())
    return (not v.get("people") and not v.get("branding")
            and (not v.get("text") or "text" in allow)
            and v.get("setting") in settings and v.get("match", 0) >= min_match
            and v.get("quality", 0) >= min_quality)
