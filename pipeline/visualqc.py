"""Look at every frame before it reaches the render, with Gemini vision.

The technical QC measures duration, loudness and black frames. It cannot see
that a frame contains garbled AI text, a face staring down the lens, or a card
reading "?" - and all three shipped in the first cut of episode one precisely
because nothing was looking.

A frame that fails is regenerated once. If it fails again it is replaced rather
than argued with, because a second bad generation usually means the prompt is
the problem and the pipeline should not stall on it.

Runs on its own model so it cannot exhaust the fact-checker's daily quota.
"""
import base64
import json
import time
import urllib.error
import urllib.request
from pathlib import Path

from core import llm
from core.logger import get_logger

log = get_logger("visualqc")

# A third quota bucket, separate from the fact-check and the scene planner.
VISION_MODEL = "gemini-flash-lite-latest"
BATCH = 16                # frames per request: the free tier allows ~20 requests a day per model, and an episode has ~140 stills
# A 223-frame video is ~38 requests. Fired back to back, a third of them came
# back 429 and those frames went unchecked - which defeats the point of the
# check. Pacing them and retrying on rejection costs a couple of minutes.
MIN_INTERVAL_S = 2.0
RETRIES = 5
# 512 was too small to see garbled lettering on a phone screen - the defect
# that prompted this whole check went undetected at that size.
MAX_SIDE = 768

# Photographic frames and drawn cards fail in opposite directions: a photo must
# contain no lettering at all, while a card is nothing but lettering. Judging
# both with one rulebook flagged every clean card as a "text" defect.
PHOTO_SYSTEM = """You inspect generated photographic frames for a documentary channel about
internet mysteries. Many frames are dark rooms lit by screens; darkness is the look.

Report a defect ONLY when you can clearly see one of these:
- text: lettering you can READ or that is crisp enough to look like real words:
  captions, signage, user interface, a screen displaying characters, garbled fake
  words in focus. Garbled crisp lettering is the most important defect to catch.
  Neon signs and shop signs with letter shapes count, even in the background,
  unless they are blurred into shapeless bokeh ("PITINE PANS" on a sign is a defect).
  So do dashboard or device displays showing numbers.
  NOT a defect: old books, letters, maps and papers whose writing is tiny, out of
  focus or worn to an unreadable texture - that is asked for.
- watermark: a watermark, logo, brand mark, signature, or a recognisable real website,
  app or product.
- face: a person facing the camera with their features LIT AND VISIBLE - you can make
  out eyes, nose and mouth and read an expression - or anyone looking into the lens.
- anatomy: malformed hands, wrong number of fingers, extra or missing limbs, distorted
  bodies or faces.
- graphic: blood, injury, weapons, bodies, anything violent; or a child.
- border: a black or dark band along an edge that is not part of the scene
  (letterboxing, a frame inside the frame).

FINE, and never a defect: anyone seen from behind or turned away; a figure
backlit or silhouetted so the face reads as a dark shape, even in profile; a
face too distant or too dark to make out features. A screen glowing with NO
characters on it is FINE. Grain, darkness and blur are FINE.

The test is whether a viewer could describe the person's expression. If not, it
is a silhouette and it passes. Two calibration points, both real: a lit group of
three facing the lens IS a defect and was wrongly passed; a dark profile against
mist is NOT a defect and was wrongly flagged.
Reply with valid JSON only."""

ARCHIVE_SYSTEM = """You inspect real archive photographs from NASA and similar
agencies, used in a science documentary.

These are genuine photographs. Lettering that belongs to the real scene is FINE
and must NOT be reported: mission patches, "NASA" on hardware, "Canada" on the
ISS arm, instrument labels, map place names, colour-scale legends, timestamps
and frame identifiers burned in by the agency.

Report a defect ONLY when you see:
- branding: a watermark, logo or caption belonging to someone who is not the
  space agency - a broadcaster's bug, a stock-footage watermark, a documentary
  lower-third, a production company name. This is the one that matters: it is
  somebody else's trade mark in our frame.

Everything else about an archive photograph is acceptable.
Reply with valid JSON only."""

STOCK_SYSTEM = """You inspect frames from licensed stock footage and archive photographs
used in a documentary about internet mysteries.

Report a defect ONLY when you clearly see:
- watermark: a stock watermark, a broadcaster bug, a channel logo or a caption.
- logo: a recognisable real brand, company logo or product name readable in frame
  (a shop sign in the far distance that cannot be read is FINE).
- face: a person looking into the lens, or a face lit and large enough to identify.
  People seen from behind, in profile at a distance, blurred or small are FINE.
- graphic: blood, injury, weapons, police scenes, a child as the subject.
- text: large readable on-screen text or a screen showing readable content.
Everything else is acceptable.
Reply with valid JSON only."""

CARD_SYSTEM = """You inspect drawn infographic cards from a science documentary.

These cards are SUPPOSED to contain large text and numbers. Do not report that
as a defect.

Report a defect ONLY when:
- card: the card shows "?" as its main figure, a word is cut off mid-way
  (for example "8 billion people f" or "Inverse Cube (1/r^"), text runs past the
  edge of the frame, or the card is effectively empty.

Correct spelling, complete words and numbers are FINE however large.
Reply with valid JSON only."""

PROMPT = """For each image, in order, return:
{"results":[{"i":<index>,"ok":true|false,
"defect":"text|watermark|face|anatomy|graphic|border|card|logo|none",
"note":"<a few words, only when ok is false>",
"detail":{"name":"<the object>","box":[x0, y0, x1, y1]} or null}]}
"detail" is ONE small, concrete, clearly visible physical object in the frame that a
red circle could be drawn around - a phone, a lamp, a door, a cable, a book, a sign
post. "box" is its tight bounding box as fractions 0-1 of the image (x0,y0 top-left,
x1,y1 bottom-right). If there is no single distinct object - an empty wall, sky,
fog, a featureless area - return null. Never box an empty area."""


def _encode(path, max_side=MAX_SIDE):
    from PIL import Image
    import io
    with Image.open(path) as im:
        im = im.convert("RGB")
        scale = min(1.0, max_side / max(im.size))
        if scale < 1.0:
            im = im.resize((max(1, round(im.width * scale)),
                            max(1, round(im.height * scale))), Image.LANCZOS)
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=82)
    return base64.b64encode(buf.getvalue()).decode("ascii")


_last_call = [0.0]


def _throttle():
    wait = MIN_INTERVAL_S - (time.monotonic() - _last_call[0])
    if wait > 0:
        time.sleep(wait)
    _last_call[0] = time.monotonic()


def _inspect_batch(paths, model=VISION_MODEL, kind="photo"):
    key = llm.get_key(llm.GEMINI_KEY_NAME)
    if not key:
        raise llm.LLMError(f"{llm.GEMINI_KEY_NAME} is not set")
    parts = [{"text": PROMPT}]
    for i, p in enumerate(paths):
        parts.append({"text": f"image {i}:"})
        parts.append({"inline_data": {"mime_type": "image/jpeg",
                                      "data": _encode(p)}})
    payload = {
        "contents": [{"role": "user", "parts": parts}],
        "systemInstruction": {"parts": [
            {"text": {"card": CARD_SYSTEM, "stock": STOCK_SYSTEM,
                      "archive": ARCHIVE_SYSTEM}.get(kind, PHOTO_SYSTEM)}]},
        "generationConfig": {"maxOutputTokens": 2048, "temperature": 0.0,
                             "responseMimeType": "application/json"},
    }
    url = llm.GEMINI_URL.format(model=model)
    last = None
    for attempt in range(RETRIES):
        _throttle()
        req = urllib.request.Request(
            url, data=json.dumps(payload).encode(), method="POST",
            headers={"Content-Type": "application/json", "x-goog-api-key": key})
        try:
            with urllib.request.urlopen(req, timeout=180) as resp:
                data = json.loads(resp.read().decode("utf-8", "replace"))
            break
        except urllib.error.HTTPError as e:
            last = e
            # 503 is the service being unavailable, not us doing anything wrong,
            # and it has been the common failure rather than rate limiting.
            # Retrying it is the difference between a frame being inspected and
            # a frame shipping unlooked-at.
            if e.code not in (429, 500, 502, 503, 504):
                raise
            body = e.read().decode("utf-8", "replace") if e.code == 429 else ""
            if "PerDay" in body:          # a daily cap will not clear today
                raise
            backoff = 4 * (attempt + 1)
            log.info("vision %s, waiting %ds", e.code, backoff)
            time.sleep(backoff)
    else:
        raise last
    cand = (data.get("candidates") or [{}])[0]
    text = "".join(p.get("text", "")
                   for p in cand.get("content", {}).get("parts", [])
                   if "text" in p)
    return llm.extract_json(text).get("results", [])


def inspect(paths, model=VISION_MODEL, kinds=None):
    """Returns {path: {"ok": bool, "defect": str, "note": str}}.

    `kinds` maps a path to "photo", "card" or "archive"; anything unlisted is a
    photo. Batches never mix kinds, because each has its own rulebook.

    Three rulebooks, not one, because the same pixel means different things by
    provenance. Lettering in a generated frame is always a defect; lettering in
    a NASA photograph is usually the label the agency put there; a card is
    nothing but lettering. Judging all three as "photo" flagged 36 frames of
    which 17 were perfectly good.
    """
    paths = [Path(p) for p in paths]
    kinds = {Path(k): v for k, v in (kinds or {}).items()}
    verdicts = {}
    groups = {"photo": [], "card": [], "archive": [], "stock": []}
    for p in paths:
        groups.setdefault(kinds.get(p, "photo"), []).append(p)

    done = 0
    # Each kind is batched separately: mixing them in one request would need two
    # rulebooks in one system prompt, and an earlier version that filtered a
    # mixed batch down to one kind silently dropped the rest.
    for kind, items in groups.items():
        for start in range(0, len(items), BATCH):
            batch = items[start:start + BATCH]
            try:
                results = _inspect_batch(batch, model, kind)
            except Exception as e:
                # Not "ok". A frame nobody looked at is unknown, not clean, and
                # recording it as clean is how a 503 turned into a shipped
                # defect. The caller decides what to do with unknowns.
                log.warning("vision batch (%s) failed (%s); %d frame(s) unchecked",
                            kind, e, len(batch))
                for p in batch:
                    verdicts[p] = {"ok": False, "defect": "unchecked",
                                   "note": str(e)[:80]}
                done += len(batch)
                continue
            by_i = {r.get("i"): r for r in results}
            for i, p in enumerate(batch):
                r = by_i.get(i)
                if r is None:
                    # The model skipped this image. Defaulting to ok here would be
                    # the same fail-open the 503 path was fixed for.
                    verdicts[p] = {"ok": False, "defect": "unchecked",
                                   "note": "no verdict returned for this image", "detail": None}
                    continue
                verdicts[p] = {"ok": bool(r.get("ok", False)),
                               "defect": (r.get("defect") or "none").lower(),
                               "note": r.get("note", "") or "",
                               "detail": real_detail(p, r.get("detail"))}
            done += len(batch)
            log.info("inspected %d/%d frames", done, len(paths))

    failed = [p for p, v in verdicts.items() if not v["ok"]]
    log.info("visual QC: %d frames, %d flagged", len(verdicts), len(failed))
    return verdicts


def real_detail(path, detail):
    """The detail Gemini boxed, if the pixels agree that something is there.

    The first hook drew red circles around empty patches: the model was asked for
    the most striking point and always answered with one. Now it names an object
    and boxes it, and the box is checked here: a sensible size, and texture or
    contrast inside it that stands out from the picture around it. A box on a
    flat wall or a patch of fog fails, and the shot gets no circle.
    Returns {"name", "box", "circle": [cx, cy, r]} or None.
    """
    import numpy as np
    from PIL import Image
    if not isinstance(detail, dict):
        return None
    box = detail.get("box")
    if not (isinstance(box, list) and len(box) == 4
            and all(isinstance(v, (int, float)) and 0 <= v <= 1 for v in box)):
        return None
    x0, y0, x1, y1 = box
    if x1 - x0 < 0.03 or y1 - y0 < 0.03:
        return None
    if (x1 - x0) * (y1 - y0) > 0.20:           # a fifth of the frame is not "a detail"
        return None
    with Image.open(path) as im:
        a = np.asarray(im.convert("L").resize((480, 270)), np.float32)
    h, w = a.shape
    X0, Y0 = int(x0 * w), int(y0 * h)
    X1, Y1 = max(int(x1 * w), X0 + 2), max(int(y1 * h), Y0 + 2)
    inner = a[Y0:Y1, X0:X1]
    pad = max(6, int(0.5 * max(X1 - X0, Y1 - Y0)))
    ring = a[max(0, Y0 - pad):min(h, Y1 + pad), max(0, X0 - pad):min(w, X1 + pad)]
    gy, gx = np.gradient(a)
    edges = np.hypot(gx, gy)
    e_in = float(edges[Y0:Y1, X0:X1].mean())
    e_all = float(edges.mean()) + 1e-3
    contrast = abs(float(inner.mean()) - float(ring.mean()))
    if inner.std() < 10 or (e_in < 1.15 * e_all and contrast < 12):
        return None
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    # radius in frame heights; the circle is drawn 1.25x wider than tall
    r = max((x1 - x0) * 16 / 9 / 2 / 1.25, (y1 - y0) / 2) * 1.25
    return {"name": str(detail.get("name", ""))[:40], "box": [round(v, 3) for v in box],
            "circle": [round(cx, 3), round(cy, 3), round(min(0.30, max(0.06, r)), 3)]}


def write_report(verdicts, path):
    payload = {
        "frames": len(verdicts),
        "flagged": sum(1 for v in verdicts.values() if not v["ok"]),
        "unchecked": sum(1 for v in verdicts.values() if v["defect"] == "unchecked"),
        "details": [{"file": Path(p).name, **v}
                    for p, v in verdicts.items() if not v["ok"]],
    }
    Path(path).write_text(json.dumps(payload, indent=2, ensure_ascii=False),
                          encoding="utf-8")
    return payload


THUMB_SYSTEM = """You judge candidate YouTube thumbnail photographs. Each shows a person
pointing at a computer monitor in a dark room. Be strict: a bad hand or a stare into
the lens costs clicks and trust.

Reject (ok=false) when ANY of these is clearly true:
- hands: a visible hand without exactly five fingers, fused or extra fingers, a
  malformed or melted hand, a pointing finger that bends wrongly.
- face: the face is distorted, asymmetric in an unnatural way, or melted.
- gaze: the person looks into the camera instead of at the monitor.
- screen: the monitor screen shows text, letters, symbols or an interface instead of
  being blank.
- text: any lettering, logo or watermark anywhere in the image.
Otherwise ok=true. Reply with valid JSON only."""

THUMB_PROMPT = """For each image, in order, return:
{"results":[{"i":<index>,"ok":true|false,"defect":"hands|face|gaze|screen|text|none",
"note":"<a few words>",
"screen":[[x,y],[x,y],[x,y],[x,y]],
"face":[x0,y0,x1,y1],
"score":<1-10, how strongly the image reads as genuine shock at something on screen>}]}
"screen" is the four corners of the monitor's display area, clockwise from top-left;
"face" is the box around the person's head. All coordinates are fractions 0-1."""


def judge_thumbnails(paths, model=VISION_MODEL):
    """{path: verdict} for thumbnail bases: hands, face, gaze, blank screen,
    plus where the screen and the face are, and a score for picking the best."""
    key = llm.get_key(llm.GEMINI_KEY_NAME)
    if not key:
        raise llm.LLMError(f"{llm.GEMINI_KEY_NAME} is not set")
    out = {}
    paths = [Path(p) for p in paths]
    for start in range(0, len(paths), 4):
        batch = paths[start:start + 4]
        parts = [{"text": THUMB_PROMPT}]
        for i, p in enumerate(batch):
            parts.append({"text": f"image {i}:"})
            parts.append({"inline_data": {"mime_type": "image/jpeg", "data": _encode(p, 1024)}})
        payload = {"contents": [{"role": "user", "parts": parts}],
                   "systemInstruction": {"parts": [{"text": THUMB_SYSTEM}]},
                   "generationConfig": {"maxOutputTokens": 2048, "temperature": 0.0,
                                        "responseMimeType": "application/json"}}
        data = None
        for attempt in range(RETRIES):
            _throttle()
            req = urllib.request.Request(llm.GEMINI_URL.format(model=model),
                                         data=json.dumps(payload).encode(), method="POST",
                                         headers={"Content-Type": "application/json",
                                                  "x-goog-api-key": key})
            try:
                with urllib.request.urlopen(req, timeout=180) as resp:
                    data = json.loads(resp.read().decode("utf-8", "replace"))
                break
            except urllib.error.HTTPError as e:
                if e.code not in (429, 500, 502, 503, 504):
                    raise
                time.sleep(4 * (attempt + 1))
        results = []
        if data:
            cand = (data.get("candidates") or [{}])[0]
            text = "".join(p.get("text", "") for p in cand.get("content", {}).get("parts", [])
                           if "text" in p)
            results = llm.extract_json(text).get("results", [])
        by_i = {r.get("i"): r for r in results}
        for i, p in enumerate(batch):
            r = by_i.get(i)
            out[p] = ({"ok": False, "defect": "unchecked", "note": "no verdict", "score": 0}
                      if r is None else
                      {"ok": bool(r.get("ok", False)), "defect": (r.get("defect") or "none").lower(),
                       "note": r.get("note", ""), "screen": r.get("screen"), "face": r.get("face"),
                       "score": float(r.get("score") or 0)})
    return out
