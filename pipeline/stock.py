"""Real footage and photographs: Pixabay video, Wikimedia Commons, NASA.

The first hook was almost all generated monitors. Real footage of the world the
story happens in - a city at night, rain on glass, hands on a keyboard, a server
room, a library, a phone in someone's hands, people in a street seen from behind -
breaks that up and grounds the generated frames.

Sources, and what each licence allows:
  Pixabay videos   Pixabay Content License: free to use, no attribution needed
                   (we credit anyway). PIXABAY_API_KEY.
  Wikimedia Commons  only Public domain, CC0, CC BY and CC BY-SA files are taken,
                   and each is credited with author and licence.
  NASA Image Library  public domain (NASA media guidelines); credited as NASA.

Every asset is looked at before it is used: three frames of each clip and each
photograph go through the stock rulebook of the visual check (watermarks, real
logos, faces to the lens, anything graphic). What fails is dropped. Credits for
everything that is used are written to credits.json and go into the description.
"""
import json
import os
import random
import re
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from core.logger import get_logger

log = get_logger("stock")

UA = "ClosedTabs/1.0 (https://edbackulin-eng.github.io/closed-tabs/; edwardcatstory@gmail.com)"

# What the channel's world looks like. Queries are generic on purpose: nothing
# here depicts a real event or a real person from a story.
VIDEO_QUERIES = [
    "city night aerial", "night city traffic", "rain window night", "rain glass city lights",
    "typing keyboard", "hands keyboard", "server room", "data center",
    "library books", "old books shelf", "smartphone hands night", "phone scrolling",
    "street night people", "neon street", "crowd walking city", "subway night",
    "night highway", "fiber optic", "code screen dark", "city lights bokeh",
]
COMMONS_QUERIES = [
    "night city skyline long exposure", "library reading room interior", "old books shelf",
    "rain on window at night", "utility pole posters", "server room data center",
    "street at night neon", "telephone booth at night",
]
NASA_QUERIES = ["city lights at night from space", "earth at night", "night lights europe"]

COMMONS_OK = re.compile(r"^(public domain|pd|cc0|cc by(-sa)? ?\d(\.\d)?)", re.I)


def _get(url, timeout=60):
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if 400 <= e.code < 500 and e.code != 429:
                body = e.read()[:160].decode("utf-8", "replace")
                raise RuntimeError(f"HTTP {e.code}: {body}") from None
            if attempt == 2:
                raise
            log.info("fetch failed (%s), retrying", e)
            time.sleep(3 * (attempt + 1))
        except Exception as e:                                   # noqa: BLE001
            if attempt == 2:
                raise
            log.info("fetch failed (%s), retrying", e)
            time.sleep(3 * (attempt + 1))


def _json(url):
    return json.loads(_get(url).decode("utf-8", "replace"))


# --- searching ------------------------------------------------------------------

def pixabay_videos(query, key, per_page=30):
    q = urllib.parse.urlencode({"key": key, "q": query, "per_page": per_page,
                                "safe_search": "true", "video_type": "film"})
    out = []
    for h in _json(f"https://pixabay.com/api/videos/?{q}").get("hits", []):
        vids = h.get("videos", {})
        best = None
        for size in ("medium", "large"):
            v = vids.get(size) or {}
            if v.get("url") and v.get("width", 0) >= 1920:
                best = v
                break
        if not best:
            continue
        out.append({"kind": "video", "source": "Pixabay", "id": f"pixabay-{h['id']}",
                    "url": best["url"], "page": h.get("pageURL", ""),
                    "author": h.get("user", ""), "license": "Pixabay Content License",
                    "title": h.get("tags", ""), "duration": float(h.get("duration", 0))})
    return out


def commons_images(query, limit=30):
    q = urllib.parse.urlencode({
        "action": "query", "format": "json", "generator": "search", "gsrnamespace": 6,
        "gsrsearch": f"{query} filetype:bitmap", "gsrlimit": limit, "prop": "imageinfo",
        "iiprop": "url|size|extmetadata", "iiurlwidth": 2400})
    pages = _json(f"https://commons.wikimedia.org/w/api.php?{q}").get("query", {}).get("pages", {})
    out = []
    for p in pages.values():
        ii = (p.get("imageinfo") or [{}])[0]
        em = ii.get("extmetadata", {})
        lic = re.sub("<[^>]+>", "", em.get("LicenseShortName", {}).get("value", "")).strip()
        if not COMMONS_OK.match(lic) or ii.get("width", 0) < 1600 or ii.get("height", 0) < 900:
            continue
        if ii.get("width", 1) / max(1, ii.get("height", 1)) < 1.2:      # portrait crops badly
            continue
        author = re.sub("<[^>]+>", "", em.get("Artist", {}).get("value", "")).strip()[:120]
        out.append({"kind": "image", "source": "Wikimedia Commons", "id": f"commons-{p['pageid']}",
                    "url": ii.get("thumburl") or ii["url"], "page": ii.get("descriptionurl", ""),
                    "author": author or "unknown", "license": lic, "title": p.get("title", "")})
    return out


def nasa_images(query, limit=20):
    q = urllib.parse.urlencode({"q": query, "media_type": "image"})
    items = _json(f"https://images-api.nasa.gov/search?{q}").get("collection", {}).get("items", [])
    out = []
    for it in items[:limit]:
        d = (it.get("data") or [{}])[0]
        nid = d.get("nasa_id")
        if not nid:
            continue
        out.append({"kind": "image", "source": "NASA", "id": f"nasa-{nid}",
                    "url": None, "nasa_id": nid,
                    "page": f"https://images.nasa.gov/details/{urllib.parse.quote(nid)}",
                    "author": "NASA", "license": "Public domain (NASA)", "title": d.get("title", "")})
    return out


def _nasa_url(nid):
    files = _json(f"https://images-api.nasa.gov/asset/{urllib.parse.quote(nid)}")
    hrefs = [i["href"] for i in files.get("collection", {}).get("items", [])]
    for tag in ("~large.jpg", "~orig.jpg", "~medium.jpg"):
        for h in hrefs:
            if h.endswith(tag):
                return h.replace("http://", "https://")
    return None


# --- downloading and checking -------------------------------------------------

def _frames(video, dest_stem, duration):
    paths = []
    for k, f in enumerate((0.15, 0.5, 0.85)):
        p = Path(f"{dest_stem}_qc{k}.jpg")
        subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-ss",
                        f"{max(0.0, duration * f):.2f}", "-i", str(video), "-frames:v", "1",
                        "-vf", "scale=768:-2", str(p)], check=False)
        if p.exists():
            paths.append(p)
    return paths


def _probe_duration(path):
    r = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                        "-of", "csv=p=0", str(path)], capture_output=True, text=True)
    try:
        return float(r.stdout.strip())
    except ValueError:
        return 0.0


def collect(n_video, n_image, dest, seed=0, min_seconds=5.0):
    """Download, check and keep up to n_video clips and n_image photographs.

    Returns the kept assets, each with a local "path". Never raises for a source
    being down: what cannot be had is simply fewer assets, and the planner gives
    those shots back to generated stills.
    """
    from pipeline import visualqc
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    # stripped: the stored secret carried leading whitespace and Pixabay answered 400
    key = os.environ.get("PIXABAY_API_KEY", "").strip()

    videos, images = [], []
    if key and n_video:
        qs = VIDEO_QUERIES[:]
        rng.shuffle(qs)
        for q in qs:
            try:
                hits = [h for h in pixabay_videos(q, key) if h["duration"] >= min_seconds]
                rng.shuffle(hits)
                videos += hits[:4]          # a few per query keeps the mix varied
            except Exception as e:                               # noqa: BLE001
                log.warning("pixabay %r failed: %s", q, e)
    elif n_video:
        log.warning("PIXABAY_API_KEY not set; no stock video")
    if n_image:
        for q in COMMONS_QUERIES:
            try:
                hits = commons_images(q)
                rng.shuffle(hits)
                images += hits[:3]
            except Exception as e:                               # noqa: BLE001
                log.warning("commons %r failed: %s", q, e)
        for q in NASA_QUERIES:
            try:
                hits = nasa_images(q)
                rng.shuffle(hits)
                images += hits[:2]
            except Exception as e:                               # noqa: BLE001
                log.warning("nasa %r failed: %s", q, e)
    seen = set()
    videos = [v for v in videos if not (v["id"] in seen or seen.add(v["id"]))]
    images = [v for v in images if not (v["id"] in seen or seen.add(v["id"]))]

    kept = []

    def take(cands, want):
        got = []
        # Download a little more than needed: some will fail the check.
        for a in cands:
            if len(got) >= want:
                break
            try:
                if a["kind"] == "video":
                    p = dest / f"{a['id']}.mp4"
                else:
                    url = a["url"] or _nasa_url(a["nasa_id"])
                    if not url:
                        continue
                    a["url"] = url
                    p = dest / f"{a['id']}.jpg"
                if not p.exists():
                    p.write_bytes(_get(a["url"], timeout=180))
                a["path"] = str(p)
                if a["kind"] == "video":
                    a["duration"] = _probe_duration(p) or a.get("duration", 0)
                    a["qc_frames"] = [str(x) for x in _frames(p, dest / a["id"], a["duration"])]
                else:
                    a["qc_frames"] = [str(p)]
                got.append(a)
            except Exception as e:                               # noqa: BLE001
                log.warning("download %s failed: %s", a["id"], e)
        return got

    batch = take(videos, int(n_video * 1.4) + 2) + take(images, int(n_image * 1.4) + 2)
    frames = [Path(f) for a in batch for f in a["qc_frames"]]
    verdicts = visualqc.inspect(frames, kinds={f: "stock" for f in frames}) if frames else {}
    nv = ni = 0
    for a in batch:
        vs = [verdicts.get(Path(f), {"ok": False, "defect": "unchecked"}) for f in a["qc_frames"]]
        # Unchecked is not clean: an asset nobody could look at is not used.
        if not vs or not all(v["ok"] for v in vs):
            bad = next((v for v in vs if not v["ok"]), {"defect": "no frames"})
            log.info("stock %s dropped: %s %s", a["id"], bad.get("defect"), bad.get("note", ""))
            continue
        if a["kind"] == "video" and nv < n_video:
            kept.append(a)
            nv += 1
        elif a["kind"] == "image" and ni < n_image:
            kept.append(a)
            ni += 1
    log.info("stock: %d clips and %d photographs kept of %d checked", nv, ni, len(batch))
    return kept


def credit(asset):
    """One line of the description's credits."""
    by = f" by {asset['author']}" if asset.get("author") and asset["source"] != "NASA" else ""
    return f"{asset['source']}{by} ({asset['license']}): {asset.get('page') or asset.get('url')}"


def write_credits(assets, path):
    rows = [{k: a.get(k) for k in ("source", "id", "title", "author", "license", "page", "url")}
            for a in assets]
    Path(path).write_text(json.dumps(rows, indent=1, ensure_ascii=False), encoding="utf-8")
    return rows
