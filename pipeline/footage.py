"""Real footage: Pixabay video, Wikimedia Commons, NASA / NOAA / USFWS public domain.

Every candidate carries its licence from the moment it is found, so credits.json
is written from the same records the edit uses. Accepted licences:
  - Pixabay Content License (no attribution required; credited anyway)
  - public domain (US federal works: NASA, NOAA, USFWS; PD-marked Commons files)
  - CC0, CC BY (any version)  ->  author named in the description
CC BY-SA, NC and ND are refused: share-alike would bind the whole film, and the
other two forbid what a monetised channel does.

Nothing with people, logos or watermarks: tags are screened here, frames are
checked by pipeline/visualqc.py later.
"""
import hashlib
import json
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "cache" / "search"
UA = "Mozilla/5.0 (prehistoric-channel; research use) Python-urllib"

PEOPLE = re.compile(r"\b(people|person|man|men|woman|women|girl|boy|child|children|kid|human|"
                    r"diver|divers|diving|scuba|snorkel\w*|swimmer|surfer|tourist|hand|hands|"
                    r"face|selfie|fisherman|crowd|family|couple|boat|ship|logo|text|city|car)\b", re.I)
BAD_LICENCE = re.compile(r"\b(SA|NC|ND)\b|share.?alike|non.?commercial|no.?deriv", re.I)


def _env(name):
    if os.environ.get(name):
        return os.environ[name]
    env = ROOT / ".env"
    if env.exists():
        for line in env.read_text(encoding="utf-8-sig").splitlines():
            k, _, v = line.partition("=")
            if k.strip() == name:
                return v.strip().strip(chr(34)).strip(chr(39))
    return None


def _get(url, cache_key=None, timeout=40):
    CACHE.mkdir(parents=True, exist_ok=True)
    key = cache_key or hashlib.sha1(url.encode()).hexdigest()
    path = CACHE / f"{key}.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    for attempt in range(4):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                data = json.loads(r.read().decode("utf-8"))
            path.write_text(json.dumps(data), encoding="utf-8")
            return data
        except Exception:                                        # noqa: BLE001
            if attempt == 3:
                raise
            time.sleep(2 + attempt * 3)


# ------------------------------------------------------------------ Pixabay
def pixabay(query, per_page=80):
    key = _env("PIXABAY_API_KEY")
    url = "https://pixabay.com/api/videos/?" + urllib.parse.urlencode(
        {"key": key, "q": query, "per_page": per_page, "safesearch": "true"})
    data = _get(url, "pixabay_" + hashlib.sha1(f"{query}|{per_page}".encode()).hexdigest())
    out = []
    for h in data.get("hits", []):
        if PEOPLE.search(h.get("tags", "")):
            continue
        v = h["videos"]
        best = max((v[s] for s in ("large", "medium") if v.get(s, {}).get("url")),
                   key=lambda x: x["width"])
        out.append({
            "id": f"pixabay-{h['id']}", "source": "Pixabay", "query": query,
            "page": h["pageURL"], "url": best["url"], "width": best["width"],
            "height": best["height"], "duration": h["duration"], "tags": h["tags"],
            "thumb": best.get("thumbnail") or v["small"].get("thumbnail"),
            "preview": v.get("tiny", {}).get("url") or v["small"]["url"],
            "author": h.get("user"), "license": "Pixabay Content License",
            "license_url": "https://pixabay.com/service/license-summary/",
        })
    return out


# --------------------------------------------------------- Wikimedia Commons
COMMONS = "https://commons.wikimedia.org/w/api.php?"


def _clean(html):
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", "", html or "")).strip()


def commons(query, limit=40, media="video"):
    """Commons files (video by default) under PD, CC0 or CC BY only."""
    filetype = "filetype:video" if media == "video" else "filetype:bitmap"
    url = COMMONS + urllib.parse.urlencode({
        "action": "query", "format": "json", "generator": "search",
        "gsrsearch": f"{query} {filetype}", "gsrnamespace": 6, "gsrlimit": limit,
        "prop": "imageinfo", "iiprop": "url|size|mime|extmetadata|mediatype",
        "iiurlwidth": 640})
    data = _get(url)
    out = []
    for page in (data.get("query", {}).get("pages", {}) or {}).values():
        info = (page.get("imageinfo") or [{}])[0]
        meta = info.get("extmetadata", {})
        lic = _clean(meta.get("LicenseShortName", {}).get("value", ""))
        lic_url = meta.get("LicenseUrl", {}).get("value", "")
        ok = (lic.lower().startswith(("public domain", "pd", "cc0", "cc by")) and not BAD_LICENCE.search(lic))
        if not ok:
            continue
        desc = _clean(meta.get("ImageDescription", {}).get("value", ""))
        title = page["title"]
        if PEOPLE.search(title + " " + desc[:300]):
            continue
        out.append({
            "id": "commons-" + hashlib.sha1(title.encode()).hexdigest()[:10],
            "source": "Wikimedia Commons", "query": query, "title": title,
            "page": info.get("descriptionurl"), "url": info.get("url"),
            "width": info.get("width"), "height": info.get("height"),
            "duration": info.get("duration"), "mime": info.get("mime"),
            "thumb": info.get("thumburl"), "tags": desc[:300],
            "author": _clean(meta.get("Artist", {}).get("value", ""))[:200] or "unknown",
            "license": lic, "license_url": lic_url,
            "attribution_required": not lic.lower().startswith(("public domain", "pd", "cc0")),
        })
    return out


# ---------------------------------------------------------------- NASA (PD)
def nasa(query, media="video"):
    url = "https://images-api.nasa.gov/search?" + urllib.parse.urlencode(
        {"q": query, "media_type": media})
    data = _get(url)
    out = []
    for item in data.get("collection", {}).get("items", [])[:40]:
        d = item["data"][0]
        if PEOPLE.search(d.get("title", "") + " " + " ".join(d.get("keywords", []))):
            continue
        out.append({
            "id": f"nasa-{d['nasa_id']}", "source": "NASA", "query": query,
            "title": d.get("title"), "page": f"https://images.nasa.gov/details/{d['nasa_id']}",
            "manifest": item.get("href"), "tags": d.get("description", "")[:300],
            "author": d.get("center") or "NASA", "license": "Public domain (NASA)",
            "license_url": "https://www.nasa.gov/nasa-brand-center/images-and-media/",
            "thumb": next((l["href"] for l in item.get("links", []) if l.get("rel") == "preview"), None),
        })
    return out


def nasa_files(manifest_url):
    """The file URLs inside a NASA asset manifest, largest video first."""
    files = _get(manifest_url)
    vids = [f for f in files if f.lower().endswith((".mp4", ".mov"))]
    order = ("~orig", "~large", "~medium", "~mobile")
    vids.sort(key=lambda f: next((i for i, o in enumerate(order) if o in f), 9))
    return [f.replace("http://", "https://") for f in vids]


def download(url, dest, min_bytes=50_000):
    dest = Path(dest)
    if dest.exists() and dest.stat().st_size >= min_bytes:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    for attempt in range(6):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=600) as r, open(tmp, "wb") as fh:
                while chunk := r.read(1 << 20):
                    fh.write(chunk)
            break
        except urllib.error.HTTPError as e:
            if e.code != 429 or attempt == 5:
                raise
            time.sleep(20 * (attempt + 1))         # Wikimedia asks for patience, not retries
    if tmp.stat().st_size < min_bytes:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"{url} came back short")
    tmp.replace(dest)
    return dest
