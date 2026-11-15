"""The real pictures of an episode (owner 15.11): no FLUX, no LTX, no 2.5D - only real footage
from open sources with a licence read from the source's own page, and palaeo-art paintings
from Wikimedia Commons as stills with a slow camera move.

    python -m pipeline.realpool <slug>          # harvest + describe -> episodes/<slug>/pool.json

1. queries: the episode's animals (the extinct ones and their living relatives), the science
   (excavation, palaeontologists at work, preparation, skeletons in museums, fossils close up)
   and the landscapes of the finds - written by the text model from the topic, in English;
2. harvest, licence first:
     Pixabay video        Pixabay Content License (the API serves only such files)
     Wikimedia Commons    video and pictures: public domain, CC0, CC BY, CC BY-SA - the licence
                          and author from the file page's own metadata (extmetadata), credited
     archive.org          films whose item metadata carries a public-domain, CC0, CC BY or
                          CC BY-SA licenceurl; each film gives a few separate stretches
     Pexels video         only when PEXELS_API_KEY is set (Pexels License)
   Computer renders, cartoons, AI pictures, text and watermarks are refused (tags and titles
   here, the picture itself in step 3);
3. describe: every candidate's picture goes to the vision model, ten at a time: what it shows
   (category, the animal, a few words), whether it is a computer render or carries text or a
   watermark, its quality. What passes is the pool the plan cuts the film from;
4. material(): how many seconds of film the pool can fill without showing anything twice -
   the topic is built only when that is enough for 10 minutes (owner 15.11).
"""
import hashlib
import json
import re
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from core.logger import get_logger  # noqa: E402
from pipeline import footage  # noqa: E402

log = get_logger("realpool")
CACHE = ROOT / "cache" / "pool"
UA = {"User-Agent": "prehistoric-channel/1.0 (documentary research; https://github.com/edbackulin-eng)"}

# owner 15.11 (second message): the films tell the HISTORY OF A FIND - who, where, when, the expeditions, the
# quarrels of the scientists. Archive pictures of that very find come first; living relatives and landscapes
# only fill in.
CATEGORIES = ["archive_photo", "archive_film", "archive_illustration", "portrait", "excavation", "palaeontologist",
              "preparation", "museum", "skeleton", "fossil", "palaeoart", "living_animal", "landscape", "other"]
ARCHIVE = {"archive_photo", "archive_film", "archive_illustration", "portrait"}
NEUTRAL = {"excavation", "preparation", "museum", "palaeontologist", "landscape", "archive_photo", "archive_film",
           "archive_illustration", "portrait"}
FILL_ONLY = {"living_animal", "landscape"}
VIDEO_SHOT_MAX = 8.0          # one clip, one shot (owner 15.11: no repeats) - of at most this long
STILL_SHOT_S = 6.0
STILL_SHARE_MAX = 0.85        # archive photographs are the heart of these films; some film all the same
MIN_VIDEO_W, MIN_STILL_W = 1280, 1600
ARCHIVE_MAX_MB = 400
ARCHIVE_STRETCHES = 4
CGI = re.compile(r"\b(3d|cgi|render(ed|ing)?|animation|animated|cartoon|illustrat\w*|vector|clip ?art|"
                 r"ai[- ]generated|generated|stable diffusion|midjourney|dall-?e|toy|figurine|lego|plastic|"
                 r"game|gaming|unreal|blender)\b", re.I)
AI_ART = re.compile(r"ai[- ]generated|stable diffusion|midjourney|dall-?e|neural network generated", re.I)
LICENCE_OK = re.compile(r"^(public domain|pd\b|pd-|cc0|cc[ -]by(-sa)?[ -]?\d?(\.\d)?)", re.I)
LICENCE_BAD = re.compile(r"\b(NC|ND)\b|non.?commercial|no.?deriv", re.I)


# ------------------------------------------------------------------ queries
QUERIES = """Eine deutsche Doku erzählt die GESCHICHTE EINER ENTDECKUNG: "{title}" ({focus}) - wer das Tier fand,
wo und wann, die Expeditionen, die Streite der Forscher. Ich suche ECHTES Material aus freien Quellen (Wikimedia
Commons, archive.org, Library of Congress, Museumssammlungen, Pixabay). KI-Bilder und 3D-Animationen gibt es nicht.
Nenne:
- "discovery": {{"people": ["<Forscher, Sammler, Finder mit vollem Namen>"], "places": ["<Fundorte, Museen>"],
  "years": ["<Jahreszahlen>"], "queries": ["<10-14 englische Suchbegriffe für ARCHIVMATERIAL genau dieser Entdeckung:
  Namen der Forscher (Porträts), 'Name expedition', 'Ort quarry 1900s', Museum + Tier, historische Stiche/Zeichnungen
  der Funde, alte Fotos der Grabung, Erstbeschreibung/Tafel>"]}};
- "animals": die Tiere des Films - die ausgestorbenen Hauptfiguren und ihre heute lebenden Verwandten -
  jeweils mit key (snake_case englisch), name_de, latin, "extinct": true|false, "relative_of": "<key>" bei
  Verwandten, und "queries": 3-5 englische Suchbegriffe (bei ausgestorbenen: "<latin>", "<latin> fossil",
  "<latin> skeleton", "<latin> restoration", "<latin> life reconstruction"; bei lebenden: ihr englischer Name,
  z. B. "monitor lizard", "komodo dragon swimming"). Bei einer Gruppe auch 1-2 bekannte Gattungen als eigene Tiere;
- "science": 6-8 englische Suchbegriffe für Forschung: Ausgrabung, Paläontologen bei der Arbeit, Präparation
  von Knochen, Skelette im Museum, Fossilien in Nahaufnahme - passend zu den Tieren und Fundorten;
- "landscapes": 4-6 englische Suchbegriffe für die Landschaften und Lebensräume (Fundorte, Meer, Küste ...).
Antworte als JSON: {{"discovery": {{...}}, "animals": [...], "science": [...], "landscapes": [...]}}"""


def queries(topic):
    from core import textllm
    try:
        ans = textllm.ask(QUERIES.format(title=topic.get("title", topic["slug"]), focus=topic.get("focus", "")),
                          step="pool-queries", temperature=0.3, max_tokens=1800)
    except Exception as e:                                               # noqa: BLE001
        log.warning("queries: no answer (%s); from the slug only", e)
        ans = {}
    animals = [a for a in (ans.get("animals") or []) if isinstance(a, dict) and a.get("key")]
    if not animals:
        name = topic["slug"].replace("-", " ")
        animals = [{"key": topic["slug"].replace("-", "_"), "name_de": topic.get("title"), "latin": name,
                    "extinct": True, "queries": [f"{name} fossil", f"{name} skeleton", f"{name} museum"]}]
    science = [q for q in (ans.get("science") or []) if isinstance(q, str)] or [
        "paleontology excavation", "paleontologist fossil dig", "fossil preparation lab", "dinosaur skeleton museum",
        "fossil close up", "natural history museum"]
    lands = [q for q in (ans.get("landscapes") or []) if isinstance(q, str)] or ["badlands", "ocean waves", "cliff coast"]
    disc = ans.get("discovery") if isinstance(ans.get("discovery"), dict) else {}
    disc = {k: [str(x) for x in (disc.get(k) or []) if x] for k in ("people", "places", "years", "queries")}
    if not disc["queries"]:
        disc["queries"] = [f"{a.get('latin') or a['key']} discovery" for a in animals if a.get("extinct")][:3]
    disc["queries"] += [p for p in disc["people"] if p not in disc["queries"]][:6]
    return {"discovery": disc, "animals": animals, "science": science, "landscapes": lands}


# ------------------------------------------------------------------ sources
def _get_json(url, timeout=40):
    for attempt in range(4):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=timeout) as r:
                return json.loads(r.read().decode("utf-8"))
        except Exception:                                                # noqa: BLE001
            if attempt == 3:
                raise
            time.sleep(3 + 5 * attempt)


def pixabay(query, cat, per_page=40):
    key = footage._env("PIXABAY_API_KEY")
    if not key:
        return []
    url = "https://pixabay.com/api/videos/?" + urllib.parse.urlencode(
        {"key": key, "q": query, "per_page": per_page, "safesearch": "true"})
    data = footage._get(url, "pixabay_" + hashlib.sha1(f"{query}|{per_page}".encode()).hexdigest())
    out = []
    for h in data.get("hits", []):
        if CGI.search(h.get("tags", "")):
            continue
        v = h["videos"]
        sizes = [v[s] for s in ("large", "medium") if v.get(s, {}).get("url")]
        if not sizes:
            continue
        best = max(sizes, key=lambda x: x["width"])
        if best["width"] < MIN_VIDEO_W or (h.get("duration") or 0) < 4:
            continue
        out.append({"id": f"pixabay-{h['id']}", "source": "Pixabay", "kind": "video", "query": query, "cat": cat,
                    "page": h["pageURL"], "url": best["url"], "width": best["width"], "height": best["height"],
                    "duration": h["duration"], "tags": h.get("tags", ""),
                    "thumb_url": best.get("thumbnail") or v.get("small", {}).get("thumbnail"),
                    "author": h.get("user"), "license": "Pixabay Content License",
                    "license_url": "https://pixabay.com/service/license-summary/", "attribution_required": False})
    return out


def pexels(query, cat, per_page=30):
    key = footage._env("PEXELS_API_KEY")
    if not key:
        return []
    url = "https://api.pexels.com/videos/search?" + urllib.parse.urlencode({"query": query, "per_page": per_page})
    try:
        req = urllib.request.Request(url, headers={**UA, "Authorization": key})
        with urllib.request.urlopen(req, timeout=40) as r:
            data = json.loads(r.read().decode("utf-8"))
    except Exception as e:                                               # noqa: BLE001
        log.warning("pexels %s: %s", query, e)
        return []
    out = []
    for v in data.get("videos", []):
        files = [f for f in v.get("video_files", []) if f.get("file_type") == "video/mp4" and (f.get("width") or 0)
                 >= MIN_VIDEO_W]
        if not files or (v.get("duration") or 0) < 4:
            continue
        f = min(files, key=lambda f: abs((f.get("width") or 0) - 3840))
        out.append({"id": f"pexels-{v['id']}", "source": "Pexels", "kind": "video", "query": query, "cat": cat,
                    "page": v.get("url"), "url": f["link"], "width": f["width"], "height": f["height"],
                    "duration": v.get("duration"), "tags": v.get("url", "").rsplit("/", 2)[-2].replace("-", " "),
                    "thumb_url": v.get("image"), "author": (v.get("user") or {}).get("name"),
                    "license": "Pexels License", "license_url": "https://www.pexels.com/license/",
                    "attribution_required": False})
    return out


WM_API = "https://commons.wikimedia.org/w/api.php?"
WM_UA = {"User-Agent": "prehistoric-channel/1.0 (https://github.com/edbackulin-eng/prehistoric-channel; "
                       "documentary research) python-urllib"}
WM_PAUSE_S = 1.2
_wm_last = [0.0]
_off = set()                  # sources that refused this run (LoC 403/429): not asked again


def wm(params):
    """One Commons API call, cached; Wikimedia's etiquette (15.11: 83 x 429 in one harvest): a client
    that says who it is, one request at a time with a pause, Retry-After respected."""
    url = WM_API + urllib.parse.urlencode({**params, "format": "json"})
    key = "wm_" + hashlib.sha1(url.encode()).hexdigest()
    cp = footage.CACHE / f"{key}.json"
    if cp.exists():
        return json.loads(cp.read_text(encoding="utf-8"))
    for attempt in range(6):
        wait = WM_PAUSE_S - (time.time() - _wm_last[0])
        if wait > 0:
            time.sleep(wait)
        _wm_last[0] = time.time()
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=WM_UA), timeout=60) as r:
                data = json.loads(r.read().decode("utf-8"))
            footage.CACHE.mkdir(parents=True, exist_ok=True)
            cp.write_text(json.dumps(data), encoding="utf-8")
            return data
        except urllib.error.HTTPError as e:
            if e.code == 429 or e.code >= 500:
                ra = e.headers.get("Retry-After") if e.headers else None
                time.sleep(min(180, float(ra) if ra and str(ra).isdigit() else 20 * (attempt + 1)))
                continue
            raise
        except Exception:                                                # noqa: BLE001
            time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"Commons kept refusing: {params.get('gsrsearch') or params.get('cmtitle') or params}")


def _clean(html):
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", "", html or "")).strip()


def commons(query, cat, media, limit=40, min_w=None):
    """Commons files with their licence from the file page's metadata: PD, CC0, CC BY, CC BY-SA."""
    ft = "filetype:video" if media == "video" else "filetype:bitmap"
    try:
        data = wm({"action": "query", "generator": "search", "gsrsearch": f"{query} {ft}",
                   "gsrnamespace": 6, "gsrlimit": limit, "prop": "imageinfo",
                   "iiprop": "url|size|mime|extmetadata|mediatype", "iiurlwidth": 4224 if media != "video" else 640})
    except Exception as e:                                               # noqa: BLE001
        log.warning("commons %s: %s", query, e)
        return []
    return _commons_items(((data.get("query") or {}).get("pages") or {}).values(), query, cat, media, min_w)


SKIP_CAT = re.compile(r"stamps?|coins?|logos?|maps?\b|diagrams?|charts?|cladogram|size comparison|toys?|"
                      r"figurines?|sculptures in|statues? of|art by|cartoons?|video games?|in culture|"
                      r"in popular culture|lego|models? of", re.I)


def commons_categories(term, cat, max_files=120, depth=2):
    """owner 15.11: the find's own pictures - Commons sorts them in categories (the specimen, its museum,
    the quarry, the discoverer), where a text search hardly reaches. The categories matching the term,
    their files and those of their subcategories (two levels), each file with its licence."""
    try:
        found = wm({"action": "query", "list": "search", "srsearch": term, "srnamespace": 14, "srlimit": 8})
    except Exception as e:                                               # noqa: BLE001
        log.warning("commons categories %s: %s", term, e)
        return []
    todo = [(c["title"], 0) for c in found.get("query", {}).get("search", []) if not SKIP_CAT.search(c["title"])]
    seen_cats, files = set(), []
    while todo and len(files) < max_files:
        title, d = todo.pop(0)
        if title in seen_cats:
            continue
        seen_cats.add(title)
        try:
            mem = wm({"action": "query", "list": "categorymembers", "cmtitle": title, "cmtype": "file|subcat",
                      "cmlimit": 200})
        except Exception as e:                                           # noqa: BLE001
            log.warning("commons category %s: %s", title, e)
            continue
        for m in mem.get("query", {}).get("categorymembers", []):
            t = m["title"]
            if t.startswith("Category:"):
                if d < depth and not SKIP_CAT.search(t):
                    todo.append((t, d + 1))
            elif re.search(r"\.(jpe?g|png|tiff?|webp|webm|ogv)$", t, re.I) and t not in files:
                files.append(t)
    out = []
    for k in range(0, min(len(files), max_files), 15):
        batch = files[k:k + 15]
        for media in ("bitmap", "video"):
            names = [f for f in batch if (f.lower().endswith((".webm", ".ogv")) == (media == "video"))]
            if not names:
                continue
            try:
                d = wm({"action": "query", "titles": "|".join(names), "prop": "imageinfo",
                        "iiprop": "url|size|mime|extmetadata|mediatype", "iiurlwidth": 4224 if media != "video" else 640})
            except Exception as e:                                       # noqa: BLE001
                log.warning("commons files of %s: %s", term, e)
                continue
            out += _commons_items(((d.get("query") or {}).get("pages") or {}).values(), term, cat, media, 900)
    log.info("commons categories %s: %d categories, %d files, %d with a free licence", term, len(seen_cats),
             len(files), len(out))
    return out


def _commons_items(pages, query, cat, media, min_w):
    out = []
    for page in pages:
        info = (page.get("imageinfo") or [{}])[0]
        meta = info.get("extmetadata", {})
        lic = _clean(meta.get("LicenseShortName", {}).get("value", ""))
        if not LICENCE_OK.search(lic) or LICENCE_BAD.search(lic):
            continue
        title = page["title"]
        desc = _clean(meta.get("ImageDescription", {}).get("value", ""))[:400]
        cats = _clean(meta.get("Categories", {}).get("value", ""))[:400]
        if AI_ART.search(f"{title} {desc} {cats}"):
            continue
        w, h = info.get("width") or 0, info.get("height") or 0
        if media == "video":
            if w < MIN_VIDEO_W or (info.get("duration") or 0) < 4:
                continue
        elif w < (min_w or MIN_STILL_W) or h < 600 or not str(info.get("mime", "")).startswith("image/"):
            continue
        item = {"id": "commons-" + hashlib.sha1(title.encode()).hexdigest()[:10], "source": "Wikimedia Commons",
                "kind": "video" if media == "video" else "still", "query": query, "cat": cat, "title": title,
                "page": info.get("descriptionurl"), "url": info.get("url"), "width": w, "height": h,
                "duration": info.get("duration"), "tags": f"{title} {desc}"[:400],
                "thumb_url": (re.sub(r"/\d+px-", "/640px-", info["thumburl"], count=1)
                              if "/thumb/" in str(info.get("thumburl") or "") else info.get("thumburl")),
                "author": _clean(meta.get("Artist", {}).get("value", ""))[:200] or "unknown",
                "license": lic, "license_url": meta.get("LicenseUrl", {}).get("value", ""),
                "attribution_required": not re.match(r"public domain|pd\b|pd-|cc0", lic, re.I)}
        if media != "video":
            item["full_url"] = info.get("thumburl") or info.get("url")   # 4224 px wide (or the original)
        out.append(item)
    return out


LOC_FREE = re.compile(r"no known restrictions|public domain|no known copyright", re.I)


def loc(query, cat, media="photos", rows=20):
    """Library of Congress photographs and films marked free to use ("No known restrictions on publication").
    A refusal of the API (it turns some clients away) is logged and the source skipped."""
    if "loc" in _off:
        return []
    url = f"https://www.loc.gov/{media}/?" + urllib.parse.urlencode({"q": query, "fo": "json", "c": rows})
    try:
        time.sleep(2)
        with urllib.request.urlopen(urllib.request.Request(url, headers=WM_UA), timeout=40) as r:
            data = json.loads(r.read().decode("utf-8"))
    except Exception as e:                                               # noqa: BLE001
        log.warning("loc %s: %s - not asked again this run", query, str(e)[:120])
        _off.add("loc")
        return []
    out = []
    for r in data.get("results", []):
        rights = " ".join(str(x) for x in ([r.get("rights_advisory")] if isinstance(r.get("rights_advisory"), str)
                                           else (r.get("rights_advisory") or [])) + [str(r.get("rights") or "")])
        if not LOC_FREE.search(rights):
            continue
        imgs = [u for u in (r.get("image_url") or []) if isinstance(u, str)]
        if not imgs:
            continue
        big = imgs[-1].split("#")[0]
        item_id = str(r.get("id") or r.get("url") or big)
        out.append({"id": "loc-" + hashlib.sha1(item_id.encode()).hexdigest()[:10], "source": "Library of Congress",
                    "kind": "still", "query": query, "cat": cat, "title": str(r.get("title") or "")[:200],
                    "page": r.get("url") or item_id, "url": big, "full_url": big, "thumb_url": imgs[0].split("#")[0],
                    "width": 0, "height": 0, "tags": " ".join(str(x) for x in (r.get("subject") or []))[:300],
                    "date": r.get("date"), "author": str((r.get("contributor") or ["Library of Congress"])[0])[:200],
                    "license": "Public domain (Library of Congress: no known restrictions on publication)",
                    "license_url": r.get("url") or "", "attribution_required": False})
    return out


ARCHIVE_LIC = re.compile(r"publicdomain|creativecommons\.org/(licenses/by(-sa)?/|publicdomain/zero)", re.I)


def archive(query, cat, rows=8):
    """archive.org films with a free licence in their own metadata; a few stretches of each."""
    q = f"({query}) AND mediatype:(movies)"
    url = "https://archive.org/advancedsearch.php?" + urllib.parse.urlencode(
        {"q": q, "fl[]": ["identifier", "title", "licenseurl"], "rows": rows * 6, "output": "json"}, doseq=True)
    try:
        docs = _get_json(url)["response"]["docs"]
    except Exception as e:                                               # noqa: BLE001
        log.warning("archive %s: %s", query, e)
        return []
    # the search's own licence field first (cheap), the item's metadata decides below
    docs = [d for d in docs if ARCHIVE_LIC.search(str(d.get("licenseurl") or ""))][:rows]
    out = []
    for d in docs:
        ident = d["identifier"]
        try:
            meta = _get_json(f"https://archive.org/metadata/{ident}")
        except Exception:                                                # noqa: BLE001
            continue
        lic = (meta.get("metadata") or {}).get("licenseurl") or ""
        if isinstance(lic, list):
            lic = lic[0]
        if not ARCHIVE_LIC.search(lic) or LICENCE_BAD.search(lic) or "/nc" in lic.lower() or "/nd" in lic.lower():
            continue
        title = str((meta.get("metadata") or {}).get("title") or ident)
        if CGI.search(title):
            continue
        mp4 = [f for f in meta.get("files", []) if str(f.get("name", "")).lower().endswith(".mp4")
               and int(f.get("size") or 0) < ARCHIVE_MAX_MB * 1e6 and float(f.get("length") or 0) >= 20]
        if not mp4:
            continue
        f = max(mp4, key=lambda f: int(f.get("height") or 0) * 10 + int(f.get("size") or 0) / 1e9)
        if int(f.get("width") or 1280) < MIN_VIDEO_W:
            continue
        length = float(f["length"]) if ":" not in str(f["length"]) else sum(
            float(x) * 60 ** i for i, x in enumerate(reversed(str(f["length"]).split(":"))))
        src = f"https://archive.org/download/{ident}/{urllib.parse.quote(f['name'])}"
        author = str((meta.get("metadata") or {}).get("creator") or "archive.org")
        for k in range(ARCHIVE_STRETCHES):
            at = length * (k + 0.5) / ARCHIVE_STRETCHES
            out.append({"id": f"archive-{hashlib.sha1(ident.encode()).hexdigest()[:8]}-{k}", "source": "archive.org",
                        "kind": "video", "query": query, "cat": cat, "title": title,
                        "page": f"https://archive.org/details/{ident}", "url": src,
                        "width": int(f.get("width") or 0), "height": int(f.get("height") or 0),
                        "duration": length, "at_s": round(at, 2), "stretch_s": VIDEO_SHOT_MAX,
                        "tags": title, "author": author[:200], "license": _archive_lic(lic), "license_url": lic,
                        "attribution_required": "publicdomain" not in lic.lower()})
    return out


def _archive_lic(url):
    u = url.lower()
    if "publicdomain/zero" in u:
        return "CC0"
    if "publicdomain" in u:
        return "Public domain"
    m = re.search(r"licenses/(by(?:-sa)?)/(\d\.\d)", u)
    return f"CC {m.group(1).upper()} {m.group(2)}" if m else url


# ------------------------------------------------------------------ pictures
def _thumb(item, dest):
    """One picture of the candidate for the vision model: the source's poster, or a frame of the film."""
    if dest.exists() and dest.stat().st_size > 2000:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    if item.get("at_s") is not None:            # an archive.org stretch: its frame, read over HTTP
        r = subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", str(item["at_s"]), "-i", item["url"],
                            "-frames:v", "1", "-vf", "scale=640:-2", str(dest)], capture_output=True, timeout=180)
        if r.returncode or not dest.exists():
            raise RuntimeError(f"frame of {item['id']}: {r.stderr[-200:]}")
        return dest
    url = item.get("thumb_url")
    if not url:
        raise RuntimeError("no picture")
    for attempt in range(5):
        try:
            if "wikimedia.org" in url:
                time.sleep(0.4)
            with urllib.request.urlopen(urllib.request.Request(url, headers=WM_UA), timeout=60) as r:
                dest.write_bytes(r.read())
            return dest
        except urllib.error.HTTPError as e:
            if e.code != 429 or attempt == 4:
                raise
            time.sleep(15 * (attempt + 1))
    return dest


DESCRIBE = """Du sichtest Material für eine deutsche Doku über die Entdeckungsgeschichte von {title}. Die Tiere des
Films: {animals}. Zu jedem nummerierten Bild (ein Standbild aus einem Film oder ein Foto/Gemälde/Stich):
- "category": eins von {categories}
  (archive_photo = altes Foto (Schwarzweiß/Sepia, vor ca. 1970) von Grabung, Expedition, Forschern, Fundort, Museum;
  archive_film = Standbild aus einem alten Film; archive_illustration = historischer Stich, Zeichnung oder Tafel
  (Fund, Fundort, Forscher, Erstbeschreibung - kein moderner Palaeo-Art); portrait = Porträt einer Person
  (Foto oder Gemälde); excavation = Grabung im Gelände (neu); palaeontologist = Forscher bei der Arbeit (neu);
  preparation = Präparation/Labor; museum = Ausstellung, Sammlung; skeleton = montiertes Skelett; fossil =
  versteinerte Knochen/Zähne/Abdrücke in Nahaufnahme; palaeoart = gemalte/gezeichnete Lebendrekonstruktion eines
  Urzeittiers; living_animal = ein lebendes Tier; landscape = Landschaft/Meer ohne Tier im Mittelpunkt;
  other = alles andere);
- "animal": der key eines der Tiere oben, wenn es (lebend, als Fossil, Skelett oder Gemälde) zu sehen ist,
  sonst "other:<englischer Name>" für ein anderes Tier, sonst "none";
- "what": 4-8 englische Wörter, was zu sehen ist;
- "cgi": true, wenn es eine 3D-Computergrafik, ein Videospiel, ein Cartoon, ein Spielzeug/Modell aus Plastik oder
  ein KI-Bild ist (ein gemaltes Palaeo-Art-Bild ist KEIN cgi);
- "text": true, wenn Schrift, Logo, Wasserzeichen oder Untertitel im Bild stehen (Museumsschilder im Hintergrund: false);
- "quality": 1-5 (scharf, gut belichtet, ruhig = 5; unscharf, verrauscht, winzig = 1).
Antworte als JSON: {{"items": [{{"i": <n>, "category": "...", "animal": "...", "what": "...", "cgi": false,
"text": false, "quality": 4}}]}}
BILDER: {n} Stück, nummeriert 0 bis {last}."""
VISION_MODELS = ("gemini-flash-lite-latest", "gemma-4-31b-it", "gemma-4-26b-a4b-it", "gemini-flash-latest")


def describe(items, title, animals, cache_path, batch=10):
    """The vision model's look at each candidate; answers kept in cache_path (a rerun asks only the rest)."""
    from core import gemini
    cache = json.loads(cache_path.read_text(encoding="utf-8")) if cache_path.exists() else {}
    todo = [it for it in items if it["id"] not in cache]
    keys = ", ".join(f"{a['key']} ({a.get('name_de') or a.get('latin')})" for a in animals)
    pics = []
    for it in todo:
        try:
            pics.append((it, _thumb(it, CACHE / "thumbs" / f"{it['id']}.jpg")))
        except Exception as e:                                           # noqa: BLE001
            cache[it["id"]] = {"error": f"picture: {str(e)[:120]}"}
    for k in range(0, len(pics), batch):
        part = pics[k:k + batch]
        prompt = DESCRIBE.format(title=title, animals=keys, categories=CATEGORIES, n=len(part), last=len(part) - 1)
        try:
            ans = gemini.ask(prompt, images=[str(p) for _, p in part], models=VISION_MODELS, temperature=0.0,
                             max_tokens=2500, step="pool-describe")
        except Exception as e:                                           # noqa: BLE001
            log.warning("describe: no answer for %d pictures (%s); asked again on the next run", len(part), e)
            continue
        res = {int(r.get("i", -1)): r for r in gemini.pick(ans, "items") if isinstance(r, dict)}
        for j, (it, _) in enumerate(part):
            if j in res:
                cache[it["id"]] = {x: res[j].get(x) for x in ("category", "animal", "what", "cgi", "text", "quality")}
        cache_path.write_text(json.dumps(cache, indent=1, ensure_ascii=False), encoding="utf-8")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(cache, indent=1, ensure_ascii=False), encoding="utf-8")
    return cache


def usable(it):
    d = it.get("desc") or {}
    if not d or d.get("error") or d.get("cgi"):
        return False
    # an old plate or engraving carries its caption; a modern picture with text, a logo or a watermark does not go
    if d.get("text") and d.get("category") not in ("archive_illustration", "archive_photo", "portrait"):
        return False
    floor = 2 if d.get("category") in ARCHIVE else 3      # an archive photograph is old; it may look it
    if int(d.get("quality") or 0) < floor or d.get("category") not in CATEGORIES or d.get("category") == "other":
        return False
    if it["kind"] == "video" and d.get("category") == "palaeoart":
        return False                  # a painted reconstruction that moves is an animation: only stills
    if it["kind"] == "still" and d.get("category") == "living_animal" and it["source"] != "Wikimedia Commons":
        return False
    return True


def seconds_of(it):
    if it["kind"] == "still":
        return STILL_SHOT_S
    return min(VIDEO_SHOT_MAX, float(it.get("stretch_s") or it.get("duration") or 0))


def used_elsewhere(slug):
    """Pictures another film of the channel already showed (owner 15.11: a stock clip once in the whole
    channel, so the films do not look alike) - from every other episode's footage.json."""
    out = set()
    for f in (ROOT / "episodes").glob("*/footage.json"):
        if f.parent.name == slug:
            continue
        try:
            d = json.loads(f.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        for v in d.values():
            if isinstance(v, dict):
                out.add(v.get("id"))
                out.update(a.get("id") for a in v.get("alternates") or [] if isinstance(a, dict))
    out.discard(None)
    return out


def material(pool):
    """Seconds of film the pool fills, each picture once, stills at most STILL_SHARE_MAX of it; the
    living relatives and landscapes count only as a fill-in (a quarter of the film at most)."""
    ok = [it for it in pool["items"] if it.get("usable")]
    core = [it for it in ok if it["desc"]["category"] not in FILL_ONLY]
    fill = sum(seconds_of(it) for it in ok if it["desc"]["category"] in FILL_ONLY)
    video = sum(seconds_of(it) for it in core if it["kind"] == "video")
    still = sum(seconds_of(it) for it in core if it["kind"] == "still")
    still_used = min(still, max(video, 60) * STILL_SHARE_MAX / (1 - STILL_SHARE_MAX))
    fill_used = min(fill, (video + still_used) / 3)
    by_cat = {}
    for it in ok:
        c = it["desc"]["category"]
        by_cat[c] = by_cat.get(c, 0) + 1
    main = {a["key"] for a in pool["animals"] if a.get("extinct")}
    of_main = sum(1 for it in ok if it["desc"].get("animal") in main)
    archive_n = sum(1 for it in ok if it["desc"]["category"] in ARCHIVE)
    return {"seconds": round(video + still_used + fill_used), "video_s": round(video), "still_s": round(still_used),
            "fill_s": round(fill_used), "items": len(ok), "by_category": by_cat, "main_animal_items": of_main,
            "archive_items": archive_n}


def topic_words(qs):
    """Names that tie a picture to the story: the animals, the people's surnames, the places."""
    out = set()
    for a in qs.get("animals", []):
        if a.get("extinct"):
            for k in ("latin", "name_de", "key"):
                if a.get(k):
                    out.add(str(a[k]).lower().replace("_", " ").split()[0])
    disc = qs.get("discovery") or {}
    for person in disc.get("people") or []:
        parts = [x for x in re.split(r"[\s,.]+", str(person)) if len(x) >= 4]
        out.update(x.lower() for x in (parts[1:] or parts))     # the surname(s), not the first name
    for place in disc.get("places") or []:
        parts = [x for x in re.split(r"[\s,()-]+", str(place)) if len(x) >= 5]
        out.update(x.lower() for x in parts[:2])
    out -= {"museum", "natural", "history", "naturalis", "national", "institute", "university"}
    return {w for w in out if len(w) >= 4}


def on_topic(it, words):
    """The picture's title or description names the animal, a person or a place of the story - or the
    vision model saw the animal itself."""
    if any(w in f"{it.get('title') or ''} {it.get('tags') or ''}".lower() for w in words):
        return True
    animal = str((it.get("desc") or {}).get("animal") or "none")
    return animal not in ("none", "") and not animal.startswith("other")


def harvest(slug, topic=None):
    """Queries, candidates, descriptions -> episodes/<slug>/pool.json."""
    ep = ROOT / "episodes" / slug
    ep.mkdir(parents=True, exist_ok=True)
    if topic is None:
        q = json.loads((ROOT / "topics" / "queue.json").read_text(encoding="utf-8"))
        topic = next(t for t in q["topics"] if t["slug"] == slug)
    qp = ep / "pool_queries.json"
    qs = json.loads(qp.read_text(encoding="utf-8")) if qp.exists() else queries(topic)
    qp.write_text(json.dumps(qs, indent=1, ensure_ascii=False), encoding="utf-8")
    cand = {}

    def add(items):
        for it in items:
            cand.setdefault(it["id"], it)

    disc = qs.get("discovery") or {}
    terms = [a.get("latin") or a["key"] for a in qs["animals"] if a.get("extinct")][:3] + \
        (disc.get("places") or [])[:4] + (disc.get("people") or [])[:5]
    for term in dict.fromkeys(t for t in terms if t):
        add(commons_categories(term, "archive"))
    for q in disc.get("queries", [])[:14]:
        add(commons(q, "archive", "bitmap", 30, min_w=900))
        add(commons(q, "archive", "video", 10))
        add(archive(q, "archive", 4))
        add(loc(q, "archive", "photos", 15))
    for a in qs["animals"]:
        for q in (a.get("queries") or [])[:(5 if a.get("extinct") else 2)]:
            cat = "animal" if not a.get("extinct") else "taxon"
            add(pixabay(q, cat, 30))
            add(pexels(q, cat, 20))
            add(commons(q, cat, "video", 20))
            if a.get("extinct"):
                add(commons(q, cat, "bitmap", 30))
    for q in qs["science"][:8]:
        add(pixabay(q, "science", 30))
        add(pexels(q, "science", 20))
        add(commons(q, "science", "video", 15))
        add(commons(q, "science", "bitmap", 15))
        add(archive(q, "science", 4))
    for q in qs["landscapes"][:3]:
        add(pixabay(q, "landscape", 15))
        add(pexels(q, "landscape", 10))
    items = list(cand.values())
    log.info("%s: %d candidates (%s)", slug, len(items),
             {s: sum(1 for i in items if i["source"] == s) for s in {i["source"] for i in items}})
    desc = describe(items, topic.get("title", slug), qs["animals"], ep / "pool_describe.json")
    seen = used_elsewhere(slug)
    words = topic_words(qs)
    for it in items:
        it["desc"] = desc.get(it["id"])
        it["usable"] = usable(it) and it["id"] not in seen
        if it["usable"] and (it["desc"] or {}).get("category") in ARCHIVE and not on_topic(it, words):
            it["usable"] = False             # an old print from the same museum, but not of this story
            it["off_topic"] = True
        if it["id"] in seen:
            it["used_elsewhere"] = True
    pool = {"slug": slug, "animals": qs["animals"], "items": items}
    pool["material"] = material(pool)
    pool["undescribed"] = sum(1 for it in items if not it.get("desc"))
    (ep / "pool.json").write_text(json.dumps(pool, indent=1, ensure_ascii=False), encoding="utf-8")
    log.info("%s: material %s; %d not yet described", slug, pool["material"], pool["undescribed"])
    return pool


if __name__ == "__main__":
    harvest(sys.argv[1])
