"""The picture library of the then-and-now format (owner 16.11): gathered once, checked once.

    python -m pipeline.library harvest            metadata of 400-500 pictures and 3D/silhouette sources
                                                  into assets/library.json (no picture is downloaded)
    python -m pipeline.library check [--max N]    looks at up to N pictures not judged yet, 10 to a Gemini
                                                  request, and keeps every answer in assets/library_checks.json;
                                                  it stops at the first 429 - the rest waits for the next day
    python -m pipeline.library report             what the library holds, per animal, era and kind

Episodes take their material from here; a picture is never sent to Gemini twice.

Licences, strictly (owner 16.11):
  public domain   only when it is public domain in the EU AND in the US: published 1930 or earlier (US) and
                  its artist dead 70 years (EU: died 1955 or earlier); an unknown artist only for pictures
                  of 1880 or earlier. US-government works (NPS, USGS, Smithsonian CC0) count as public domain.
                  Charles R. Knight (d. 1953) therefore only with works of 1930 or earlier.
  CC0             always
  CC BY           with the author and licence in every description
  never           NC, ND, BY-SA (share-alike would reach into the film), "all rights reserved", unknown
"""
import argparse
import hashlib
import json
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from core.logger import get_logger  # noqa: E402

log = get_logger("library")
LIB = ROOT / "assets" / "library.json"
CHECKS = ROOT / "assets" / "library_checks.json"
THUMBS = ROOT / "cache" / "library_thumbs"
UA = {"User-Agent": "UrzeitArchiv-library/1.0 (https://github.com/edbackulin-eng/prehistoric-channel)"}
COMMONS = "https://commons.wikimedia.org/w/api.php"

# where the old pictures are; depth = how far into sub-categories
CATEGORIES = [
    ("Heinrich Harder", 1), ("Charles R. Knight", 1), ("Paleoart", 2), ("Dinosaurs in art", 2),
    ("Animals of the past (1901)", 0), ("Extinct monsters (Hutchinson)", 0), ("Tiere der Vorwelt", 0),
    ("Benjamin Waterhouse Hawkins", 1), ("Crystal Palace Dinosaurs", 1), ("Édouard Riou", 1),
    ("Paleontological illustrations", 1), ("Fossil illustrations", 1), ("Mary Anning", 1),
    ("Plesiosaurus", 1), ("Ichthyosaurus", 1), ("Mosasaurus hoffmannii", 1), ("Archaeopteryx", 1),
    ("Iguanodon", 1), ("Megalosaurus", 1), ("Megatherium", 1), ("Mammuthus primigenius", 1),
    ("Smilodon", 1), ("Trilobita", 1), ("Dunkleosteus", 1), ("Burgess Shale", 1),
    ("Tyrannosaurus rex", 1), ("Spinosaurus", 1), ("Triceratops", 1), ("Stegosaurus", 1),
    ("Brontosaurus", 1), ("Diplodocus", 1), ("Pteranodon", 1), ("Meganeura", 1),
]
# artists whose death year decides the EU term (70 years); Burian (d. 1981) and later are never PD
ARTISTS = {
    "heinrich harder": 1935, "charles r. knight": 1953, "charles robert knight": 1953, "knight": 1953,
    "benjamin waterhouse hawkins": 1894, "waterhouse hawkins": 1894, "édouard riou": 1900, "edouard riou": 1900,
    "riou": 1900, "joseph smit": 1929, "john martin": 1854, "george scharf": 1860, "henry de la beche": 1855,
    "camille flammarion": 1925, "louis figuier": 1894, "othniel charles marsh": 1899, "edward drinker cope": 1897,
    "georges cuvier": 1832, "gideon mantell": 1852, "richard owen": 1892, "wilhelm kuhnert": 1926,
    "ernst haeckel": 1919, "alice b. woodward": 1951, "joseph wolf": 1899, "frank bond": 1940,
    "erwin s. christman": 1921, "robert bruce horsfall": 1948, "zdeněk burian": 1981, "zdenek burian": 1981,
    "rudolph zallinger": 1995, "frederic a. lucas": 1929, "lucas, frederic": 1929,
    "henry neville hutchinson": 1927, "hutchinson, h. n": 1927, "william diller matthew": 1930,
    "henry fairfield osborn": 1935, "osborn, henry fairfield": 1935, "carl dahl": 1950,
}
EU_DEAD_BY = 1955          # 70 years before 2026
US_PUBLISHED_BY = 1930     # works published 1930 or earlier are PD in the US in 2026
UNKNOWN_ARTIST_BY = 1880

# animals of the topic queue and more: German, Latin, English names -> the key used in tags
ANIMALS = {
    "tyrannosaurus": ["tyrannosaurus", "t. rex", "t-rex", "trex"], "triceratops": ["triceratops"],
    "stegosaurus": ["stegosaurus"], "brontosaurus": ["brontosaurus", "apatosaurus"],
    "diplodocus": ["diplodocus"], "iguanodon": ["iguanodon"], "megalosaurus": ["megalosaurus"],
    "mosasaurus": ["mosasaurus", "mosasaur", "mosasaurier", "maastricht"], "plesiosaurus": ["plesiosaur"],
    "ichthyosaurus": ["ichthyosaur", "fischsaurier"], "archaeopteryx": ["archaeopteryx", "urvogel"],
    "pteranodon": ["pteranodon", "pterodactyl", "pterosaur", "flugsaurier", "rhamphorhynchus"],
    "quetzalcoatlus": ["quetzalcoatlus"], "spinosaurus": ["spinosaurus"], "megalodon": ["megalodon", "carcharodon"],
    "mammoth": ["mammoth", "mammut", "mammuthus"], "mastodon": ["mastodon"], "smilodon": ["smilodon", "sabre", "saber",
                                                                                          "säbelzahn"],
    "megatherium": ["megatherium", "ground sloth", "riesenfaultier"], "glyptodon": ["glyptodon"],
    "woolly_rhino": ["coelodonta", "woolly rhinoceros", "wollnashorn"], "cave_bear": ["cave bear", "höhlenbär",
                                                                                        "ursus spelaeus"],
    "irish_elk": ["megaloceros", "irish elk", "riesenhirsch"], "dunkleosteus": ["dunkleosteus", "placoderm",
                                                                             "panzerfisch", "dinichthys"],
    "trilobite": ["trilobit"], "meganeura": ["meganeura", "griffinfly"], "arthropleura": ["arthropleura"],
    "dimetrodon": ["dimetrodon"], "titanoboa": ["titanoboa"], "coelacanth": ["coelacanth", "latimeria",
                                                                             "quastenflosser"],
    "hadrosaur": ["hadrosaur", "trachodon", "edmontosaurus", "anatosaurus"], "allosaurus": ["allosaurus"],
    "ceratosaurus": ["ceratosaurus"], "laelaps": ["laelaps", "dryptosaurus"], "hylaeosaurus": ["hylaeosaurus"],
    "anomalocaris": ["anomalocaris"], "elasmosaurus": ["elasmosaurus"], "dinotherium": ["dinotherium",
                                                                                        "deinotherium"],
    "uintatherium": ["uintatherium"], "basilosaurus": ["basilosaurus", "zeuglodon"],
}
ERAS = {
    "cambrian": ["cambrian", "kambrium", "burgess"], "devonian": ["devon"], "carboniferous": ["carbonifer", "karbon"],
    "permian": ["permian", "perm "], "triassic": ["triass", "trias"], "jurassic": ["jurass", "jura"],
    "cretaceous": ["cretac", "kreide"], "paleogene": ["eocene", "paleocene", "eozän", "paläozän"],
    "ice_age": ["pleistocene", "ice age", "eiszeit", "pleistozän", "quaternary"],
}
ERA_OF = {"tyrannosaurus": "cretaceous", "triceratops": "cretaceous", "mosasaurus": "cretaceous",
          "spinosaurus": "cretaceous", "quetzalcoatlus": "cretaceous", "pteranodon": "cretaceous",
          "hadrosaur": "cretaceous", "laelaps": "cretaceous", "elasmosaurus": "cretaceous", "iguanodon": "cretaceous",
          "stegosaurus": "jurassic", "brontosaurus": "jurassic", "diplodocus": "jurassic", "allosaurus": "jurassic",
          "ceratosaurus": "jurassic", "archaeopteryx": "jurassic", "megalosaurus": "jurassic",
          "plesiosaurus": "jurassic", "ichthyosaurus": "jurassic", "hylaeosaurus": "cretaceous",
          "dimetrodon": "permian", "meganeura": "carboniferous", "arthropleura": "carboniferous",
          "dunkleosteus": "devonian", "trilobite": "cambrian", "anomalocaris": "cambrian",
          "titanoboa": "paleogene", "uintatherium": "paleogene", "basilosaurus": "paleogene",
          "mammoth": "ice_age", "mastodon": "ice_age", "smilodon": "ice_age", "megatherium": "ice_age",
          "glyptodon": "ice_age", "woolly_rhino": "ice_age", "cave_bear": "ice_age", "irish_elk": "ice_age",
          "dinotherium": "ice_age", "megalodon": "paleogene"}


# ------------------------------------------------------------------ http
def _get(url, params=None, tries=6):
    if params:
        url += "?" + urllib.parse.urlencode(params)
    for k in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60) as r:
                return json.loads(r.read().decode("utf-8"))
        except (urllib.error.HTTPError, urllib.error.URLError, json.JSONDecodeError, TimeoutError) as e:
            code = getattr(e, "code", None)
            if code in (400, 401, 403, 404):
                raise
            time.sleep(5 * (k + 1))
    raise RuntimeError(f"no answer from {url[:120]}")


def _plain(html):
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", "", html or "")).strip()


# ------------------------------------------------------------------ licences
def _year(text):
    years = [int(y) for y in re.findall(r"\b(1[5-9]\d\d|20[0-2]\d)\b", text or "")]
    return min(years) if years else None


def _artist_death(author):
    a = (author or "").lower()
    for name, died in ARTISTS.items():
        if name in a:
            return died
    return None


def licence_ok(lic, author, year, credit=""):
    """(ok, kind, why): kind is 'pd', 'cc0' or 'cc-by'."""
    l = (lic or "").lower()
    if re.search(r"\bnc\b|non.?commercial|\bnd\b|no.?deriv|by-sa|sharealike|share alike|all rights", l):
        return False, None, f"licence {lic}"
    if "cc0" in l or "cc-zero" in l:
        return True, "cc0", ""
    if re.match(r"^cc[ -]by( |-|$)", l):
        return True, "cc-by", ""
    # BHL's Flickr scans say "no known copyright restrictions": held to the same EU and US test as PD
    if "public domain" in l or l.startswith("pd") or "no known copyright" in l or l == "no restrictions":
        if re.search(r"pd-usgov|us government|nps|usgs|pd-us-gov", l + " " + (credit or "").lower()):
            return True, "pd", ""
        died = _artist_death(author)
        if died is not None and died > EU_DEAD_BY:
            return False, None, f"artist died {died}: not public domain in the EU"
        if not year:
            return False, None, "public domain without a year"
        if year > US_PUBLISHED_BY:
            return False, None, f"{year}: not public domain in the US"
        if died is None and year > UNKNOWN_ARTIST_BY:
            return False, None, f"unknown artist, {year}: EU term not provable"
        return True, "pd", ""
    return False, None, f"licence {lic or 'unknown'}"


# ------------------------------------------------------------------ tags
def tags_of(text):
    t = (text or "").lower()
    animals = sorted(k for k, names in ANIMALS.items() if any(n in t for n in names))
    eras = {k for k, words in ERAS.items() if any(w in t for w in words)} | {ERA_OF[a] for a in animals if a in ERA_OF}
    return animals, sorted(eras)


def kind_of(text):
    t = (text or "").lower()
    if re.search(r"skeleton|skelett|mounted|museum|exhibit|fossil|specimen|bone|skull|schädel|tooth|zahn", t):
        return "fossil"
    if re.search(r"plate|tafel|lithograph|engraving|stich|tab\.|pl\.", t):
        return "plate"
    if re.search(r"map|karte", t):
        return "map"
    return "illustration"


# ------------------------------------------------------------------ Commons
def _members(cat, depth, seen):
    if cat in seen or depth < 0:
        return []
    seen.add(cat)
    files, cont = [], {}
    while True:
        d = _get(COMMONS, {"action": "query", "format": "json", "generator": "categorymembers",
                           "gcmtitle": f"Category:{cat}", "gcmtype": "file", "gcmlimit": 200,
                           "prop": "imageinfo", "iiprop": "url|size|extmetadata|mime",
                           "iiurlwidth": 640, **cont})
        for pg in ((d.get("query") or {}).get("pages") or {}).values():
            pg["_cat"] = cat
            files.append(pg)
        time.sleep(1.0)
        if "continue" not in d:
            break
        cont = d["continue"]
    if depth > 0:
        d = _get(COMMONS, {"action": "query", "format": "json", "list": "categorymembers",
                           "cmtitle": f"Category:{cat}", "cmtype": "subcat", "cmlimit": 200})
        time.sleep(1.0)
        for c in (d.get("query") or {}).get("categorymembers", []):
            files += _members(c["title"].split(":", 1)[1], depth - 1, seen)
    return files


def commons_items():
    seen_cats, out, seen = set(), {}, set()
    for cat, depth in CATEGORIES:
        try:
            pages = _members(cat, depth, seen_cats)
        except Exception as e:                                           # noqa: BLE001
            log.warning("category %s: %s", cat, e)
            continue
        kept = 0
        for p in pages:
            if p.get("title") in seen:
                continue
            seen.add(p.get("title"))
            info = (p.get("imageinfo") or [{}])[0]
            if not (info.get("mime") or "").startswith("image/") or info.get("mime") == "image/gif":
                continue
            if min(info.get("width") or 0, info.get("height") or 0) < 700:
                continue
            m = info.get("extmetadata") or {}
            author = _plain(m.get("Artist", {}).get("value"))[:200]
            lic = _plain(m.get("LicenseShortName", {}).get("value"))
            desc = _plain(m.get("ImageDescription", {}).get("value"))[:600]
            datestr = _plain(m.get("DateTimeOriginal", {}).get("value"))
            credit = _plain(m.get("Credit", {}).get("value"))[:200]
            year = _year(datestr) or _year(desc) or _year(p["title"])
            ok, lk, why = licence_ok(lic, author, year, credit)
            if not ok:
                continue
            text = f"{p['title']} {desc} {p.get('_cat', '')} {cat}"
            animals, eras = tags_of(text)
            # an artist's or palaeo-art category is kept untagged: the Gemini look names the animal
            if not animals and not eras and not re.search(r"harder|knight|hawkins|riou|paleoart|in art", cat, re.I):
                continue
            iid = "commons-" + hashlib.sha1(p["title"].encode()).hexdigest()[:10]
            out[iid] = {"id": iid, "source": "Wikimedia Commons", "kind": kind_of(text), "title": p["title"],
                        "page": info.get("descriptionurl"), "file_url": info.get("url"),
                        "thumb_url": info.get("thumburl"), "width": info.get("width"), "height": info.get("height"),
                        "author": author or "unknown", "license": lic, "license_kind": lk,
                        "license_url": m.get("LicenseUrl", {}).get("value", ""), "year": year,
                        "animals": animals, "eras": eras, "found_in": cat, "description": desc[:300]}
            kept += 1
        log.info("Commons %s: %d pages, %d kept (library %d)", cat, len(pages), kept, len(out))
    return out


# ------------------------------------------------------------------ 3D and silhouettes (catalogued; files at build time)
def smithsonian_items():
    out = {}
    for q in ("dinosaur", "fossil", "Triceratops", "Stegosaurus", "mammoth", "mastodon", "Smilodon", "whale fossil",
              "ichthyosaur", "plesiosaur", "trilobite", "skull fossil"):
        try:
            d = _get("https://3d-api.si.edu/api/v1.0/content/file/search", {"q": q, "file_type": "glb", "rows": 200})
        except Exception as e:                                           # noqa: BLE001
            log.warning("Smithsonian %s: %s", q, e)
            continue
        for r in d.get("rows", []):
            if not isinstance(r, dict):
                continue
            c = r.get("content") or {}
            usage = (c.get("usage") or {}).get("access") or ""
            if usage.upper() != "CC0":
                continue
            title = r.get("title") or ""
            animals, eras = tags_of(title + " " + q)
            iid = "si3d-" + hashlib.sha1((c.get("uri") or title).encode()).hexdigest()[:10]
            if iid in out:
                continue
            out[iid] = {"id": iid, "source": "Smithsonian Open Access 3D", "kind": "scan3d", "title": title,
                        "file_url": c.get("uri"), "page": "https://3d.si.edu", "author": "Smithsonian Institution",
                        "license": "CC0", "license_kind": "cc0", "license_url": "https://creativecommons.org/publicdomain/zero/1.0/",
                        "animals": animals, "eras": eras, "quality": c.get("quality"), "model_type": c.get("model_type")}
        time.sleep(1)
    log.info("Smithsonian 3D: %d CC0 models", len(out))
    return out


def sketchfab_items():
    """Museum scans on Sketchfab under CC0 or CC BY only (owner 16.11); downloadable ones."""
    out = {}
    for q in ("dinosaur skeleton", "dinosaur skull", "fossil skull", "mosasaurus", "tyrannosaurus skull",
              "triceratops skull", "mammoth skeleton", "smilodon skull", "ichthyosaur", "plesiosaur", "archaeopteryx",
              "trilobite", "megalodon tooth", "spinosaurus", "stegosaurus", "dunkleosteus", "megatherium"):
        for lic in ("cc0", "by"):
            try:
                d = _get("https://api.sketchfab.com/v3/search", {"type": "models", "q": q, "license": lic,
                                                                 "downloadable": "true", "count": 24})
            except Exception as e:                                       # noqa: BLE001
                log.warning("Sketchfab %s: %s", q, e)
                continue
            for m in d.get("results", []):
                label = ((m.get("license") or {}).get("label") or "").strip().lower()
                ls = {"cc attribution": "by", "cc0 public domain": "cc0", "public domain": "cc0"}.get(label)
                if ls is None:                       # NC, ND, SA and anything else: never (owner 16.11)
                    continue
                text = f"{m.get('name')} {m.get('description', '')[:300]} {' '.join(t.get('name', '') for t in m.get('tags', []))}"
                animals, eras = tags_of(text + " " + q)
                if not animals:
                    continue
                iid = "skfb-" + m["uid"][:12]
                user = m.get("user") or {}
                out[iid] = {"id": iid, "source": "Sketchfab", "kind": "scan3d", "title": m.get("name"),
                            "page": m.get("viewerUrl"), "uid": m["uid"],
                            "author": user.get("displayName") or user.get("username") or "unknown",
                            "author_url": user.get("profileUrl"), "license": "CC0" if ls == "cc0" else "CC BY 4.0",
                            "license_kind": "cc0" if ls == "cc0" else "cc-by",
                            "license_url": "https://creativecommons.org/publicdomain/zero/1.0/" if ls == "cc0"
                            else "https://creativecommons.org/licenses/by/4.0/",
                            "faces": m.get("faceCount"), "animals": animals, "eras": eras,
                            "thumb_url": max((m.get("thumbnails") or {}).get("images") or [{}],
                                             key=lambda i: i.get("width") or 0).get("url")}
            time.sleep(1)
    log.info("Sketchfab: %d CC0/CC BY models", len(out))
    return out


def phylopic_items():
    """Silhouettes for the size comparisons; CC0 and CC BY only."""
    out = {}
    try:
        build = _get("https://api.phylopic.org/")["build"]
    except Exception as e:                                               # noqa: BLE001
        log.warning("PhyloPic: %s", e)
        return out
    for key, names in ANIMALS.items():
        for name in names[:1]:
            try:
                d = _get("https://api.phylopic.org/images", {"build": build, "filter_name": name.split()[0],
                                                              "embed_items": "true", "page": 0})
            except Exception:                                            # noqa: BLE001
                continue
            for it in ((d.get("_embedded") or {}).get("items") or [])[:6]:
                links = it.get("_links") or {}
                lic = (links.get("license") or {}).get("href", "")
                if "/by-nc" in lic or "/by-sa" in lic or "/by-nd" in lic:
                    continue
                lk = "cc0" if ("publicdomain" in lic) else ("cc-by" if "/by/" in lic else None)
                if not lk:
                    continue
                svg = (links.get("vectorFile") or {}).get("href")
                iid = "phylo-" + it["uuid"][:12]
                out[iid] = {"id": iid, "source": "PhyloPic", "kind": "silhouette", "title": it.get("attribution") or key,
                            "file_url": svg, "page": f"https://www.phylopic.org/images/{it['uuid']}",
                            "author": it.get("attribution") or "unknown", "license": "CC0" if lk == "cc0" else "CC BY",
                            "license_kind": lk, "license_url": lic, "animals": [key],
                            "eras": [ERA_OF[key]] if key in ERA_OF else []}
            time.sleep(0.5)
    log.info("PhyloPic: %d silhouettes", len(out))
    return out


# ------------------------------------------------------------------ the library
def load():
    return json.loads(LIB.read_text(encoding="utf-8")) if LIB.exists() else {"items": {}}


def save(lib):
    LIB.parent.mkdir(parents=True, exist_ok=True)
    LIB.write_text(json.dumps(lib, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")


def harvest():
    lib = load()
    items = lib["items"]
    for fn in (commons_items, smithsonian_items, sketchfab_items, phylopic_items):
        try:
            new = fn()
        except Exception as e:                                           # noqa: BLE001
            log.warning("%s: %s", fn.__name__, e)
            continue
        for k, v in new.items():
            items.setdefault(k, v)                     # an item once gathered keeps its record
    lib["harvested_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    save(lib)
    report(lib)


# ------------------------------------------------------------------ the one-time Gemini look
CHECK_PROMPT = """Each numbered picture is from an archive for a German documentary series "So stellte man sich
... vor - und so sah er wirklich aus" (how prehistoric animals were imagined then, and what we know now).
For each picture answer:
 "animal": the prehistoric animal shown, a Latin genus in lower case ("tyrannosaurus", "mosasaurus", "mammuthus"),
           "several" for a scene with several kinds, "none" if no prehistoric animal is shown;
 "kind": one of "painting", "drawing", "plate" (scientific plate / lithograph of bones or animals),
         "fossil_photo", "skeleton_photo", "museum_photo", "sculpture", "map", "other";
 "style_era": "old" if it looks made before about 1930, else "modern";
 "text": "none", "edge" (a caption or number only near an edge, croppable) or "inside" (lettering over the picture);
 "watermark": true/false; "people_close": true if a person's face fills much of the frame;
 "parallax": true if a clear subject stands before a background (good for a 2.5D camera move);
 "quality": 1-5 (sharpness, scan damage, how well it reads on a phone);
 "what": 6-14 English words of what it shows.
Answer as JSON: {"items": [{"i": 0, "animal": "...", "kind": "...", "style_era": "old", "text": "none",
"watermark": false, "people_close": false, "parallax": true, "quality": 4, "what": "..."}]}"""


def _thumb(item):
    THUMBS.mkdir(parents=True, exist_ok=True)
    dest = THUMBS / f"{item['id']}.jpg"
    if not dest.exists():
        with urllib.request.urlopen(urllib.request.Request(item["thumb_url"], headers=UA), timeout=60) as r:
            dest.write_bytes(r.read())
    return dest


def checks():
    return json.loads(CHECKS.read_text(encoding="utf-8")) if CHECKS.exists() else {}


def check(max_items=300, batch=10):
    """Up to max_items pictures not judged yet; each answer kept at once. Stops at the day's quota."""
    from core import gemini
    lib, done = load(), checks()
    todo = [it for it in lib["items"].values() if it["id"] not in done and it.get("thumb_url")
            and it["kind"] not in ("scan3d", "silhouette")]
    todo.sort(key=lambda it: (-len(it.get("animals") or []), it["id"]))
    todo = todo[:max_items]
    log.info("pictures to look at: %d of %d not judged yet", len(todo),
             sum(1 for it in lib["items"].values() if it["id"] not in done and it["kind"] not in ("scan3d", "silhouette")))
    looked = 0
    for k in range(0, len(todo), batch):
        part, paths = [], []
        for it in todo[k:k + batch]:
            try:
                paths.append(str(_thumb(it)))
                part.append(it)
            except Exception as e:                                       # noqa: BLE001
                done[it["id"]] = {"error": f"thumbnail: {str(e)[:80]}"}
        if not part:
            continue
        try:
            ans = gemini.ask(CHECK_PROMPT, images=paths, models=("gemini-flash-lite-latest",), temperature=0,
                             step="library_check")
        except Exception as e:                                           # noqa: BLE001
            log.warning("Gemini stopped the check (%s); %d looked at, the rest waits for the next day",
                        str(e)[:160], looked)
            break
        res = {int(r.get("i", -1)): r for r in (ans.get("items") or []) if isinstance(r, dict)}
        for j, it in enumerate(part):
            if j in res:
                done[it["id"]] = {**res[j], "checked_at": time.strftime("%Y-%m-%d")}
                looked += 1
        CHECKS.write_text(json.dumps(done, indent=0, ensure_ascii=False), encoding="utf-8")
    CHECKS.write_text(json.dumps(done, indent=0, ensure_ascii=False), encoding="utf-8")
    log.info("looked at %d pictures this run; %d judged in all", looked, len(done))
    return looked


def usable(item, verdict):
    """A picture an episode may use: judged, an animal, no lettering over it, no watermark, readable."""
    if item["kind"] in ("scan3d", "silhouette"):
        return True
    v = verdict or {}
    return bool(v) and not v.get("error") and v.get("animal") not in (None, "none") and v.get("text") != "inside" \
        and not v.get("watermark") and not v.get("people_close") and int(v.get("quality") or 0) >= 3


def report(lib=None):
    from collections import Counter
    lib, done = lib or load(), checks()
    items = list(lib["items"].values())
    by_kind = Counter(it["kind"] for it in items)
    by_src = Counter(it["source"] for it in items)
    by_lic = Counter(it["license_kind"] for it in items)
    animals = Counter(a for it in items for a in it.get("animals") or [])
    ok = sum(1 for it in items if it["kind"] not in ("scan3d", "silhouette") and usable(it, done.get(it["id"])))
    print(f"library: {len(items)} items; kinds {dict(by_kind)}; sources {dict(by_src)}; licences {dict(by_lic)}")
    print(f"judged {len(done)}, usable pictures so far {ok}")
    print("animals:", ", ".join(f"{a} {n}" for a, n in animals.most_common(40)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("harvest", "check", "report"))
    ap.add_argument("--max", type=int, default=300)
    a = ap.parse_args()
    if a.cmd == "harvest":
        harvest()
    elif a.cmd == "check":
        check(a.max)
    else:
        report()


if __name__ == "__main__":
    main()
