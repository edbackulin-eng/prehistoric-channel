"""The pool of an interim episode (owner 18.11): the checked picture library first, real footage beside it.

    python -m pipeline.libpool <slug>        -> episodes/<slug>/pool.json

While the then-and-now format is being built the channel does not pause: episodes are cut, as the real
format does, from what is already there - the library's checked old illustrations, plates and fossils
(assets/library.json + library_checks.json, judged once, nothing sent to the vision model again), plus
what the real-format harvest finds for the story (Commons archive pictures, Pixabay footage of the living
relative, the landscapes), and the maps of the find (episodes/<slug>/interim.json "places").

The episode's animals, the people, places and searches of its story are written by hand in
episodes/<slug>/pool_queries.json (the harvest reads them instead of asking the text model), so the
animal keys are the library's own (mammuthus, plesiosaurus, ichthyosaurus ...).
"""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from core.logger import get_logger  # noqa: E402

log = get_logger("libpool")

OLD_KINDS = {"painting", "illustration", "plate", "engraving", "drawing", "print"}


def _category(item, verdict):
    """The library's look at a picture -> the real format's picture kind."""
    kind = str(verdict.get("kind") or item.get("kind") or "").lower()
    what = str(verdict.get("what") or "").lower()
    if item.get("kind") == "fossil" or kind == "fossil":
        return "skeleton" if re.search(r"skeleton|mounted|mount\b|cast", what) else "fossil"
    if kind in OLD_KINDS or item.get("kind") in ("illustration", "plate"):
        # an old picture is how people imagined the animal then - the real format's archive illustration
        return "archive_illustration" if verdict.get("style_era") == "old" else "palaeoart"
    if kind == "map" or item.get("kind") == "map":
        return "archive_illustration"
    return "palaeoart"


def _licence(item):
    lk = item.get("license_kind")
    lic = item.get("license") or {"pd": "Public domain", "cc0": "CC0", "cc-by": "CC BY"}.get(lk, str(lk))
    return lic, lk != "pd" and lk != "cc0"


def library_items(animals):
    """The library's usable pictures of the episode's animals, in the pool's shape."""
    from pipeline import library
    lib = json.loads((ROOT / "assets" / "library.json").read_text(encoding="utf-8"))["items"]
    checks = json.loads((ROOT / "assets" / "library_checks.json").read_text(encoding="utf-8"))
    keys = {a["key"]: a for a in animals}
    words = {k: {w.lower() for w in (k, a.get("latin") or "", a.get("name_de") or "", a.get("name_en") or "")
                 if len(w) >= 4} | {str(a.get("latin") or k).split()[0].lower()} for k, a in keys.items()}
    out = []
    for iid, v in checks.items():
        it = lib.get(iid)
        if not it or it["kind"] in ("scan3d", "silhouette") or not library.usable(it, v):
            continue
        animal = str(v.get("animal") or "none").lower()
        if animal not in keys:
            # 'several' and the like: the picture's own words decide which of ours it shows
            text = f"{v.get('what') or ''} {it.get('title') or ''} {' '.join(it.get('animals') or [])}".lower()
            hit = [k for k, ws in words.items() if any(w in text for w in ws)]
            if animal == "several" and hit:
                animal = hit[0]
            else:
                continue
        lic, attrib = _licence(it)
        cat = _category(it, v)
        out.append({
            "id": f"lib-{iid}", "source": it.get("source") or "Wikimedia Commons", "kind": "still",
            "query": "library", "cat": "archive" if cat == "archive_illustration" else "taxon",
            "title": it.get("title") or "", "page": it.get("page"), "url": it.get("file_url"),
            "full_url": it.get("file_url"), "thumb_url": it.get("thumb_url"),
            "width": it.get("width"), "height": it.get("height"),
            "tags": f"{it.get('title') or ''} {it.get('description') or ''} {it.get('year') or ''}"[:400],
            "author": it.get("author") or "unknown", "license": lic, "license_url": it.get("license_url") or "",
            "attribution_required": attrib, "library": True,
            "desc": {"category": cat, "animal": animal, "what": v.get("what"), "cgi": False,
                     "text": v.get("text") not in (None, False, "none", "no"), "quality": int(v.get("quality") or 3),
                     "parallax": bool(v.get("parallax"))},
            "usable": True})
    return out


def build(slug):
    ep = ROOT / "episodes" / slug
    qs = json.loads((ep / "pool_queries.json").read_text(encoding="utf-8"))
    interim = json.loads((ep / "interim.json").read_text(encoding="utf-8")) if (ep / "interim.json").exists() else {}
    lib = library_items(qs["animals"])
    log.info("%s: %d library pictures (%s)", slug, len(lib),
             {c: sum(1 for x in lib if x["desc"]["category"] == c) for c in {x["desc"]["category"] for x in lib}})
    pool = None
    try:
        from pipeline import realpool
        q = json.loads((ROOT / "topics" / "queue.json").read_text(encoding="utf-8"))
        topic = next((t for t in q["topics"] if t["slug"] == slug), {"slug": slug, "title": slug})
        pool = realpool.harvest(slug, topic)
    except Exception as e:                                               # noqa: BLE001
        log.warning("%s: the real-footage harvest failed (%s); the library alone", slug, str(e)[:200])
    if pool is None:
        pool = {"slug": slug, "animals": qs["animals"], "items": []}
    have = {x["id"] for x in pool["items"]}
    from pipeline import realpool
    seen = realpool.used_elsewhere(slug)
    for x in lib:
        if x["id"] not in have:
            x["usable"] = x["id"] not in seen
            pool["items"].append(x)
    pool["animals"] = qs["animals"]
    pool["places"] = interim.get("places") or []
    pool["interim"] = True
    pool["material"] = realpool.material(pool)
    # the real format caps stills against video; an interim film is mostly the library's stills (owner 18.11)
    pool["material"]["seconds"] = round(sum(realpool.seconds_of(x) for x in pool["items"] if x.get("usable")))
    (ep / "pool.json").write_text(json.dumps(pool, indent=1, ensure_ascii=False), encoding="utf-8")
    ok = [x for x in pool["items"] if x.get("usable")]
    log.info("%s: pool %d usable (%d library, %d video); material %s", slug, len(ok),
             sum(1 for x in ok if x.get("library")), sum(1 for x in ok if x["kind"] == "video"), pool["material"])
    return pool


if __name__ == "__main__":
    build(sys.argv[1])
