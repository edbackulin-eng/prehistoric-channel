"""The topic queue: which episode is built next, and the weekly top-up.

    python automation/topics.py --pick        # prints slug=<next> for GitHub, marks it building
    python automation/topics.py --refill      # Gemini proposes new topics until 20 wait in the queue

A topic is picked when no other is being built and fewer than `buffer`
(config/channel.json) episodes are ahead - built and waiting, or scheduled.
New topics come with the Wikipedia articles (de, en) the script is written and
checked from; each article is confirmed to exist before the topic is queued.
"""
import json
import os
import sys
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from core.logger import get_logger  # noqa: E402

log = get_logger("topics")
Q = ROOT / "topics" / "queue.json"
CFG = json.loads((ROOT / "config" / "channel.json").read_text(encoding="utf-8"))
UA = {"User-Agent": "prehistoric-channel/1.0 (vladgula557@gmail.com)"}

REFILL = """Schlage {n} neue Themen für einen deutschen YouTube-Kanal vor, der die GESCHICHTE BERÜHMTER
FOSSIL-ENTDECKUNGEN erzählt (10-15-Minuten-Dokus auf echtem Material): jede Folge ein bestimmter Fund - wer ihn
machte, wo und wann, die Expedition, die Abenteuer, der Streit der Forscher (z. B. die Knochenkriege, Mary Anning,
Stromers Spinosaurus, die Gobi-Expeditionen, der Berliner Brachiosaurus aus Tendaguru, Sue der T. rex). Nur Funde,
zu denen es ALTES ARCHIVMATERIAL gibt: historische Fotos der Grabung oder Expedition, Stiche, Porträts der Forscher,
Museumsstücke. Nicht doppelt zu diesen: {have}. Jedes Thema mit 2-4 Wikipedia-Artikeln (das Tier, die Entdecker,
die Expedition), deutsch und englisch, genau mit ihrem Titel. Antworte als JSON:
{{"topics": [{{"slug": "<kebab-case, deutsch>", "title": "<deutscher Titel>", "focus": "<ein Satz: wer fand was, wo, wann>",
"articles": {{"de": ["..."], "en": ["..."]}}}}]}}"""


def load():
    return json.loads(Q.read_text(encoding="utf-8"))


def save(q):
    Q.write_text(json.dumps(q, indent=1, ensure_ascii=False), encoding="utf-8")


def pick():
    from automation import release
    from automation import llm_budget
    q = load()
    d = llm_budget.deferred_now()
    if d:
        log.info("%s waits for the next quota day (%s, until %s)", d["slug"], d["why"], d["until"])
        return None
    building = next((t for t in q["topics"] if t["status"] == "building"), None)
    if building:
        log.info("resuming %s", building["slug"])
        return building["slug"]
    if len(release.ahead()) >= CFG["buffer"]:
        log.info("buffer full: %d ahead", len(release.ahead()))
        return None
    # owner 12.11: a topic is built only once there is material for it; one without goes, the next is tried
    real = CFG.get("format") == "real"
    tried = 0
    for t in [t for t in q["topics"] if t["status"] == "queued"]:
        if real:
            if tried >= REAL_TRIES:
                log.info("material of %d topics checked this run; the next ones on the next run", tried)
                return None
            tried += 1
            m = real_material(t)
        else:
            m = t.get("material") or material(t)
        t["material"] = m
        if m.get("ok") is None:
            log.warning("%s: material not yet known (%s); checked again on the next run", t["slug"], m)
            save(q)
            return None
        if m.get("ok") is False:
            t["status"] = "skipped-material"
            log.warning("%s skipped: too little material (%s)", t["slug"], m)
            save(q)
            continue
        t["status"] = "building"
        save(q)
        return t["slug"]
    return None


MATERIAL_MIN = {"commons_files": 100, "stock_clips": 10}     # either is enough
REAL_TRIES = 3                 # topics whose material one run looks at (each costs vision requests)
REAL_MIN_S = 10.5 * 60 / 0.72  # pipeline/realplan uses ~72 % of the pool's seconds (write_episode.FILL)
REAL_MAIN_MIN = 5              # pictures of the extinct animal itself (fossils, skeletons, paintings)
REAL_ARCHIVE_MIN = 8           # owner 15.11: archive pictures of THIS find (old photos, engravings, portraits)


def real_material(t):
    """owner 15.11: the real footage, stills and paintings with a confirmed licence, looked at by the
    vision model (pipeline/realpool.py) - enough for 10 minutes, each picture once, or the topic goes."""
    from pipeline import realpool
    pool = realpool.harvest(t["slug"], t)
    m = dict(pool["material"])
    n = len(pool["items"])
    if pool.get("undescribed", 0) > 0.3 * max(1, n):
        m["ok"] = None                 # the vision model's day ran out: not judged yet
    else:
        m["ok"] = m["seconds"] >= REAL_MIN_S and m["main_animal_items"] >= REAL_MAIN_MIN             and m.get("archive_items", 0) >= REAL_ARCHIVE_MIN
    m["need_s"] = round(REAL_MIN_S)
    return m


def material(t):
    """What the free sources hold for a topic, checked before its script is written (owner 12.11):
    media files on Wikimedia Commons (fossils, skeletons, illustrations, photos of the animal or its
    living relatives) and stock clips of 1280 px or more (Pixabay). The FLUX reconstructions and
    their 2.5D moves are made for every topic; these are what the film shows besides them."""
    import time as _t
    terms = ((t.get("articles") or {}).get("en") or [t["slug"].replace("-", " ")])[:2]
    files, clips, unknown = 0, 0, False
    for term in terms:
        for attempt in range(3):
            try:
                u = "https://commons.wikimedia.org/w/api.php?" + urllib.parse.urlencode(
                    {"action": "query", "list": "search", "srsearch": term, "srnamespace": 6, "format": "json",
                     "srlimit": 1})
                d = json.loads(urllib.request.urlopen(urllib.request.Request(u, headers=UA), timeout=30).read())
                files += int(d["query"]["searchinfo"]["totalhits"])
                break
            except Exception:                                        # noqa: BLE001
                _t.sleep(10 * (attempt + 1))
        else:
            unknown = True
        try:
            from pipeline import footage
            clips += len([c for c in footage.pixabay(term, per_page=40) if (c.get("width") or 0) >= 1280])
        except Exception:                                            # noqa: BLE001
            pass
    ok = None if unknown and not clips else (files >= MATERIAL_MIN["commons_files"] or clips >= MATERIAL_MIN["stock_clips"])
    return {"commons_files": files, "stock_clips": clips, "ok": ok, "terms": terms}


def done(slug, status="built"):
    q = load()
    for t in q["topics"]:
        if t["slug"] == slug:
            t["status"] = status
    save(q)


def _exists(lang, title):
    q = urllib.parse.urlencode({"action": "query", "titles": title, "format": "json", "redirects": 1})
    d = json.loads(urllib.request.urlopen(urllib.request.Request(f"https://{lang}.wikipedia.org/w/api.php?{q}",
                                                                 headers=UA), timeout=30).read())
    return all(int(k) > 0 for k in d["query"]["pages"])


def refill(target=20):
    from core import gemini
    q = load()
    waiting = [t for t in q["topics"] if t["status"] == "queued"]
    if len(waiting) >= target:
        log.info("%d topics waiting; nothing to add", len(waiting))
        return 0
    have = ", ".join(t["title"] for t in q["topics"])
    from core import textllm
    ans = textllm.ask(REFILL.format(n=target - len(waiting) + 3, have=have), step="topics", temperature=0.8,
                      max_tokens=3000)
    added = 0
    slugs = {t["slug"] for t in q["topics"]}
    for t in gemini.pick(ans, "topics"):
        if t["slug"] in slugs or len(waiting) + added >= target:
            continue
        arts = {lang: [a for a in t.get("articles", {}).get(lang, []) if _exists(lang, a)] for lang in ("de", "en")}
        if not arts["de"] and not arts["en"]:
            continue
        q["topics"].append({**t, "articles": arts, "status": "queued"})
        slugs.add(t["slug"])
        added += 1
    save(q)
    log.info("added %d topics", added)
    return 0


if __name__ == "__main__":
    if "--pick" in sys.argv:
        s = pick()
        out = os.environ.get("GITHUB_OUTPUT")
        if out:
            with open(out, "a", encoding="utf-8") as f:
                f.write(f"slug={s or ''}\n")
        print(f"slug={s or ''}")
    elif "--done" in sys.argv:
        i = sys.argv.index("--done")
        done(sys.argv[i + 1], *sys.argv[i + 2:i + 3])
    elif "--refill" in sys.argv:
        sys.exit(refill())
