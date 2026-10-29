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

REFILL = """Schlage {n} neue Themen für einen deutschen YouTube-Kanal über Urzeit-Tiere vor (25-Minuten-Dokus):
lebende Fossilien, Meeresreptilien, Megafauna, Rätsel der Aussterben, Dinosaurier, urzeitliche Fische,
Insekten, Eiszeit. Nicht doppelt zu diesen: {have}. Jedes Thema mit 2-4 Wikipedia-Artikeln, deutsch und
englisch, genau mit ihrem Titel. Antworte als JSON:
{{"topics": [{{"slug": "<kebab-case, deutsch>", "title": "<deutscher Titel>", "focus": "<ein Satz: worum es geht>",
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
    t = next((t for t in q["topics"] if t["status"] == "queued"), None)
    if t is None:
        return None
    t["status"] = "building"
    save(q)
    return t["slug"]


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
        done(sys.argv[sys.argv.index("--done") + 1])
    elif "--refill" in sys.argv:
        sys.exit(refill())
