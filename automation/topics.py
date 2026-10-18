"""Keep the topic queue full and scripts written ahead, with nobody choosing.

    python automation/topics.py            (from .github/workflows/topics.yml)

1. Fewer than MIN_TOPICS queued topics: the backup list is used first, then a
   model proposes new internet mysteries. Each candidate must pass, in order:
     - the channel's filters (topics/queue.json "filters") as a word screen on
       its name and pitch, then the same filters as a yes/no review by a model;
     - an English Wikipedia article that exists;
     - a second, independent open source that mentions it (sources.gather);
     - not already in the queue, the backup list or the published list.
   Only then is it queued, with the two sources recorded.
2. Fewer than SCRIPTS_AHEAD queued topics with a script: the next ones in queue
   order get their scripts written now, so a build never waits on writing.
"""
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from core import llm  # noqa: E402
from core.logger import get_logger  # noqa: E402
from pipeline import sources  # noqa: E402

log = get_logger("topics")
QUEUE = ROOT / "topics" / "queue.json"
STATE = ROOT / "state" / "episodes.json"
MIN_TOPICS = 10
SCRIPTS_AHEAD = 3
PROPOSE = 20

# a first, mechanical screen; the model review below is the second
_BANNED = re.compile(
    r"\b(death|died|dies|dead|suicide|kill(ed|ing|er)?|murder\w*|victim\w*|homicide|"
    r"child|children|kid|kids|minor|teen\w*|girl|boy|abduct\w*|kidnap\w*|missing person|"
    r"rape|abuse\w*|shooting|massacre|terror\w*|trial|arrest\w*|convict\w*|charged|"
    r"gore|corpse|body|bodies)\b", re.I)

PROPOSER = """You suggest topics for a YouTube channel about real internet mysteries:
strange websites, ARGs, unexplained videos, hijacked broadcasts, lost media, anonymous
puzzles, cryptic accounts, odd signals. Each must have its own English Wikipedia
article. Never: real deaths or suicides, crime victims, named private individuals,
minors, active criminal cases, graphic violence. Return JSON only:
{"topics": [{"name": "...", "wikipedia": "<exact article title>",
"pitch": "<one sentence, what is strange about it>"}]}"""

REVIEW = """You check topic ideas against a channel's rules. A topic FAILS if it is
centred on, or cannot be told without: real deaths or suicides, crime victims, named
private individuals, minors, an active criminal case, graphic violence. Return JSON
only: {"results": [{"i": <index>, "ok": true|false, "why": "<few words>"}]}"""


def slugify(name):
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")[:40]


def load():
    return json.loads(QUEUE.read_text(encoding="utf-8"))


def save(q):
    QUEUE.write_text(json.dumps(q, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def published():
    rows = json.loads(STATE.read_text(encoding="utf-8")) if STATE.exists() else []
    return {r["slug"] for r in rows}


def queued(q):
    done = published()
    return [t for t in q["topics"] if t.get("status", "queued") == "queued" and t["slug"] not in done]


def known(q):
    names = set()
    for t in q["topics"] + q.get("backup", []):
        names |= {t["slug"], t["name"].lower(), t.get("wikipedia", "").lower()}
    return names


def verified(t, filters_ok=True):
    """The topic with its two sources, or None."""
    try:
        found = sources.gather(t)
    except Exception as e:                                       # noqa: BLE001
        log.info("%s: %s", t["name"], e)
        return None
    t["sources_checked"] = [s["url"] for s in found[:2]]
    t["status"] = "queued"
    return t


def refill(q):
    added = 0
    # the backup list first: those were chosen by hand
    while len(queued(q)) < MIN_TOPICS and q.get("backup"):
        b = q["backup"].pop(0)
        t = {"slug": b["slug"], "name": b["name"], "wikipedia": b["wikipedia"],
             "pitch": b.get("pitch", ""), "aliases": b.get("aliases", [])}
        if verified(t):
            q["topics"].append(t)
            added += 1
    rounds = 0
    while len(queued(q)) < MIN_TOPICS and rounds < 3:
        rounds += 1
        have = known(q)
        ask = (f"Already used, do not repeat: {', '.join(sorted(t['name'] for t in q['topics']))[:3000]}\n"
               f"Suggest {PROPOSE} new topics.")
        try:
            cands = llm.complete_json(ask, system=PROPOSER, max_tokens=3000, temperature=0.8).get("topics", [])
        except Exception as e:                                   # noqa: BLE001
            log.warning("no proposals (%s)", e)
            break
        cands = [c for c in cands if isinstance(c, dict) and c.get("name") and c.get("wikipedia")]
        cands = [c for c in cands if not (_BANNED.search(c["name"] + " " + c.get("pitch", "")))]
        cands = [c for c in cands if slugify(c["name"]) not in have and c["name"].lower() not in have
                 and c["wikipedia"].lower() not in have]
        if not cands:
            continue
        try:
            rev = llm.complete_json(json.dumps([{"i": i, "name": c["name"], "pitch": c.get("pitch", "")}
                                                for i, c in enumerate(cands)]),
                                    system=REVIEW, max_tokens=2000, temperature=0.0).get("results", [])
            ok = {int(r["i"]) for r in rev if r.get("ok") is True}
        except Exception as e:                                   # noqa: BLE001
            log.warning("review failed (%s); nothing added this round", e)
            continue
        for i, c in enumerate(cands):
            if i not in ok or len(queued(q)) >= MIN_TOPICS:
                continue
            t = {"slug": slugify(c["name"]), "name": c["name"], "wikipedia": c["wikipedia"],
                 "pitch": c.get("pitch", ""), "aliases": [], "added_by": "topics.py"}
            if verified(t):
                q["topics"].append(t)
                added += 1
                log.info("queued %s", t["name"])
    return added


def scripts_ahead(q):
    written = 0
    for t in queued(q):
        ready = [x for x in queued(q) if (ROOT / "queue" / x["slug"] / "script.txt").exists()]
        if len(ready) >= SCRIPTS_AHEAD:
            break
        if (ROOT / "queue" / t["slug"] / "script.txt").exists():
            continue
        r = subprocess.run([sys.executable, str(ROOT / "automation" / "write_script.py"), t["slug"]])
        if r.returncode == 0:
            written += 1
        else:
            log.warning("script for %s failed; it stays queued and is tried next time", t["slug"])
    return written


def main():
    q = load()
    added = refill(q)
    save(q)
    written = scripts_ahead(q)
    ready = [t["slug"] for t in queued(q) if (ROOT / "queue" / t["slug"] / "script.txt").exists()]
    print(json.dumps({"queued": len(queued(q)), "added": added, "scripts_written": written,
                      "scripts_ready": ready}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
