"""The script queue: written weekly, drained on publication days.

The two halves of the pipeline are deliberately separated. Writing a script needs
Claude and takes minutes of model time; building a video needs a GPU and takes an
hour. Doing both in one run means a bad day for either service costs a video.

So a weekly job fills the queue with finished, fact-checked scripts, and each
publication day takes the oldest one and renders it without calling any language
model at all.
"""
import json
import re
from datetime import date, datetime, timezone
from pathlib import Path

from core.logger import get_logger

log = get_logger("queue")

ROOT = Path(__file__).resolve().parent.parent
QUEUE = ROOT / "queue"
JOURNAL = ROOT / "queue" / "journal.md"

# Publication days: Mon, Tue, Thu, Fri, Sat (weekday() 0=Mon).
PUBLISH_WEEKDAYS = (0, 1, 3, 4, 5)
PER_WEEK = len(PUBLISH_WEEKDAYS)
LOW_WATER = PER_WEEK          # fewer than a week's worth is a warning

STATUSES = ("ready-to-render", "rendered", "published", "failed")


def slugify(title):
    s = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")
    return re.sub(r"-{2,}", "-", s)[:60]


def episodes():
    """Every queued episode, in position order."""
    if not QUEUE.exists():
        return []
    out = []
    for d in sorted(QUEUE.iterdir()):
        meta = d / "episode.json"
        if d.is_dir() and meta.exists():
            try:
                data = json.loads(meta.read_text(encoding="utf-8"))
            except json.JSONDecodeError as e:
                log.error("unreadable episode.json in %s: %s", d.name, e)
                continue
            data["_dir"] = d
            out.append(data)
    return sorted(out, key=lambda e: (e.get("position", 9999), e.get("slug", "")))


def pending():
    return [e for e in episodes() if e.get("status") == "ready-to-render"]


def next_episode():
    """The episode a publication-day run should build, or None."""
    ready = pending()
    return ready[0] if ready else None


def next_position():
    used = [e.get("position", 0) for e in episodes()]
    return (max(used) + 1) if used else 1


def add(title, script_text, source, extra=None):
    """Write a new episode into the queue. Returns its directory."""
    slug = f"{next_position():03d}-{slugify(title)}"
    d = QUEUE / slug
    d.mkdir(parents=True, exist_ok=True)
    (d / "script.txt").write_text(script_text, encoding="utf-8")
    meta = {
        "slug": slug,
        "title": title,
        "status": "ready-to-render",
        "position": next_position(),
        "script_source": source,
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "words": len(re.findall(r"[A-Za-z0-9']+", script_text)),
    }
    meta.update(extra or {})
    (d / "episode.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False),
                                    encoding="utf-8")
    log.info("queued %s (%d words, via %s)", slug, meta["words"], source)
    return d


def set_status(episode_dir, status, **fields):
    if status not in STATUSES:
        raise ValueError(f"unknown status {status!r}")
    meta_path = Path(episode_dir) / "episode.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["status"] = status
    meta.update(fields)
    meta["updated"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False),
                         encoding="utf-8")
    return meta


def health():
    """Queue depth, in videos and in weeks of publishing."""
    ready = pending()
    weeks = len(ready) / PER_WEEK if PER_WEEK else 0
    return {
        "ready": len(ready),
        "weeks_of_stock": round(weeks, 2),
        "low": len(ready) < LOW_WATER,
        "empty": not ready,
        "titles": [e.get("title") for e in ready[:10]],
    }


def log_entry(message):
    """Append to the journal. This file is also what keeps the repository from
    going quiet: GitHub disables scheduled workflows on a public repo after 60
    days without a commit, and the weekly queue run commits this."""
    JOURNAL.parent.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    line = f"- **{stamp}** - {message}\n"
    if not JOURNAL.exists():
        JOURNAL.write_text("# Queue journal\n\n", encoding="utf-8")
    with JOURNAL.open("a", encoding="utf-8") as fh:
        fh.write(line)
    log.info("journal: %s", message)


def is_publication_day(when=None):
    when = when or date.today()
    return when.weekday() in PUBLISH_WEEKDAYS
