"""After a render: the episode goes to YouTube on the nearest free slot, or into the buffer.

    python automation/release.py record <slug> <run_id>   # merge job: buffer it, upload if a token exists
    python automation/release.py upload <slug>            # upload.yml: a buffered episode, once a token exists

state/episodes.json holds every episode: built (in the buffer, its files in the
render run's "release-<slug>" artifact), scheduled (uploaded, private with a
publishAt), or failed. Without a YouTube token nothing stops: finished episodes
wait in the buffer, and the watchdog uploads them as soon as YT_TOKEN_JSON is set.
"""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from core.logger import get_logger  # noqa: E402
from pipeline import youtube  # noqa: E402

log = get_logger("release")
STATE = ROOT / "state" / "episodes.json"


def rows():
    return json.loads(STATE.read_text(encoding="utf-8")) if STATE.exists() else []


def save(r):
    STATE.parent.mkdir(exist_ok=True)
    STATE.write_text(json.dumps(r, indent=1, ensure_ascii=False), encoding="utf-8")


def ahead(r=None, now=None):
    """Episodes still to come: built and waiting, or scheduled in the future."""
    now = now or datetime.now(timezone.utc)
    out = []
    for x in (r if r is not None else rows()):
        if x["status"] == "built":
            out.append(x)
        elif x["status"] == "scheduled" and datetime.fromisoformat(x["publish_at"].replace("Z", "+00:00")) > now:
            out.append(x)
    return out


def description(slug, release_dir):
    meta = json.loads((ROOT / "episodes" / slug / "meta.json").read_text(encoding="utf-8"))
    credits = Path(release_dir) / "description_credits.txt"
    text = meta["description"]
    if credits.exists():
        text += "\n\n" + credits.read_text(encoding="utf-8")
    return meta["title"], text, meta.get("tags", [])


def upload(slug, release_dir=None):
    release_dir = Path(release_dir or ROOT / "work" / slug / "release")
    r = rows()
    row = next((x for x in r if x["slug"] == slug), None)
    taken = [x["publish_at"] for x in r if x.get("publish_at")]
    slot = youtube.next_slot(taken)
    title, desc, tags = description(slug, release_dir)
    vid = youtube.upload(slug, release_dir, title, desc, tags, slot)
    saved = json.loads((ROOT / "episodes" / slug / "uploaded.json").read_text(encoding="utf-8"))
    if row is None:
        row = {"slug": slug}
        r.append(row)
    row.update({"status": "scheduled", "youtube_id": vid, "publish_at": saved["publish_at"],
                "uploaded_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")})
    save(r)
    log.info("%s scheduled for %s (%s)", slug, saved["publish_at"], vid)
    return 0


def record(slug, run_id):
    r = rows()
    row = next((x for x in r if x["slug"] == slug), None)
    if row is None:
        row = {"slug": slug}
        r.append(row)
    if (ROOT / "episodes" / slug / "uploaded.json").exists():
        log.info("%s was uploaded already; nothing to record", slug)
        return 0
    row.update({"status": "built", "run_id": str(run_id), "artifact": f"release-{slug}",
                "built_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")})
    save(r)
    if youtube.available():
        return upload(slug)
    log.info("%s is in the buffer (no YouTube token yet): artifact release-%s of run %s", slug, slug, run_id)
    return 0


if __name__ == "__main__":
    cmd, slug = sys.argv[1], sys.argv[2]
    sys.exit(record(slug, sys.argv[3]) if cmd == "record" else upload(slug))
