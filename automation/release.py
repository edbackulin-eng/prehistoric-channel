"""After a render: the episode goes to YouTube, or into the buffer.

    python automation/release.py record <slug> <run_id>   # merge job: buffer it, upload if a token exists
    python automation/release.py upload <slug>            # upload.yml: a buffered episode, once a token exists
    python automation/release.py approve <slug>           # approve.yml: the nearest free slot for an approved one
    python automation/release.py reject <slug> <why>      # the video stays private; the episode is built again

state/episodes.json holds every episode:
  built              rendered, QA passed, in the buffer (files in the render run's "release-<slug>" artifact)
  awaiting_approval  uploaded private without publishAt; waits for approve.yml (config release.manual_left)
  scheduled          uploaded private with a publishAt
  rejected-rebuild   its video stays private and unscheduled; a rebuild may upload one new video, once
  qa-failed          the QA gate (automation/qa.py) refused it; nothing was uploaded

Nothing is uploaded without a passing QA report (release/qa.json). One episode,
one video: pipeline/youtube.py asks the channel first, and the id is committed
the moment YouTube returns it (automation/gitsave.py), so no rerun uploads twice.
"""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from automation import gitsave  # noqa: E402
from core.logger import get_logger  # noqa: E402
from pipeline import youtube  # noqa: E402

log = get_logger("release")
STATE = ROOT / "state" / "episodes.json"
CFG_PATH = ROOT / "config" / "channel.json"


def rows():
    return json.loads(STATE.read_text(encoding="utf-8")) if STATE.exists() else []


def save(r):
    STATE.parent.mkdir(exist_ok=True)
    STATE.write_text(json.dumps(r, indent=1, ensure_ascii=False), encoding="utf-8")


def row_of(r, slug):
    row = next((x for x in r if x["slug"] == slug), None)
    if row is None:
        row = {"slug": slug}
        r.append(row)
    return row


def now_s():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def manual_left():
    return int(json.loads(CFG_PATH.read_text(encoding="utf-8")).get("release", {}).get("manual_left", 0))


def ahead(r=None, now=None):
    """Episodes still to come: built and waiting, uploaded and waiting for approval, or scheduled ahead."""
    now = now or datetime.now(timezone.utc)
    out = []
    for x in (r if r is not None else rows()):
        if x["status"] in ("built", "awaiting_approval"):
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


def qa_passed(release_dir):
    f = Path(release_dir) / "qa.json"
    if not f.exists():
        return False, ["no QA report (release/qa.json)"]
    q = json.loads(f.read_text(encoding="utf-8"))
    return bool(q.get("pass")), [c["name"] + ": " + c.get("detail", "") for c in q.get("checks", [])
                                 if not c.get("ok")]


def upload(slug, release_dir=None):
    release_dir = Path(release_dir or ROOT / "work" / slug / "release")
    ok, why = qa_passed(release_dir)
    if not ok:
        raise SystemExit(f"{slug}: QA did not pass, not uploading: {why}")
    r = rows()
    row = row_of(r, slug)
    manual = manual_left() > 0
    slot = None if manual else youtube.next_slot([x["publish_at"] for x in r if x.get("publish_at")])
    title, desc, tags = description(slug, release_dir)
    marker = ROOT / "episodes" / slug / "uploaded.json"

    def keep(vid):                      # committed before thumbnail and subtitles are even tried
        row.update({"status": "awaiting_approval" if manual else "scheduled", "youtube_id": vid,
                    "publish_at": None if manual else slot.strftime("%Y-%m-%dT%H:%M:%SZ"), "uploaded_at": now_s()})
        save(r)
        gitsave.save([marker, STATE], f"Uploaded {slug} ({vid})")

    vid = youtube.upload(slug, release_dir, title, desc, tags, slot, ignore=row.get("rejected_videos", []),
                         on_id=keep)
    saved = json.loads(marker.read_text(encoding="utf-8"))
    row["extras"] = saved.get("extras")
    save(r)
    gitsave.save([marker, STATE], f"Uploaded {slug}: extras")
    log.info("%s: %s (%s)", slug, row["status"], vid)
    return 0


def record(slug, run_id):
    r = rows()
    row = row_of(r, slug)
    if (ROOT / "episodes" / slug / "uploaded.json").exists():
        log.info("%s was uploaded already; nothing to record", slug)
        return 0
    rel = ROOT / "work" / slug / "release"
    ok, why = qa_passed(rel)
    row.update({"run_id": str(run_id), "artifact": f"release-{slug}", "built_at": now_s()})
    if not ok:
        row.update({"status": "qa-failed", "qa_failed": why})
        save(r)
        log.error("%s: QA failed, kept back: %s", slug, why)
        return 2
    row["status"] = "built"
    save(r)
    if youtube.available():
        return upload(slug)
    log.info("%s is in the buffer (no YouTube token yet): artifact release-%s of run %s", slug, slug, run_id)
    return 0


def approve(slug):
    """An episode uploaded without a date gets the nearest free slot; one manual release fewer to go."""
    r = rows()
    row = next(x for x in r if x["slug"] == slug)
    if row["status"] != "awaiting_approval":
        raise SystemExit(f"{slug} is {row['status']}, not awaiting approval")
    slot = youtube.next_slot([x["publish_at"] for x in r if x.get("publish_at")])
    st = youtube.set_schedule(row["youtube_id"], slot)
    row.update({"status": "scheduled", "publish_at": st.get("publishAt"), "approved_at": now_s()})
    save(r)
    import re
    left = max(0, manual_left() - 1)                  # the file keeps its layout: only the number changes
    text = CFG_PATH.read_text(encoding="utf-8")
    CFG_PATH.write_text(re.sub(r'("manual_left":\s*)\d+', rf"\g<1>{left}", text, count=1), encoding="utf-8")
    gitsave.save([STATE, CFG_PATH], f"Approved {slug}: {row['publish_at']}")
    log.info("%s approved: %s; manual releases left %d", slug, row["publish_at"], left)
    return 0


def reject(slug, why):
    """The episode's video stays private and unscheduled; the episode is built again and may upload once more."""
    r = rows()
    row = row_of(r, slug)
    marker = ROOT / "episodes" / slug / "uploaded.json"
    vids = list(row.get("rejected_videos", []))
    if marker.exists():
        vids.append(json.loads(marker.read_text(encoding="utf-8"))["youtube_id"])
        marker.unlink()
    if row.get("youtube_id"):
        vids.append(row.pop("youtube_id"))
    row.update({"status": "rejected-rebuild", "rejected_videos": sorted(set(vids)), "rejected_why": why,
                "rejected_at": now_s(), "publish_at": None})
    save(r)
    return 0


if __name__ == "__main__":
    cmd, slug = sys.argv[1], sys.argv[2]
    if cmd == "record":
        sys.exit(record(slug, sys.argv[3]))
    if cmd == "approve":
        sys.exit(approve(slug))
    if cmd == "reject":
        sys.exit(reject(slug, " ".join(sys.argv[3:])))
    sys.exit(upload(slug))
