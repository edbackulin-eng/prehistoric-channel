"""Put a new thumbnail on an episode already uploaded (owner 07.11: thumbnail words fixed after
lebende-fossilien went up). Nothing else is touched: not the video, its privacy or date.

    python automation/set_thumbnail.py <slug>        # work/<slug>/release/thumbnail.jpg

Only a thumbnail the judge passed (thumbnails_report.json choice.judge_pass) is set.
Exit 0 when set, 1 when there is nothing to set, 2 when YouTube refused.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from core.logger import get_logger  # noqa: E402

log = get_logger("set-thumbnail")


def main(slug):
    marker = ROOT / "episodes" / slug / "uploaded.json"
    rel = ROOT / "work" / slug / "release"
    if not marker.exists():
        log.info("%s is not uploaded; its thumbnail goes up with the video", slug)
        return 1
    rep = json.loads((rel / "thumbnails_report.json").read_text(encoding="utf-8")) \
        if (rel / "thumbnails_report.json").exists() else {}
    if not (rep.get("choice") or {}).get("judge_pass") or not (rel / "thumbnail.jpg").exists():
        log.warning("%s: no judged thumbnail to set", slug)
        return 1
    from googleapiclient.http import MediaFileUpload
    from pipeline import youtube
    saved = json.loads(marker.read_text(encoding="utf-8"))
    vid = saved["youtube_id"]
    try:
        youtube.service().thumbnails().set(videoId=vid, media_body=MediaFileUpload(str(rel / "thumbnail.jpg"))).execute()
    except Exception as e:                                               # noqa: BLE001
        log.error("%s (%s): thumbnail refused: %s", slug, vid, str(e)[:300])
        return 2
    extras = {**(saved.get("extras") or {}), "thumbnail": "set", "thumbnail_file": rep["choice"]["best"]}
    marker.write_text(json.dumps({**saved, "extras": extras}), encoding="utf-8")
    log.info("%s (%s): thumbnail %s set", slug, vid, rep["choice"]["best"])
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
