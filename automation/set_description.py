"""Add lines to the description of an episode already on YouTube (owner 07.11: the photo credits
of a thumbnail the owner sets in Studio). Nothing else of the video changes: not its title,
tags, privacy, date or thumbnail.

    python automation/set_description.py <slug> "<lines to add>"

The lines are added once (not again when they are there already) and kept in
episodes/<slug>/meta.json ("description_extra"), so a later upload of the episode carries them.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from core.logger import get_logger  # noqa: E402

log = get_logger("set-description")


def main(slug, extra):
    extra = extra.replace("\\n", "\n").strip()
    marker = ROOT / "episodes" / slug / "uploaded.json"
    if not marker.exists():
        raise SystemExit(f"{slug} is not uploaded")
    vid = json.loads(marker.read_text(encoding="utf-8"))["youtube_id"]
    from pipeline import youtube
    yt = youtube.service()
    item = yt.videos().list(part="snippet", id=vid).execute()["items"][0]
    sn = item["snippet"]
    desc = sn.get("description", "")
    if extra in desc:
        log.info("%s (%s): the lines are in the description already", slug, vid)
    else:
        body = {"id": vid, "snippet": {k: sn[k] for k in ("title", "categoryId", "tags", "defaultLanguage",
                                                          "defaultAudioLanguage") if k in sn}}
        body["snippet"]["description"] = (desc.rstrip() + "\n\n" + extra).strip()
        yt.videos().update(part="snippet", body=body).execute()
        log.info("%s (%s): %d characters added to the description", slug, vid, len(extra))
    mp = ROOT / "episodes" / slug / "meta.json"
    meta = json.loads(mp.read_text(encoding="utf-8"))
    if extra not in (meta.get("description_extra") or ""):
        meta["description_extra"] = ((meta.get("description_extra") or "") + "\n\n" + extra).strip()
        mp.write_text(json.dumps(meta, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2]))
