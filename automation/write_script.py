"""Write the script for one queued topic and store it as text in queue/<slug>/.

    python automation/write_script.py cicada-3301
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from core.logger import get_logger  # noqa: E402
from pipeline import scriptwriter, sources  # noqa: E402

log = get_logger("write-script")


def main(slug):
    queue = json.loads((ROOT / "topics" / "queue.json").read_text(encoding="utf-8"))
    topic = next(t for t in queue["topics"] if t["slug"] == slug)
    found = sources.gather(topic)
    recent = [t.get("title") for t in queue["topics"] if t.get("title")][-10:]
    result = scriptwriter.write(topic, found, recent)
    out = ROOT / "queue" / slug
    out.mkdir(parents=True, exist_ok=True)
    (out / "script.txt").write_text(result["script"] + "\n", encoding="utf-8")
    (out / "meta.json").write_text(json.dumps({
        "slug": slug, "name": topic["name"], "title": result["title"],
        "thumbnail_words": result["thumbnail_words"], "claims": result["claims"],
        "sources": [{"title": s["title"], "url": s["url"]} for s in found],
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info("%s: %d words, title %r", slug, len(result["script"].split()), result["title"])
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
