"""Everything a render machine needs that the repository does not hold:
the Piper voice, the music and atmosphere beds named in their libraries, and
NASA's Blue Marble for the maps. Each is checked for size, so an error page
is never mistaken for a file.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from automation.fetch_models import fetch  # noqa: E402
from pipeline import docanim  # noqa: E402


def main():
    from automation import fetch_models
    for url, dest, min_bytes in fetch_models.FILES[:2]:
        fetch(url, dest, min_bytes)
    for t in json.loads((ROOT / "assets" / "music" / "library.json").read_text(encoding="utf-8"))["tracks"]:
        fetch(t["url"], ROOT / "assets" / "music" / t["file"], 500_000)
    beds = json.loads((ROOT / "assets" / "ambience" / "library.json").read_text(encoding="utf-8"))["beds"]
    for b in beds.values():
        if "file" in b:
            fetch(b["url"], ROOT / "assets" / "ambience" / b["file"], 50_000)
    fetch(docanim.BLUE_MARBLE_URL, docanim.BLUE_MARBLE, 20_000_000)
    return 0


if __name__ == "__main__":
    sys.exit(main())
