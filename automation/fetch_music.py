"""Fetch the music tracks named in the library onto the runner.

The repository holds code only, so the mp3s are not in it. Without this the
library would list tracks that are not on disk and every automated video would
ship silent - which is exactly what would have happened: nothing in the
workflows referenced assets/music at all, and the mixer simply skipped the bed
when the file was missing.
"""
import json
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.logger import get_logger  # noqa: E402

log = get_logger("fetch-music")

MUSIC_DIR = ROOT / "assets" / "music"
LIBRARY = MUSIC_DIR / "library.json"
MIN_BYTES = 500_000          # anything smaller is an error page, not a track
TIMEOUT_S = 120


def fetch(track):
    dest = MUSIC_DIR / track["file"]
    if dest.exists() and dest.stat().st_size >= MIN_BYTES:
        log.info("have %s (%.1f MB)", track["file"], dest.stat().st_size / 1e6)
        return True
    url = track.get("url")
    if not url:
        log.error("%s has no url in the library", track["file"])
        return False
    tmp = dest.with_suffix(".part")
    try:
        request = urllib.request.Request(
            url, headers={"User-Agent": "whatif-channel/1.0"})
        with urllib.request.urlopen(request, timeout=TIMEOUT_S) as response:
            tmp.write_bytes(response.read())
    except Exception as e:                                       # noqa: BLE001
        log.error("could not fetch %s: %s", track["file"], e)
        tmp.unlink(missing_ok=True)
        return False
    size = tmp.stat().st_size
    if size < MIN_BYTES:
        log.error("%s came back %d bytes, which is not a track", track["file"], size)
        tmp.unlink(missing_ok=True)
        return False
    tmp.replace(dest)
    log.info("fetched %s (%.1f MB)", track["file"], size / 1e6)
    return True


def main():
    MUSIC_DIR.mkdir(parents=True, exist_ok=True)
    tracks = json.loads(LIBRARY.read_text(encoding="utf-8"))["tracks"]
    ok = [fetch(t) for t in tracks]
    log.info("%d of %d tracks available", sum(ok), len(ok))
    # One missing track is survivable - the planner falls back to what is
    # there. None is not: the video would ship with no music.
    if not any(ok):
        log.error("no music is available, so the video would ship silent")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
