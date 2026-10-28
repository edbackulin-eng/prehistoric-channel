"""Fetch the local models a build needs. Runs on a clean CI machine every time.

Only the voice is fetched here: Piper's German Thorsten voice (CC0), which
narrates on the runner's CPU. The image model lives on Kaggle and never
touches this runner, which is the whole reason a 20-minute video can be built on
a free 2-core box.
"""
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "assets" / "models"
FONTS = ROOT / "assets" / "fonts"

FILES = [
    ("https://huggingface.co/rhasspy/piper-voices/resolve/main/de/de_DE/thorsten/high/de_DE-thorsten-high.onnx",
     MODELS / "piper" / "de_DE-thorsten-high.onnx", 100_000_000),
    ("https://huggingface.co/rhasspy/piper-voices/resolve/main/de/de_DE/thorsten/high/de_DE-thorsten-high.onnx.json",
     MODELS / "piper" / "de_DE-thorsten-high.onnx.json", 3_000),
    ("https://github.com/google/fonts/raw/main/ofl/anton/Anton-Regular.ttf",
     FONTS / "Anton-Regular.ttf", 100_000),
    ("https://github.com/google/fonts/raw/main/ofl/bebasneue/BebasNeue-Regular.ttf",
     FONTS / "BebasNeue-Regular.ttf", 50_000),
]


def fetch(url, dest, min_bytes):
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size >= min_bytes:
        print(f"  have {dest.name}")
        return
    print(f"  fetching {dest.name}...")
    tmp = dest.with_suffix(dest.suffix + ".part")
    import time
    for attempt in range(6):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (prehistoric-channel)"})
            with urllib.request.urlopen(req, timeout=900) as r, open(tmp, "wb") as fh:
                while chunk := r.read(1 << 20):
                    fh.write(chunk)
            break
        except Exception as e:
            tmp.unlink(missing_ok=True)
            if attempt == 5 or "429" not in str(e) and "timed out" not in str(e):
                raise
            time.sleep(20 * (attempt + 1))
    if tmp.stat().st_size < min_bytes:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"{dest.name} came back short "
                           f"({tmp.stat().st_size} bytes); refusing it")
    tmp.replace(dest)
    print(f"  {dest.name}: {dest.stat().st_size/1e6:.0f} MB")


def main():
    for url, dest, min_bytes in FILES:
        fetch(url, dest, min_bytes)
    print("models ready")
    return 0


if __name__ == "__main__":
    sys.exit(main())
