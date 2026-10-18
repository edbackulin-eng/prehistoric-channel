"""Fetch the local models a build needs. Runs on a clean CI machine every time.

Only the voice is fetched here. The image model lives on Kaggle and never
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
    ("https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx",
     MODELS / "kokoro-v1.0.onnx", 300_000_000),
    ("https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin",
     MODELS / "voices-v1.0.bin", 20_000_000),
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
    try:
        with urllib.request.urlopen(url, timeout=900) as r, open(tmp, "wb") as fh:
            while chunk := r.read(1 << 20):
                fh.write(chunk)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise
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
