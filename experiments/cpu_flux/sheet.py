"""Collect the five CPU FLUX images and their measurements into one sheet and one report.

    python experiments/cpu_flux/sheet.py <dir with the jobs' artifacts> <out dir>
"""
import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]


def main(src, out):
    src, out = Path(src), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    metrics = sorted((json.loads(p.read_text()) for p in src.rglob("metrics_*.json")), key=lambda m: m["id"])
    font = ImageFont.truetype(str(ROOT / "assets" / "fonts" / "Anton-Regular.ttf"), 30)
    tiles = []
    for m in metrics:
        p = next(src.rglob(f"cpu_flux_{m['id']}_{m['name']}.png"), None)
        if p is None:
            continue
        im = Image.open(p).convert("RGB")
        im.save(out / p.name)
        t = im.resize((896, 512), Image.LANCZOS)
        d = ImageDraw.Draw(t)
        label = f"{m['name']}  |  {m['generate_s']:.0f} s  |  {m['peak_rss_gb']:.1f} GB RAM"
        d.rectangle([0, 470, 896, 512], fill=(0, 0, 0))
        d.text((12, 474), label, font=font, fill=(255, 255, 255))
        tiles.append(t)
    cols = 2
    rows = (len(tiles) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * 896, rows * 512), (20, 20, 20))
    for i, t in enumerate(tiles):
        sheet.paste(t, ((i % cols) * 896, (i // cols) * 512))
    sheet.save(out / "cpu_flux_sheet.jpg", quality=92)
    gen = [m["generate_s"] for m in metrics]
    dl = [m["download_s"] for m in metrics]
    rep = {"images": len(metrics), "per_image": metrics,
           "generate_s_mean": round(sum(gen) / len(gen), 1) if gen else None,
           "generate_s_max": max(gen) if gen else None,
           "download_s_mean": round(sum(dl) / len(dl), 1) if dl else None,
           "peak_rss_gb_max": max(m["peak_rss_gb"] for m in metrics) if metrics else None}
    (out / "cpu_flux_report.json").write_text(json.dumps(rep, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in rep.items() if k != "per_image"}, indent=1))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
