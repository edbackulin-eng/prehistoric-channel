"""Contact sheets of footage candidates, for a human (or Claude) to judge when
no vision quota of this channel's own is available.

    python automation/sheets.py living-fossils hsc_beach nautilus ...

work/sheets/<need>.jpg: one row per candidate, three frames (10/50/90 %),
numbered; work/sheets/<need>.json maps the numbers back to the candidates.
"""
import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from automation import pick_footage as P  # noqa: E402
from pipeline import footage, select  # noqa: E402

OUT = ROOT / "work" / "sheets"


def sheet(need, cands, per=12):
    pools, pattern = P.POOLS[need]
    pool = [x for p in pools for x in cands.get(p, [])]
    chosen = json.loads((ROOT / "episodes" / sys.argv[1] / "footage.json").read_text(encoding="utf-8"))
    taken = {c["id"] for c in chosen.values()}
    old = P.PER_BEAT
    P.PER_BEAT = per
    short = P.shortlist(pool, pattern, taken, P.MIN_WIDTH.get(need, 1920))
    P.PER_BEAT = old
    rows = []
    for x in short:
        src = x.get("preview") or x["url"]
        if x["source"] != "Pixabay":
            try:
                src = str(footage.download(x["url"], ROOT / "cache" / "media" /
                                           f"{x['id']}{Path(x['url'].split('?')[0]).suffix}"))
            except Exception:                                     # noqa: BLE001
                continue
        fr = select.frames(src, x.get("duration"), P.FRAMES_DIR, x["id"])
        if len(fr) == 3:
            rows.append((x, fr))
    if not rows:
        return None
    tw, th = 400, 225
    im = Image.new("RGB", (tw * 3 + 60, th * len(rows)), (20, 20, 20))
    d = ImageDraw.Draw(im)
    font = ImageFont.truetype(str(ROOT / "assets" / "fonts" / "Anton-Regular.ttf"), 34)
    for r, (x, fr) in enumerate(rows):
        d.text((10, r * th + 90), str(r), fill=(255, 220, 0), font=font)
        for k, f in enumerate(fr):
            t = Image.open(f).convert("RGB")
            t.thumbnail((tw, th))
            im.paste(t, (60 + k * tw, r * th))
    OUT.mkdir(parents=True, exist_ok=True)
    im.save(OUT / f"{need}.jpg", quality=80)
    (OUT / f"{need}.json").write_text(json.dumps([x for x, _ in rows], indent=0), encoding="utf-8")
    return len(rows)


if __name__ == "__main__":
    cands = json.loads((ROOT / "work" / "candidates.json").read_text(encoding="utf-8"))
    for need in sys.argv[2:]:
        print(need, sheet(need, cands), flush=True)
