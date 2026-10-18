"""Record a clip chosen by eye from a contact sheet (automation/sheets.py).

    python automation/choose.py <slug> <need> <sheet> <row> [at] [alt_sheet:row ...]

`at` is where in the clip the shot begins, as a fraction of its length.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
KEYS = ("id", "source", "page", "url", "width", "height", "duration", "tags", "title", "author",
        "license", "license_url", "attribution_required", "manifest")


def row(sheet, i):
    return json.loads((ROOT / "work" / "sheets" / f"{sheet}.json").read_text(encoding="utf-8"))[int(i)]


def main(slug, need, sheet, i, at=None, *alts):
    path = ROOT / "episodes" / slug / "footage.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    x = row(sheet, i)
    entry = {k: x.get(k) for k in KEYS}
    entry["verdict"] = {"by": "visual review of a contact sheet", "sheet": f"{sheet}#{i}"}
    if at is not None:
        entry["at"] = float(at)
    entry["alternates"] = []
    for a in alts:
        s, j = a.split(":")
        y = row(s, j.split("@")[0])
        alt = {k: y.get(k) for k in KEYS}
        if "@" in j:
            alt["at"] = float(j.split("@")[1])
        entry["alternates"].append(alt)
    data[need] = entry
    path.write_text(json.dumps(data, indent=1, ensure_ascii=False), encoding="utf-8")
    print(need, "->", entry["id"], entry.get("title") or entry.get("tags", "")[:50], entry["license"])


if __name__ == "__main__":
    main(*sys.argv[1:])
