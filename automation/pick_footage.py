"""Pick one clip per real-footage beat of an episode.

    python automation/pick_footage.py living-fossils

Candidates come from work/candidates.json (pipeline/footage.py searches); the
pools a beat draws on are listed in POOLS. Each beat gets the best clip that
Gemini passes (pipeline/select.py) and that is not a perceptual near-duplicate
of a clip already chosen. Writes episodes/<slug>/footage.json.
"""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.logger import get_logger  # noqa: E402
from pipeline import footage, select  # noqa: E402

log = get_logger("pick")

POOLS = {
    "croc_eye": (["croc_eye"], r"crocodil|alligator|caiman"),
    "croc_bank": (["croc_bank", "croc_eye"], r"crocodil|alligator|caiman"),
    "croc_bank2": (["croc_bank", "croc_eye"], r"crocodil|alligator|caiman"),
    "hsc_beach": (["hsc_beach"], r"horseshoe|limulus"),
    "hsc_close": (["hsc_beach"], r"horseshoe|limulus"),
    "nautilus": (["nautilus"], r"nautilus"),
    "fern_forest": (["fern_forest"], r"fern|forest|jungle|rainforest|mist|fog"),
    "ocean_dawn": (["ocean_dawn"], r"ocean|sea|sunrise|sunset|horizon"),
    "ocean_horizon": (["ocean_dawn", "rough_sea"], r"ocean|sea|wave|horizon"),
    "rough_sea": (["rough_sea"], r"sea|ocean|wave|storm"),
    "river_mouth": (["river_mouth"], r"coast|river|estuary|shore|cliff|beach|aerial"),
    "books": (["books"], r"book|page|paper"),
    "strata": (["strata"], r"rock|strata|cliff|canyon|sediment|stone"),
    "deep_blue": (["deep_blue"], r"underwater|ocean|sea|ray|blue"),
    "island": (["island"], r"island|volcan|coast|cliff"),
    "island2": (["island"], r"island|volcan|coast|beach|shore"),
    "reef": (["reef", "deep_blue"], r"reef|coral|underwater"),
    "deep_night": (["deep_night"], r"deep|jelly|plankton|biolumin|dark|night|underwater"),
}
PER_BEAT = 10
# Rare animals have almost no 4K footage under an open licence; HD is accepted
# for them and upscaled in the render.
MIN_WIDTH = {"hsc_beach": 1280, "hsc_close": 1280, "nautilus": 1280}
ALLOW = {"books": ("text", "studio")}
FRAMES_DIR = ROOT / "work" / "qc" / "footage"


def beats(slug):
    shots = json.loads((ROOT / "episodes" / slug / "shots.json").read_text(encoding="utf-8"))
    return [b for line in shots["lines"] for b in line["beats"] if b["kind"] == "real"]


def shortlist(pool, pattern, taken, min_width=1920):
    rx = re.compile(pattern, re.I)
    c = [x for x in pool if x["id"] not in taken and rx.search(f"{x.get('tags','')} {x.get('title','')}")
         and (x.get("width") or 0) >= min_width and (x.get("duration") or 0) >= 6
         and (x.get("mime") or "video").startswith(("video", "application/ogg"))]
    c.sort(key=lambda x: (-(x["width"] >= 3840), -min(x.get("duration") or 0, 20)))
    return c[:PER_BEAT]


def main(slug):
    cands = json.loads((ROOT / "work" / "candidates.json").read_text(encoding="utf-8"))
    out_path = ROOT / "episodes" / slug / "footage.json"
    chosen = json.loads(out_path.read_text(encoding="utf-8")) if out_path.exists() else {}
    hashes = [c["phash"] for c in chosen.values()]
    taken = {c["id"] for c in chosen.values()}
    for b in beats(slug):
        need = b["need"]
        if need in chosen:
            continue
        pools, pattern = POOLS[need]
        pool = [x for p in pools for x in cands.get(p, [])]
        short = shortlist(pool, pattern, taken, MIN_WIDTH.get(need, 1920))
        clips = []
        for x in short:
            src = x.get("preview") or x["url"]
            if x["source"] != "Pixabay":
                # Commons webm/ogv often has no seek index: reading from the
                # middle over HTTP stalls, so the file is fetched whole first.
                try:
                    src = str(footage.download(x["url"], ROOT / "cache" / "media" /
                                               f"{x['id']}{Path(x['url'].split('?')[0]).suffix}"))
                except Exception as e:                               # noqa: BLE001
                    log.info("%s: download failed (%s)", x["id"], e)
                    continue
            fr = select.frames(src, x.get("duration"), FRAMES_DIR, x["id"])
            if len(fr) == 3:
                clips.append({**x, "frames": [str(f) for f in fr]})
        if not clips:
            log.warning("%s: no candidates", need)
            continue
        verdicts = select.judge(clips, need, b["brief"])
        ranked = []
        for c, v in zip(clips, verdicts):
            c["verdict"] = v
            c["phash"] = select.phash(c["frames"])
            log.info("%s | %s | m%s q%s %s %s%s%s | %s", need, c["id"], v.get("match"), v.get("quality"),
                     v.get("setting"), "P" if v.get("people") else "", "B" if v.get("branding") else "",
                     "T" if v.get("text") else "", v.get("subject"))
            if select.accepted(v, allow=ALLOW.get(need, ())) and not any(select.similar(c["phash"], h) for h in hashes):
                ranked.append(c)
        ranked.sort(key=lambda c: (-c["verdict"]["match"], -c["verdict"]["quality"],
                                   -(c["width"] >= 3840)))
        if not ranked:
            log.warning("%s: nothing passed", need)
            continue
        best = ranked[0]
        chosen[need] = {k: best.get(k) for k in ("id", "source", "page", "url", "width", "height",
                                                 "duration", "tags", "title", "author", "license",
                                                 "license_url", "attribution_required", "verdict",
                                                 "phash", "manifest")}
        chosen[need]["alternates"] = [{k: r.get(k) for k in ("id", "url", "width", "duration", "page",
                                                             "author", "license", "phash")}
                                      for r in ranked[1:3]]
        hashes.append(best["phash"])
        taken.add(best["id"])
        out_path.write_text(json.dumps(chosen, indent=1, ensure_ascii=False), encoding="utf-8")
        log.info("==> %s: %s (%dx%d, %s)", need, best["id"], best["width"], best["height"], best["source"])
    missing = [b["need"] for b in beats(slug) if b["need"] not in chosen]
    log.info("chosen %d, missing %s", len(chosen), missing)
    return 0 if not missing else 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
