"""Set the species prefilter's thresholds on Gemini's own answers (owner 08.11).

    python scripts/calibrate_prefilter.py [--write]

Every stock-footage verdict Gemini gave (episodes/*/species_verdicts.json) is matched to its
clip and its beat - through every version of the episode's footage.json and shots.json in the
git history, since a replan keeps the verdicts but not the old beats - and the clip's poster
frame is scored on the CPU against the beat's topic (pipeline/prefilter.py). Gemini's answer,
read by speciescheck.passes, is the label. Per kind of beat ("species": an animal asked for,
"scene": none), the thresholds are where

    accept: of the clips scoring at or above it, at least ACCEPT_PRECISION passed with Gemini;
    reject: of the clips scoring at or below it, at most REJECT_OK_RATE passed with Gemini, and
            at most REJECT_LOST of all the passing ones are lost there.

--write keeps them in config/channel.json "prefilter".
"""
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from pipeline import prefilter, speciescheck  # noqa: E402

ACCEPT_PRECISION = 0.95
REJECT_OK_RATE = 0.10
REJECT_LOST = 0.10
MIN_SIDE = 15


def git(*a):
    return subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True, encoding="utf-8").stdout


def versions(path):
    """Every committed version of a file, newest first, then the working copy."""
    out = []
    p = ROOT / path
    if p.exists():
        out.append(p.read_text(encoding="utf-8"))
    for sha in git("log", "--format=%H", "--", path).split():
        t = git("show", f"{sha}:{path}")
        if t.strip():
            out.append(t)
    return out


def rows_of():
    rows = []
    for vf in sorted((ROOT / "episodes").glob("*/species_verdicts.json")):
        slug = vf.parent.name
        known = json.loads(vf.read_text(encoding="utf-8"))
        src, need_of, beat_of = {}, {}, {}
        for t in versions(f"episodes/{slug}/footage.json"):
            for need, e in json.loads(t).items():
                for c in [e] + list(e.get("alternates") or []):
                    if isinstance(c, dict) and c.get("id") and c["id"] not in src:
                        src[c["id"]] = c
                        need_of[c["id"]] = (need, e.get("query"))
        for t in versions(f"episodes/{slug}/shots.json"):
            for line in json.loads(t)["lines"]:
                for b in line["beats"]:
                    if b.get("kind") == "real" and b.get("need") and b["need"] not in beat_of:
                        beat_of[b["need"]] = (b, line.get("at", ""))
        species = {}
        sp = ROOT / "episodes" / slug / "species.json"
        if sp.exists():
            species = json.loads(sp.read_text(encoding="utf-8"))
        speciescheck.SPECIES.update({k: v for k, v in species.items() if k not in speciescheck.SPECIES})
        for cid, v in known.items():
            if "__s" in cid or cid not in src or "error" in v or v.get("model") == "siglip" or v.get("missing"):
                continue
            need, query = need_of[cid]
            b, _line = beat_of.get(need, ({}, ""))
            sp_key = b.get("species")
            ok, _why = speciescheck.passes(v, sp_key, bool(b.get("archive")), b.get("allow", ()))
            if ok is None:
                continue
            rows.append({"slug": slug, "cid": cid, "src": src[cid], "kind": "species" if sp_key else "scene",
                         "text": prefilter.text_for(query=b.get("topic") or query), "ok": bool(ok),
                         "v": v, "beat": b, "line": _line, "species": sp_key})
    return rows


def choose(sc, ok):
    pairs = sorted(zip(sc, ok), key=lambda x: -x[0])
    n_ok = sum(ok)
    accept, good = None, 0
    for k, (s, o) in enumerate(pairs, 1):
        good += o
        if k >= MIN_SIDE and good / k >= ACCEPT_PRECISION:
            accept = s
    reject, lost = None, 0
    for k, (s, o) in enumerate(pairs[::-1], 1):
        lost += o
        if k >= MIN_SIDE and lost / k <= REJECT_OK_RATE and lost <= REJECT_LOST * max(1, n_ok):
            reject = s
    if accept is not None and reject is not None and reject >= accept:
        reject = None
    return accept, reject


def report(sc, ok, accept, reject):
    a = [o for s, o in zip(sc, ok) if accept is not None and s >= accept]
    r = [o for s, o in zip(sc, ok) if reject is not None and s <= reject]
    n = len(sc)
    return {"clips": n, "gemini_pass": sum(ok), "accept": accept, "reject": reject,
            "taken_on_cpu": len(a), "their_gemini_pass_share": round(sum(a) / len(a), 3) if a else None,
            "dropped_on_cpu": len(r), "good_clips_lost": sum(r), "still_for_gemini": n - len(a) - len(r),
            "share_decided_on_cpu": round((len(a) + len(r)) / n, 3) if n else 0}


def main(args):
    rows = rows_of()
    have = []
    for r in rows:
        p = prefilter.poster(r["src"], r["cid"])
        if p:
            r["img"] = p
            have.append(r)
    print(f"{len(rows)} Gemini verdicts on stock clips, {len(have)} with a poster frame")
    sc = prefilter.scores([(r["text"], r["img"]) for r in have])
    out = {}
    for kind in ("species", "scene"):
        idx = [i for i, r in enumerate(have) if r["kind"] == kind]
        s, o = [sc[i] for i in idx], [have[i]["ok"] for i in idx]
        a, rj = choose(s, o)
        out[kind] = report(s, o, a, rj)
    print(json.dumps(out, indent=1))
    if "--write" in args:
        p = ROOT / "config" / "channel.json"
        cfg = json.loads(p.read_text(encoding="utf-8"))
        cfg["prefilter"] = {"arch": prefilter.ARCH, "pretrained": prefilter.PRETRAINED,
                            "thresholds": {k: {"accept": None if v["accept"] is None else round(v["accept"], 4),
                                               "reject": None if v["reject"] is None else round(v["reject"], 4)}
                                           for k, v in out.items()},
                            "_set_on": f"{datetime.now(timezone.utc):%Y-%m-%d}: " + "; ".join(
                                f"{k}: {v['clips']} Gemini verdicts, {v['taken_on_cpu']} taken at "
                                f"{v['their_gemini_pass_share']} agreement, {v['dropped_on_cpu']} dropped losing "
                                f"{v['good_clips_lost']} passing" for k, v in out.items())}
        p.write_text(json.dumps(cfg, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        print("written to", p)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
