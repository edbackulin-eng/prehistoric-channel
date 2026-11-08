"""Does Gemini judge a clip the same when it sees four at once? (owner 08.11)

    python scripts/validate_batch.py [--size 4]          # needs GEMINI_API_KEY_PH (validate-batch.yml)

The clips of scripts/validate_batch_sample.json - each judged on its own before, the verdict kept in
episodes/<slug>/species_verdicts.json - are asked again, --size to a request, mixed across beats
and species as the real check mixes them (pipeline/speciescheck.ask_batch). The new answer and the
saved one are read the same way (speciescheck.passes); the share where pass/fail agrees is the
measure, and the owner's bar is 95 %. Nothing in the repository is changed: the report goes to
work/validate_batch.json (the workflow's artifact) and the log.
"""
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import calibrate_prefilter as cal  # noqa: E402
from pipeline import speciescheck as sc  # noqa: E402

BAR = 0.95


def main(args):
    size = int(args[args.index("--size") + 1]) if "--size" in args else 4
    key = sc._key()
    if not key:
        raise SystemExit("GEMINI_API_KEY_PH is not set")
    sample = json.loads((ROOT / "scripts" / "validate_batch_sample.json").read_text(encoding="utf-8"))["clips"]
    want = {(c["slug"], c["cid"]) for c in sample}
    rows = [r for r in cal.rows_of() if (r["slug"], r["cid"]) in want]
    random.seed(11)
    random.shuffle(rows)                      # beats and species mixed in every request, as in the check
    work = ROOT / "work" / "validate_batch"
    items, kept = [], []
    for r in rows:
        fr = sc._frames(r["cid"], r["src"], work)
        if not fr:
            print(f"{r['cid']}: no frames, left out")
            continue
        kept.append(r)
        items.append({"images": fr, "line": r["line"], "species": r["species"], "topic": r["beat"].get("topic"),
                      "archive": bool(r["beat"].get("archive")),
                      "what": "three frames from the start, middle and end of one video clip"})
    out, agree, errors, requests = [], 0, 0, 0
    for k in range(0, len(items), size):
        vs = sc.ask_batch(key, items[k:k + size])
        requests += 1
        for r, v in zip(kept[k:k + size], vs):
            if "error" in v:
                errors += 1
                out.append({"cid": r["cid"], "error": v["error"]})
                continue
            new, why = sc.passes(v, r["species"], bool(r["beat"].get("archive")), r["beat"].get("allow", ()))
            same = bool(new) == r["ok"]
            agree += same
            out.append({"slug": r["slug"], "cid": r["cid"], "kind": r["kind"], "saved_pass": r["ok"], "batch_pass": new,
                        "same": same, "why": why, "saved_note": r["v"].get("note"), "batch_note": v.get("note")})
    judged = len(out) - errors
    share = agree / judged if judged else 0
    by = {}
    for kind in ("species", "scene"):
        for ok in (True, False):
            g = [o for o in out if o.get("kind") == kind and o.get("saved_pass") is ok]
            by[f"{kind}, saved {'pass' if ok else 'fail'}"] = f"{sum(o['same'] for o in g)} of {len(g)} the same"
    rep = {"clips": len(out), "judged": judged, "errors": errors, "requests": requests, "size": size,
           "agreement": round(share, 3), "bar": BAR, "passed": judged >= 0.9 * len(rows) and share >= BAR,
           "by_class": by, "differences": [o for o in out if o.get("same") is False]}
    work.mkdir(parents=True, exist_ok=True)
    (work / "report.json").write_text(json.dumps({**rep, "all": out}, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(rep, indent=1, ensure_ascii=False))
    return 0 if rep["passed"] else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
