"""Build the one-minute hook of an episode in full edit: the checkpoint render.

    python automation/build_hook.py cicada-3301 [--stills DIR]

Expects queue/<slug>/build/ to hold hook.wav, hook_lines.json, hook_words.json and
hook_plan.json (automation/narrate_hook.py and plan_hook.py). Stills come from
--stills (NNN.png, numbered by shot) or from build/stills/.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from pipeline import edit  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("slug")
    ap.add_argument("--stills")
    args = ap.parse_args()
    b = ROOT / "queue" / args.slug / "build"
    plan = json.loads((b / "hook_plan.json").read_text(encoding="utf-8"))
    report = edit.render(plan, Path(args.stills) if args.stills else b / "stills",
                         b / "hook.wav", b / "hook.mp4", b / "work")
    report["checks"]["families"] = [s["family"] for s in plan["shots"]]
    report["checks"]["types"] = [s["type"] for s in plan["shots"]]
    edit.write(report, b / "hook_report.json")
    print(json.dumps(report["checks"], indent=1))
    if report["problems"]:
        print("PROBLEMS:", report["problems"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
