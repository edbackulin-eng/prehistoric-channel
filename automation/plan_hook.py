"""Plan the edit of the narrated hook and list the stills it needs.

    python automation/plan_hook.py cicada-3301
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from core import style  # noqa: E402
from pipeline import editplan  # noqa: E402

NEEDS_STILL = {"flux", "annotate", "notification", "cursor"}


def main(slug):
    b = ROOT / "queue" / slug / "build"
    lines = json.loads((b / "hook_lines.json").read_text(encoding="utf-8"))
    words = json.loads((b / "hook_words.json").read_text(encoding="utf-8"))
    import soundfile as sf
    total = sf.info(str(b / "hook.wav")).duration
    plan = editplan.plan(lines, words, total, seed=3301)
    prompts = {}
    for s in plan["shots"]:
        if s["type"] in NEEDS_STILL:
            prompts[s["move"]] = style.scene_prompt(s["prompt"], with_people="person" in s["prompt"])
    plan["prompts"] = prompts
    (b / "hook_plan.json").write_text(json.dumps(plan, indent=1, ensure_ascii=False), encoding="utf-8")
    for s in plan["shots"]:
        what = s.get("prompt") or s.get("heading") or s.get("label") or s.get("header") or ""
        print(f"{s['start']:5.1f}-{s['end']:5.1f}  {s['family']:6s} {s['type']:12s} "
              f"{(s['transition'] or '-'):4s}  {str(what)[:60]}")
    print(f"\n{len(prompts)} stills to generate; accents: "
          f"{ {k: len(v) for k, v in plan['accents'].items()} }")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
