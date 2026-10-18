"""Find real footage for the stock shots of an edit plan, before stills are drawn.

    python automation/stock_assets.py queue/cicada-3301/build/hook_plan.json

Each "stock" shot gets a checked clip (Pixabay) or photograph (Wikimedia Commons,
NASA), never the same one twice. A shot that cannot have one becomes a generated
still with its fallback prompt, added to the plan's prompt list so stills.py
draws it. Credits for everything used go to <build>/credits.json.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from core import style  # noqa: E402
from core.logger import get_logger  # noqa: E402
from pipeline import stock  # noqa: E402

log = get_logger("stock_assets")


def main(plan_path):
    plan_path = Path(plan_path)
    b = plan_path.parent
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    shots = [s for s in plan["shots"] if s["type"] == "stock"]
    credits_path = b / "credits.json"
    used = []
    if shots:
        n = len(shots)
        # About three clips for every photograph: movement is what the stills lack.
        n_video = round(n * 0.75)
        assets = stock.collect(n_video, n - n_video, b / "stock", seed=len(plan["shots"]))
        videos = sorted((a for a in assets if a["kind"] == "video"), key=lambda a: -a["duration"])
        images = [a for a in assets if a["kind"] == "image"]
        # Longest shots take the longest clips; a clip far too short for its shot
        # gives the shot to a photograph instead.
        for s in sorted(shots, key=lambda s: -(s["end"] - s["start"])):
            d = s["end"] - s["start"]
            if videos and videos[0]["duration"] * 2 >= d:
                a = videos.pop(0)
            elif images:
                a = images.pop(0)
            elif videos:
                a = videos.pop(0)
            else:
                a = None
            if a:
                s["asset"] = {"kind": a["kind"], "path": a["path"], "duration": a.get("duration", 0),
                              "id": a["id"]}
                used.append(a)
            else:
                s["type"] = "flux"
                plan.setdefault("prompts", {})[str(s["move"])] = style.scene_prompt(s["prompt"])
    stock.write_credits(used, credits_path)
    plan_path.write_text(json.dumps(plan, indent=1, ensure_ascii=False), encoding="utf-8")
    kinds = {}
    for s in plan["shots"]:
        kinds[s["type"]] = kinds.get(s["type"], 0) + 1
    log.info("stock shots filled: %d; shot types now %s", len(used), kinds)
    print(json.dumps({"stock_used": len(used), "types": kinds}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
