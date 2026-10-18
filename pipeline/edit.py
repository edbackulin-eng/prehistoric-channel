"""Render a planned edit over its narration: picture, transitions, sound, accents.

Shared by the one-minute checkpoint and the full episode, so the two can never
be cut differently.
"""
import json
import time
from pathlib import Path

import soundfile as sf

from core.logger import get_logger
from pipeline import assemble, motion, music, sound

log = get_logger("edit")

NEEDS_STILL = {"flux", "annotate", "notification", "cursor"}
TRANSITION_SFX = {"vhs": "glitch", "rgb": "glitch", "fade": "thump"}


def prepare_shot(s, frames, report):
    """Make a shot renderable whatever is missing: stock without its file falls
    back to its still, and a missing still borrows the nearest one."""
    if s["type"] == "stock" and not (s.get("asset") and Path(s["asset"]["path"]).exists()):
        s["type"] = "flux"                    # no footage after all: its fallback still
        report["problems"].append(f"shot {s['move']}: stock asset missing, still used")
    if s["type"] in NEEDS_STILL and s["move"] not in frames:
        # A missing still is a gap, not a reason to lose the video.
        if not frames:
            raise RuntimeError("no stills at all; nothing to put on screen")
        near = min(frames, key=lambda k: abs(k - s["move"]))
        frames[s["move"]] = frames[near]
        report["problems"].append(f"shot {s['move']} had no still; borrowed {near:03d}")


def render(plan, stills_dir, narration_wav, out_path, work, script_text=""):
    """Returns a report dict; raises only when the video physically cannot be built."""
    work = Path(work)
    work.mkdir(parents=True, exist_ok=True)
    t0 = time.monotonic()
    shots = plan["shots"]
    frames = {int(p.stem): p for p in Path(stills_dir).glob("*.png")}
    report = {"checks": {}, "problems": []}

    for k, p in frames.items():
        ok, where = motion.check_edges(p)
        if not ok:
            report["problems"].append(f"still {k:03d}: dark band at {where}")
    dups = motion.duplicates(list(frames.values()))
    if dups:
        report["problems"].append(f"{len(dups)} near-duplicate still pair(s)")

    paths, cues, cache = [], [], {}
    for s in shots:
        prepare_shot(s, frames, report)
        p, c = assemble.render_shot(s, frames, work, cache)
        paths.append(p)
        cues += c
        if s.get("transition"):
            cues.append({"t": s["start"], "kind": TRANSITION_SFX[s["transition"]]})
    log.info("%d shots rendered in %.0fs", len(shots), time.monotonic() - t0)

    paths = assemble.apply_transitions(paths, shots, work)
    picture = assemble.concat(paths, work / "picture.mp4", work)

    acc = plan["accents"]
    cues += [{"t": t, "kind": "riser"} for t in acc.get("riser", [])]
    cues += [{"t": t, "kind": "impact"} for t in acc.get("impact", [])]

    seconds = sf.info(str(narration_wav)).duration
    library = music.load_library()
    bed_plan = music.plan(script_text, seconds, tracks=library, use_llm=bool(script_text))
    bed = music.build_bed(bed_plan, work / "bed.wav")
    report["music"] = music.attribution(bed_plan)
    report["checks"]["audio"] = sound.mix(narration_wav, bed, cues, seconds,
                                          work / "mix.wav", work)
    if bed:
        try:
            report["checks"]["music_levels"] = sound.measure_duck(narration_wav, bed, work)
        except Exception as e:                       # a measurement never costs the video
            report["problems"].append(f"music level measurement failed: {e}")

    assemble.finish(picture, work / "mix.wav", acc, out_path, work)
    # Defects in the picture are repaired, not reported and shipped, and never a
    # reason to lose the video: black stretches are covered, a dark film lifted.
    from pipeline import autofix
    fixed = work / "fixed.mp4"
    try:
        rep = autofix.repair(out_path, fixed, work / "autofix")
        report["checks"]["black_repair"] = rep
        if rep.get("covered") and fixed.exists():
            fixed.replace(out_path)
        if rep.get("left"):
            report["problems"].append(f"black stretches left after repair: {rep['left']}")
        lit = autofix.lift(out_path, fixed, work)
        report["checks"]["brightness"] = lit.get("after", lit["before"])
        report["checks"]["brightness_before_lift"] = lit["before"] if "after" in lit else None
        if "after" in lit and fixed.exists():
            fixed.replace(out_path)
        if not report["checks"]["brightness"]["ok"]:
            report["problems"].append(f"picture still darker than the floor: {report['checks']['brightness']}")
    except Exception as e:                                       # noqa: BLE001
        report["problems"].append(f"autofix failed: {e}")
    from pipeline import editplan
    report["checks"]["monitor_share"] = round(
        sum(1 for s in shots if editplan.is_monitor(s)) / max(1, len(shots)), 3)
    from pipeline import qc
    meta = qc.probe(out_path)
    v = next(s for s in meta["streams"] if s["codec_type"] == "video")
    duration = float(meta["format"]["duration"])
    report["checks"].update({
        "duration_s": round(duration, 2), "narration_s": round(seconds, 2),
        "resolution": f"{v['width']}x{v['height']}", "fps": v.get("r_frame_rate"),
        "size_mb": round(int(meta["format"]["size"]) / 1e6, 1),
        "render_s": round(time.monotonic() - t0, 1), "shots": len(shots),
    })
    if abs(duration - seconds) > 0.5:
        report["problems"].append(f"video {duration:.1f}s vs narration {seconds:.1f}s")
    if (v["width"], v["height"]) != (1920, 1080):
        report["problems"].append(f"resolution {v['width']}x{v['height']}")
    return report


def write(report, path):
    Path(path).write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
