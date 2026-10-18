"""Render an episode in parts on several machines, then join them without re-encoding.

A 13-minute episode rendered in one job took about an hour, and with the GPU
wait in front of it the whole build ran into GitHub's six-hour job limit. Here
the shots are split into PARTS stretches of similar length; each is rendered by
its own job (picture only: shots, transitions, accents, VHS layer, black repair
and exposure lift), and a last step joins the parts with -c copy, builds the
one sound mix for the whole film and muxes it in.

A part always begins on a shot whose transition is a cut (vhs, rgb or none),
never a crossfade, because a crossfade needs the previous shot's tail and that
belongs to another part. Every final encode uses the same H.264 profile, level,
frame rate and GOP (assemble.H264, autofix.FINAL_VIDEO_ARGS), which is what
joining with -c copy requires.
"""
import json
import subprocess
import time
from pathlib import Path

import soundfile as sf

from core.logger import get_logger
from pipeline import assemble, autofix, edit, motion, music, sound

log = get_logger("parts")

PARTS = 5


def split(shots, parts=PARTS):
    """[(first, end)] shot index ranges, each starting on a cut."""
    total = shots[-1]["end"]
    bounds = [0]
    for k in range(1, parts):
        target = total * k / parts
        cands = [i for i, s in enumerate(shots)
                 if i > bounds[-1] and s.get("transition") != "fade"]
        if not cands:
            break
        bounds.append(min(cands, key=lambda i: abs(shots[i]["start"] - target)))
    bounds.append(len(shots))
    return [(a, b) for a, b in zip(bounds, bounds[1:]) if b > a]


def _shift(accents, off, dur):
    """The accents that fall inside a part, on the part's own clock."""
    def win(ws):
        return [[max(0.0, a - off), min(dur, b - off)] for a, b in ws if b > off and a < off + dur]
    out = {"punch": win(accents.get("punch", [])), "shake": win(accents.get("shake", []))}
    out["kinetic"] = [{**k, "start": k["start"] - off, "end": min(dur, k["end"] - off)}
                      for k in accents.get("kinetic", []) if off <= k["start"] < off + dur]
    return out


def render_part(plan, stills_dir, k, work, out_dir):
    """Render part k. Writes part_k.mp4, cues_k.json and part_k.json into out_dir;
    returns the part's report. A k beyond the number of parts writes nothing."""
    work, out_dir = Path(work), Path(out_dir)
    work.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.monotonic()
    shots = plan["shots"]
    ranges = split(shots)
    if k >= len(ranges):
        log.info("part %d: nothing to render (%d parts)", k, len(ranges))
        return None
    a, b = ranges[k]
    mine = shots[a:b]
    off = mine[0]["start"]
    dur = mine[-1]["end"] - off
    frames = {int(p.stem): p for p in Path(stills_dir).glob("*.png")}
    report = {"part": k, "shots": [a, b], "start": off, "seconds": round(dur, 3), "problems": []}

    paths, cues, cache = [], [], {}
    for s in mine:
        edit.prepare_shot(s, frames, report)
        p, c = assemble.render_shot(s, frames, work, cache)
        paths.append(p)
        cues += c
        if s.get("transition"):
            cues.append({"t": s["start"], "kind": edit.TRANSITION_SFX[s["transition"]]})
    # transitions inside the part; the first shot's own cut-in glitch too
    paths = assemble.apply_transitions(paths, mine, work)
    first = mine[0].get("transition")
    if first in ("vhs", "rgb"):
        head = work / f"head_{a:03d}.mp4"
        assemble._ff("-i", str(paths[0]), "-vf", assemble._glitch_filter(first),
                     *assemble._x264(), str(head))
        paths[0] = head
    picture = assemble.concat(paths, work / "picture.mp4", work)
    out = out_dir / f"part_{k}.mp4"
    assemble.finish(picture, None, _shift(plan["accents"], off, dur), out, work, duration=dur)

    # repairs on the part itself, so the join never has to re-encode
    fixed = work / "fixed.mp4"
    rep = autofix.repair(out, fixed, work / "autofix")
    if rep.get("covered") and fixed.exists():
        fixed.replace(out)
    report["black_repair"] = rep
    lit = autofix.lift(out, fixed, work)
    if "after" in lit and fixed.exists():
        fixed.replace(out)
    report["brightness"] = lit.get("after", lit["before"])
    report["render_s"] = round(time.monotonic() - t0, 1)
    (out_dir / f"cues_{k}.json").write_text(json.dumps(cues), encoding="utf-8")
    (out_dir / f"part_{k}.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    log.info("part %d: shots %d-%d, %.1f s of picture in %.0f s", k, a, b - 1, dur, report["render_s"])
    return report


def merge(plan, parts_dir, narration_wav, out_path, work, script_text=""):
    """Join the parts with -c copy, mix the sound for the whole film, mux.
    Returns the episode report, in the same shape edit.render() gives."""
    from pipeline import editplan, grade, qc
    work, parts_dir = Path(work), Path(parts_dir)
    work.mkdir(parents=True, exist_ok=True)
    t0 = time.monotonic()
    n = len(split(plan["shots"]))
    files = [parts_dir / f"part_{k}.mp4" for k in range(n)]
    missing = [f.name for f in files if not f.exists()]
    if missing:
        raise RuntimeError(f"parts missing: {missing}")
    report = {"checks": {"parts": []}, "problems": []}
    cues = []
    for k in range(n):
        cues += json.loads((parts_dir / f"cues_{k}.json").read_text(encoding="utf-8"))
        pr = json.loads((parts_dir / f"part_{k}.json").read_text(encoding="utf-8"))
        report["checks"]["parts"].append({x: pr.get(x) for x in ("part", "shots", "seconds", "render_s")})
        report["problems"] += pr.get("problems", [])
    picture = assemble.concat(files, work / "joined.mp4", work)

    acc = plan["accents"]
    cues += [{"t": t, "kind": "riser"} for t in acc.get("riser", [])]
    cues += [{"t": t, "kind": "impact"} for t in acc.get("impact", [])]
    seconds = sf.info(str(narration_wav)).duration
    bed_plan = music.plan(script_text, seconds, tracks=music.load_library(), use_llm=bool(script_text))
    bed = music.build_bed(bed_plan, work / "bed.wav")
    report["music"] = music.attribution(bed_plan)
    report["checks"]["audio"] = sound.mix(narration_wav, bed, cues, seconds, work / "mix.wav", work)
    if bed:
        try:
            report["checks"]["music_levels"] = sound.measure_duck(narration_wav, bed, work)
        except Exception as e:                                   # noqa: BLE001
            report["problems"].append(f"music level measurement failed: {e}")
    r = subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(picture),
                        "-i", str(work / "mix.wav"), "-map", "0:v", "-map", "1:a", "-c:v", "copy",
                        "-c:a", "aac", "-b:a", "192k", "-shortest", "-movflags", "+faststart",
                        str(out_path)], capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(r.stderr[-1500:])

    # measured on the joined film; the parts were already repaired and lifted
    report["checks"]["brightness"] = grade.brightness(out_path)
    blacks = autofix.black_segments(out_path)
    report["checks"]["black_segments"] = blacks
    if blacks:                       # a seam can only come from a part; cover it here
        fixed = work / "fixed.mp4"
        rep = autofix.repair(out_path, fixed, work / "autofix")
        report["checks"]["black_repair"] = rep
        if rep.get("covered") and fixed.exists():
            fixed.replace(out_path)
    shots = plan["shots"]
    report["checks"]["monitor_share"] = round(
        sum(1 for s in shots if editplan.is_monitor(s)) / max(1, len(shots)), 3)
    meta = qc.probe(out_path)
    v = next(s for s in meta["streams"] if s["codec_type"] == "video")
    duration = float(meta["format"]["duration"])
    report["checks"].update({
        "duration_s": round(duration, 2), "narration_s": round(seconds, 2),
        "resolution": f"{v['width']}x{v['height']}", "fps": v.get("r_frame_rate"),
        "size_mb": round(int(meta["format"]["size"]) / 1e6, 1),
        "merge_s": round(time.monotonic() - t0, 1), "shots": len(shots),
    })
    if abs(duration - seconds) > 0.5:
        report["problems"].append(f"video {duration:.1f}s vs narration {seconds:.1f}s")
    return report
