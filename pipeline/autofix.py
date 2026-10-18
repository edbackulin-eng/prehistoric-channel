"""Repair black or flat stretches in a finished video, then check again.

A stretch of near-black, featureless picture (blackdetect) is covered by the
neighbouring shot carried on: a still taken just before the stretch (or just
after it, when the shot before is dark too), given a slow push-in, and laid over
the stretch with short fades so it reads as the previous shot held a little
longer. The sound is left exactly as it was, so narration and subtitles keep
their timing.

The whole picture is re-encoded once at the channel's final bitrate. The result
is checked again; only what is still black afterwards is reported as a defect.

Ported from What If, where episode 004 shipped a 6.2 s black shot because a
regenerated frame was never looked at again. A defect found here is repaired,
never a reason to lose the day's video.

Closed Tabs adds one more repair: exposure. The first hook averaged a brightness
of 16 out of 255. If the finished film still measures under the channel's floor
(pipeline/grade.py), the whole picture is lifted with a gamma that brings it to
the target, and measured again.
"""
import json
import re
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image

from core.logger import get_logger

log = get_logger("autofix")

FPS = 30
PIX_TH = 0.10            # blackdetect pixel threshold, the same one QC uses
MIN_S = 0.5              # the shortest stretch QC reports
LEAD_S = 0.6             # how far outside the stretch the cover still is taken
FADE_S = 0.25
FINAL_VIDEO_ARGS = ["-b:v", "12M", "-maxrate", "16M", "-bufsize", "24M",
                    "-profile:v", "high", "-level:v", "4.1", "-g", "60"]


def _run(args):
    r = subprocess.run(["ffmpeg", "-hide_banner", *args], capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(r.stderr[-1500:])
    return r.stderr


def black_segments(video, min_s=MIN_S):
    err = _run(["-nostats", "-i", str(video), "-vf", f"blackdetect=d={min_s}:pix_th={PIX_TH}",
                "-an", "-f", "null", "-"])
    return [(float(a), float(b)) for a, b in
            re.findall(r"black_start:([\d.]+) black_end:([\d.]+)", err)]


def _duration(video):
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                          "-of", "csv=p=0", str(video)], capture_output=True, text=True).stdout
    return float(out.strip())


def _still(video, t, out):
    _run(["-loglevel", "error", "-y", "-ss", f"{max(0.0, t):.3f}", "-i", str(video),
          "-frames:v", "1", str(out)])
    a = np.asarray(Image.open(out).convert("L").resize((320, 180)), np.float32)
    return float(a.std()), float(np.percentile(a, 95))


def _cover_still(video, start, end, work, k):
    """A still worth showing instead: just before the stretch, else just after."""
    total = _duration(video)
    for t in (start - LEAD_S, end + LEAD_S, start - 2.0, end + 2.0):
        if not 0 <= t < total:
            continue
        p = Path(work) / f"cover_{k}_{t:.2f}.png"
        std, p95 = _still(video, t, p)
        if std > 8 and p95 > 25:          # not itself dark and flat
            return p, t
    return None, None


def _push_in(still, out, seconds):
    """A slow 1.00 -> 1.06 push on the still, so the held shot is not frozen."""
    d = max(seconds, 0.5)
    e = f"(st(0,clip(t/{d:.3f},0,1));ld(0)*ld(0)*(3-2*ld(0)))"
    vf = (f"scale=w='trunc(3840*(1+0.06*{e})/2)*2':h='trunc(2160*(1+0.06*{e})/2)*2':eval=frame,"
          f"crop=3840:2160,scale=1920:1080:flags=lanczos,fps={FPS},format=yuv420p")
    _run(["-loglevel", "error", "-y", "-loop", "1", "-framerate", str(FPS), "-t", f"{d:.3f}",
          "-i", str(still), "-vf", vf, "-c:v", "libx264", "-preset", "veryfast", "-crf", "16",
          str(out)])
    return out


def repair(video, out_path, work):
    """Cover every black stretch. Returns a report; writes out_path only if anything
    was repaired."""
    work = Path(work)
    work.mkdir(parents=True, exist_ok=True)
    segs = black_segments(video)
    report = {"found": [{"start": a, "end": b} for a, b in segs], "covered": [], "left": []}
    if not segs:
        return report
    inputs, chains, last = ["-i", str(video)], [], "0:v"
    n = 0
    for k, (a, b) in enumerate(segs):
        still, t = _cover_still(video, a, b, work, k)
        if still is None:
            log.warning("no usable neighbour for the stretch at %.2f-%.2f s", a, b)
            continue
        start, end = max(0.0, a - FADE_S), b + FADE_S
        clip = _push_in(still, work / f"cover_{k}.mp4", end - start)
        n += 1
        idx = n
        inputs += ["-i", str(clip)]
        chains.append(f"[{idx}:v]format=yuva420p,fade=t=in:st=0:d={FADE_S}:alpha=1,"
                      f"fade=t=out:st={max(0.0, end - start - FADE_S):.3f}:d={FADE_S}:alpha=1,"
                      f"setpts=PTS+{start:.3f}/TB[c{k}]")
        chains.append(f"[{last}][c{k}]overlay=0:0:eof_action=pass:"
                      f"enable='between(t,{start:.3f},{end:.3f})'[o{k}]")
        last = f"o{k}"
        report["covered"].append({"start": a, "end": b, "cover_from_s": round(t, 2)})
        log.info("covering %.2f-%.2f s with the shot at %.2f s", a, b, t)
    if not chains:
        report["left"] = report["found"]
        return report
    _run(["-loglevel", "error", "-y", *inputs, "-filter_complex", ";".join(chains),
          "-map", f"[{last}]", "-map", "0:a?", "-c:v", "libx264", "-preset", "veryfast",
          *FINAL_VIDEO_ARGS, "-pix_fmt", "yuv420p", "-r", str(FPS), "-c:a", "copy",
          "-movflags", "+faststart", str(out_path)])
    report["left"] = [{"start": a, "end": b} for a, b in black_segments(out_path)]
    report["duration_in"], report["duration_out"] = _duration(video), _duration(out_path)
    return report


def write(report, path):
    Path(path).write_text(json.dumps(report, indent=1), encoding="utf-8")


def lift(video, out_path, work):
    """Lift a film that measures too dark. Returns the before/after measurements."""
    import math
    from pipeline import grade
    before = grade.brightness(video)
    report = {"before": before}
    if before["ok"] or not before["frames"]:
        return report
    m = max(4.0, before["mean"]) / 255
    gamma = min(1.7, max(1.05, math.log(m) / math.log((grade.TARGET_MEAN - 5) / 255)))
    _run(["-loglevel", "error", "-y", "-i", str(video), "-vf",
          f"eq=gamma={gamma:.3f}:saturation=1.08,format=yuv420p", "-map", "0:v", "-map", "0:a?",
          "-c:v", "libx264", "-preset", "veryfast", *FINAL_VIDEO_ARGS, "-r", str(FPS),
          "-c:a", "copy", "-movflags", "+faststart", str(out_path)])
    report["gamma"] = round(gamma, 3)
    report["after"] = grade.brightness(out_path)
    return report
