"""Render a documentary in parts on several machines, then join them untouched.

    render_part(plan, k, inputs, work, out)   -> part_k.mp4   (one machine each)
    join(plan, parts_dir, out)                -> picture.mp4  (-c copy)

Every shot is first rendered to its own short intermediate (fast, near
lossless), then one ffmpeg graph per part lays the dissolves, the cuts, the
captions and the title, adds the grain and makes the only real encode. All
parts share one encoder setting (H264 below) - profile, level, frame rate, a
fixed GOP, no scene-cut keyframes - which is what -c copy joining needs, and a
part always starts on a hard cut, so no dissolve is ever split between parts.

One grade for the whole film (GRADE): exposure is first evened out per clip
towards a common mid-grey, so stock from twenty cameras sits together, then
the same curve, saturation and warm-highlight/cool-shadow balance is applied
to everything. Natural and bright, not the crushed "cinematic" look.
"""
import json
import math
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np

from core.logger import get_logger

log = get_logger("docrender")

FPS = 30
SIZES = {"2160p": (3840, 2160), "1440p": (2560, 1440), "1080p": (1920, 1080)}
GRADE = ("eq=contrast=1.05:saturation=1.10:gamma={gamma:.3f},"
         "colorbalance=rs=-0.015:bs=0.02:rh=0.025:gh=0.008:bh=-0.02,"
         "curves=all='0/0.02 0.25/0.245 0.75/0.765 1/0.985'")
GRAIN = "noise=c0s=5:c0f=t+u:allf=t"
TARGET_LUMA = 0.43


def H264(size):
    level = "5.1" if size[0] > 2560 else "5.0"
    return ["-c:v", "libx264", "-preset", "medium", "-crf", "18", "-tune", "film",
            "-profile:v", "high", "-level:v", level, "-pix_fmt", "yuv420p",
            "-r", str(FPS), "-g", str(FPS * 2), "-keyint_min", str(FPS * 2), "-sc_threshold", "0",
            "-bf", "2", "-color_primaries", "bt709", "-color_trc", "bt709", "-colorspace", "bt709",
            "-movflags", "+faststart"]


INTER = ["-c:v", "libx264", "-preset", "veryfast", "-crf", "12", "-pix_fmt", "yuv420p", "-r", str(FPS)]


def _ff(*args, label="ffmpeg"):
    r = subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", *map(str, args)],
                       capture_output=True, text=True)
    if r.returncode:
        raise RuntimeError(f"{label}: {r.stderr[-2500:]}")


def probe(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-print_format", "json", "-show_streams",
                          "-show_format", str(path)], capture_output=True, text=True).stdout
    d = json.loads(out)
    v = next(s for s in d["streams"] if s["codec_type"] == "video")
    num, den = (v.get("avg_frame_rate") or v.get("r_frame_rate") or "30/1").split("/")
    fps = float(num) / float(den or 1) if float(den or 1) else 30.0
    dur = float(d["format"].get("duration") or v.get("duration") or 0)
    return {"w": int(v["width"]), "h": int(v["height"]), "fps": fps, "duration": dur}


def mean_luma(path, start, length):
    vals = []
    for f in (0.2, 0.5, 0.8):
        r = subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{start + length * f:.2f}", "-i", str(path),
                            "-frames:v", "1", "-vf", "scale=320:-2,format=gray", "-f", "rawvideo", "-"],
                           capture_output=True)
        if r.stdout:
            vals.append(np.frombuffer(r.stdout, np.uint8).mean() / 255)
    return float(np.mean(vals)) if vals else TARGET_LUMA


def gamma_for(luma, dark_ok=False):
    """eq gamma that moves the clip's mean toward TARGET_LUMA (a deep-sea clip
    is lifted only part of the way: it is meant to be dark)."""
    target = TARGET_LUMA if not dark_ok else min(TARGET_LUMA, max(luma * 1.6, 0.22))
    luma = min(0.9, max(0.04, luma))
    # eq maps v -> v ** (1 / gamma); solve mean ** (1 / gamma) = target
    g = math.log(luma) / math.log(target)
    return float(min(1.45, max(0.8, g)))


# ------------------------------------------------------------- shot clips
def real_clip(shot, src, need_s, out, size):
    info = probe(src)
    clip = shot["clip"]
    at = clip.get("at")
    at = 0.15 if at is None else at
    speed = 1.0
    avail = info["duration"]
    if info["fps"] >= 47 and avail * (1 - at) >= need_s * 0.5 + 0.5:
        speed = 0.5                                   # 50-60 fps footage plays at half speed
    if avail * (1 - at) * (1 / speed) < need_s:
        at = max(0.0, 1 - need_s * speed / avail - 0.02)
    if avail * (1 / speed) < need_s:                  # even from the start it is short: slow it
        speed = min(speed, avail / need_s * 0.98)
    start = avail * at
    src_len = need_s * speed + 0.2
    dark = shot.get("need") in ("deep_night", "reef", "nautilus", "nautilus2", "books")
    g = gamma_for(mean_luma(src, start, src_len), dark_ok=dark)
    w, h = size
    vf = (f"scale=trunc(iw*sar/2)*2:ih,setsar=1,setpts={1 / speed:.4f}*PTS,fps={FPS},"
          f"scale={w}:{h}:force_original_aspect_ratio=increase:flags=lanczos,crop={w}:{h},"
          f"{GRADE.format(gamma=g)},format=yuv420p")
    _ff("-ss", f"{start:.3f}", "-t", f"{src_len:.3f}", "-i", src, "-vf", vf, "-t", f"{need_s:.3f}",
        "-an", *INTER, out, label=f"real {shot['need']}")
    return {"speed": speed, "in": round(start, 2), "gamma": round(g, 3), "src_fps": info["fps"],
            "src": f"{info['w']}x{info['h']}"}


def _pipe(frames_iter, n_frames, out, size, gamma=1.0, grade=True):
    w, h = size
    vf = (GRADE.format(gamma=gamma) + ",format=yuv420p") if grade else "format=yuv420p"
    p = subprocess.Popen(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo",
                          "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", str(FPS), "-i", "-",
                          "-vf", vf, *INTER, str(out)], stdin=subprocess.PIPE)
    k = 0
    for fr in frames_iter:
        p.stdin.write(np.ascontiguousarray(fr).tobytes())
        k += 1
        if k >= n_frames:
            break
    p.stdin.close()
    if p.wait():
        raise RuntimeError(f"pipe encode failed for {out}")


def still_clip(shot, layers_dir, need_s, out, size):
    from pipeline import parallax
    name = shot["image"]
    d = Path(layers_dir)
    raw = cv2.imread(str(d / f"{name}.jpg"))
    if raw is None:
        raise FileNotFoundError(f"2.5D source {name}.jpg is not in {d}")
    img = raw[:, :, ::-1]
    bgp, ap = d / f"{name}_bg.jpg", d / f"{name}_alpha.png"
    bg = cv2.imread(str(bgp))[:, :, ::-1] if bgp.exists() else None
    alpha = cv2.imread(str(ap), cv2.IMREAD_GRAYSCALE) if ap.exists() else None
    if alpha is None:
        bg = None
    img, bg = np.ascontiguousarray(img), (np.ascontiguousarray(bg) if bg is not None else None)
    lum = float(cv2.resize(img, (320, 180)).mean() / 255)
    n = int(round(need_s * FPS))
    gen = parallax.frames(img, bg, alpha, shot.get("move", "push_in"), need_s, FPS, size[0], size[1],
                          fx=shot.get("fx"), seed=hash(name) % 1000)
    _pipe(gen, n, out, size, gamma=gamma_for(lum, dark_ok=True))
    return parallax.frames.last_report


def anim_clip(shot, need_s, out, size, backdrop=None):
    from pipeline import docanim
    n = int(round(need_s * FPS))
    if shot["anim"] == "map":
        gen = docanim.map_frames(shot["places"], need_s, size[0], size[1], FPS, caption=shot.get("caption"))
    else:
        gen = docanim.timeline_frames(need_s, size[0], size[1], FPS, backdrop=backdrop)
    _pipe(gen, n, out, size, grade=False)
    return {"anim": shot["anim"]}


# ------------------------------------------------------------------ parts
def render_part(plan, k, inputs, work, out_dir, res="2160p"):
    """inputs: {"media": dir of real clips by id, "layers": dir of 2.5D layers}."""
    size = SIZES[res]
    work, out_dir = Path(work), Path(out_dir)
    work.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    a, b = plan["parts"][k]
    shots = plan["shots"][a:b]
    X = plan["xfade_s"]
    t0 = time.monotonic()
    report = {"part": k, "shots": [a, b], "res": res, "clips": []}
    clips, durs = [], []
    backdrop = None
    for i, s in enumerate(shots):
        nxt = shots[i + 1] if i + 1 < len(shots) else None
        tail = X if (nxt is not None and nxt["transition"] == "fade") else 0.0
        need = s["end"] - s["start"] + tail
        out = work / f"shot_{a + i:03d}.mp4"
        t1 = time.monotonic()
        if s["kind"] == "real":
            src = next(Path(inputs["media"]).glob(f"{s['clip']['id']}.*"))
            info = real_clip(s, src, need, out, size)
        elif s["kind"] in ("recon", "photo"):
            info = still_clip(s, inputs["layers"], need, out, size)
        else:
            if s["anim"] == "timeline":
                bp = Path(inputs["layers"]) / "devonian_sea__s0.jpg"
                backdrop = cv2.imread(str(bp))[:, :, ::-1] if bp.exists() else None
            info = anim_clip(s, need, out, size, backdrop)
        clips.append(out)
        durs.append(need)
        report["clips"].append({"i": a + i, "kind": s["kind"], "s": round(time.monotonic() - t1, 1),
                                **(info or {})})
        log.info("part %d shot %d (%s) %.1fs in %.0fs", k, a + i, s["kind"], need, time.monotonic() - t1)

    # overlays: captions and the title, as PNG inputs faded in and out
    from pipeline import docanim
    overlays = []
    p0 = shots[0]["start"]
    for s in shots:
        if s.get("label"):
            png = work / f"cap_{s['label'][0].replace(' ', '_')}.png"
            docanim.caption_png(s["label"][0], s["label"][1], png, *size)
            st = s["start"] - p0 + 0.8
            overlays.append((png, st, min(4.2, s["end"] - s["start"] - 1.2)))
        if s.get("title"):
            png = work / "title.png"
            docanim.title_png(s["title"], docanim.TITLE_SUB, png, *size)
            st = s["start"] - p0 + 0.5
            overlays.append((png, st, min(s.get("hold", 3.0) + 1.0, s["end"] - s["start"] - 0.9)))

    args, graph = [], []
    for c in clips:
        args += ["-i", c]
    for png, _, _ in overlays:
        args += ["-loop", "1", "-framerate", str(FPS), "-i", png]
    # every input on one timebase, or xfade refuses a concat's output
    for i in range(len(clips)):
        graph.append(f"[{i}:v]fps={FPS},settb=AVTB,setpts=PTS-STARTPTS[c{i}]")
    # dissolves and cuts
    label, length = "[c0]", durs[0]
    for i in range(1, len(clips)):
        if shots[i]["transition"] == "fade":
            off = length - X
            graph.append(f"{label}[c{i}]xfade=transition=fade:duration={X}:offset={off:.3f},"
                         f"settb=AVTB[v{i}]")
            length = off + durs[i]
        else:
            graph.append(f"{label}[c{i}]concat=n=2:v=1:a=0,settb=AVTB[v{i}]")
            length += durs[i]
        label = f"[v{i}]"
    for j, (png, st, dur) in enumerate(overlays):
        idx = len(clips) + j
        dur = max(1.6, dur)
        graph.append(f"[{idx}:v]format=rgba,trim=duration={dur:.3f},"
                     f"fade=t=in:st=0:d=0.7:alpha=1,fade=t=out:st={dur - 0.8:.3f}:d=0.8:alpha=1,"
                     f"settb=AVTB,setpts=PTS+{st:.3f}/TB[o{j}]")
        graph.append(f"{label}[o{j}]overlay=0:0:eof_action=pass[w{j}]")
        label = f"[w{j}]"
    total = shots[-1]["end"] - shots[0]["start"]
    graph.append(f"{label}{GRAIN},trim=duration={total:.3f},format=yuv420p[out]")
    part = out_dir / f"part_{k}.mp4"
    if len(clips) == 1 and not overlays:
        graph = [f"[0:v]{GRAIN},trim=duration={total:.3f},format=yuv420p[out]"]
    graph = [g for g in graph if g]
    t1 = time.monotonic()
    _ff(*args, "-filter_complex", ";".join(graph), "-map", "[out]", "-an", *H264(size), part,
        label=f"part {k}")
    report["encode_s"] = round(time.monotonic() - t1, 1)
    report["render_s"] = round(time.monotonic() - t0, 1)
    report["seconds"] = round(probe(part)["duration"], 3)
    report["expected_s"] = round(total, 3)
    (out_dir / f"part_{k}.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    log.info("part %d: %.1f s of picture in %.0f s", k, total, report["render_s"])
    return report


def join(parts, out):
    lst = Path(out).with_suffix(".txt")
    lst.write_text("".join(f"file '{Path(p).resolve().as_posix()}'\n" for p in parts), encoding="utf-8")
    _ff("-f", "concat", "-safe", "0", "-i", lst, "-c", "copy", "-movflags", "+faststart", out, label="join")
    return out
