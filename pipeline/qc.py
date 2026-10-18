"""Automatic checks on a finished video, plus the contact sheet.

Every video is measured before it is packaged: anything outside these bounds is
reported rather than silently shipped. The thresholds come from what KT2 actually
produced, not from guesswork.
"""
import json
import re
import subprocess
from pathlib import Path

from core.logger import get_logger

log = get_logger("qc")

TARGET_LUFS = -14.0
LUFS_TOLERANCE = 1.0
MIN_W, MIN_H, FPS = 1920, 1080, 30
MAX_SILENCE_S = 3.0          # longer than this mid-video means a gap in narration
MAX_BLACK_S = 0.5
THUMB_W, THUMB_H = 1280, 720
THUMB_MAX_BYTES = 2 * 1024 * 1024


def _run(args):
    return subprocess.run(args, capture_output=True, text=True)


def probe(path):
    out = _run(["ffprobe", "-v", "error", "-print_format", "json",
                "-show_format", "-show_streams", str(path)])
    return json.loads(out.stdout)


def measure_loudness(path):
    err = _run(["ffmpeg", "-hide_banner", "-nostats", "-i", str(path),
                "-af", "ebur128=framelog=quiet", "-f", "null", "-"]).stderr
    i = re.findall(r"I:\s*(-?[\d.]+)\s*LUFS", err)
    lra = re.findall(r"LRA:\s*(-?[\d.]+)\s*LU", err)
    return (float(i[-1]) if i else None), (float(lra[-1]) if lra else None)


def find_black(path, min_s=MAX_BLACK_S):
    err = _run(["ffmpeg", "-hide_banner", "-nostats", "-i", str(path),
                "-vf", f"blackdetect=d={min_s}:pix_th=0.10", "-f", "null", "-"]).stderr
    return [(float(a), float(b)) for a, b in
            re.findall(r"black_start:([\d.]+) black_end:([\d.]+)", err)]


def find_silence(path, min_s=MAX_SILENCE_S, threshold="-45dB"):
    err = _run(["ffmpeg", "-hide_banner", "-nostats", "-i", str(path),
                "-af", f"silencedetect=n={threshold}:d={min_s}", "-f", "null", "-"]).stderr
    starts = [float(x) for x in re.findall(r"silence_start: ([\d.]+)", err)]
    ends = [float(x) for x in re.findall(r"silence_end: ([\d.]+)", err)]
    return list(zip(starts, ends))


# How much of a video may sit on a blurred plate before it is worth a look.
# A plate is legitimate for a genuinely odd-shaped archive still; a lot of them
# means something upstream stopped filling the frame.
PLATED_FAIL_FRACTION = 0.15


def plated_frames(scenes):
    """Scenes whose source was inset on a blurred plate rather than filling.

    Read from what core.grade actually did. An earlier version tried to find
    these in the finished video by looking for a dim, low-detail band at the
    edges, and flagged eleven of sixteen frames in a video that was correct:
    a flat night sky, a bright hazy horizon and every infographic card all look
    exactly like that from the outside.
    """
    from core import grade
    plated = {str(p) for p in grade.plated}
    return [s for s in scenes if str(s.get("image")) in plated]


def contact_sheet(video, out_path, every_s=15, cols=5, rows=6, width=400):
    _run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(video),
          "-vf", f"fps=1/{every_s},scale={width}:-1,"
                 f"tile={cols}x{rows}:margin=8:padding=6:color=0x10131c",
          "-frames:v", "1", "-q:v", "3", str(out_path)])
    return Path(out_path).exists()


def check_video(video_path, thumbnail_path=None, srt_path=None,
                expected_min_s=None, expected_max_s=None):
    """Returns (report dict, list of problems)."""
    problems, report = [], {}
    meta = probe(video_path)
    v = next((s for s in meta["streams"] if s["codec_type"] == "video"), None)
    a = next((s for s in meta["streams"] if s["codec_type"] == "audio"), None)
    duration = float(meta["format"]["duration"])

    report["duration_s"] = round(duration, 2)
    report["duration_min"] = round(duration / 60, 2)
    report["size_mb"] = round(int(meta["format"]["size"]) / 1e6, 1)

    if not v:
        problems.append("no video stream")
    else:
        w, h = int(v["width"]), int(v["height"])
        num, den = (v["r_frame_rate"].split("/") + ["1"])[:2]
        fps = float(num) / float(den)
        report["resolution"] = f"{w}x{h}"
        report["fps"] = round(fps, 2)
        report["video_codec"] = v["codec_name"]
        if (w, h) != (MIN_W, MIN_H):
            problems.append(f"resolution is {w}x{h}, expected {MIN_W}x{MIN_H}")
        if abs(fps - FPS) > 0.1:
            problems.append(f"frame rate is {fps:.2f}, expected {FPS}")

    if not a:
        problems.append("no audio stream")
    else:
        report["audio_codec"] = a["codec_name"]
        report["sample_rate"] = int(a["sample_rate"])
        report["channels"] = int(a["channels"])
        if int(a["sample_rate"]) != 48000:
            problems.append(f"sample rate is {a['sample_rate']}, expected 48000")

    lufs, lra = measure_loudness(video_path)
    report["lufs"] = lufs
    report["lra"] = lra
    if lufs is None:
        problems.append("could not measure loudness")
    elif abs(lufs - TARGET_LUFS) > LUFS_TOLERANCE:
        problems.append(f"loudness {lufs} LUFS is outside "
                        f"{TARGET_LUFS}+/-{LUFS_TOLERANCE}")

    blacks = find_black(video_path)
    report["black_segments"] = [{"start": s, "end": e} for s, e in blacks]
    if blacks:
        problems.append(f"{len(blacks)} black segment(s) over {MAX_BLACK_S}s")

    # Silence at the very start or end is normal padding; only mid-video matters.
    silences = [(s, e) for s, e in find_silence(video_path)
                if s > 2.0 and e < duration - 2.0]
    report["silent_gaps"] = [{"start": round(s, 1), "end": round(e, 1)}
                             for s, e in silences]
    if silences:
        problems.append(f"{len(silences)} silent gap(s) over {MAX_SILENCE_S}s mid-video")

    if expected_min_s and duration < expected_min_s:
        problems.append(f"duration {duration / 60:.1f} min is under the "
                        f"{expected_min_s / 60:.0f} min target")
    if expected_max_s and duration > expected_max_s:
        problems.append(f"duration {duration / 60:.1f} min is over the "
                        f"{expected_max_s / 60:.0f} min target")

    if thumbnail_path:
        tp = Path(thumbnail_path)
        if not tp.exists():
            problems.append("thumbnail missing")
        else:
            from PIL import Image
            with Image.open(tp) as im:
                tw, th = im.size
            size = tp.stat().st_size
            report["thumbnail"] = {"size": f"{tw}x{th}", "bytes": size}
            if (tw, th) != (THUMB_W, THUMB_H):
                problems.append(f"thumbnail is {tw}x{th}, expected {THUMB_W}x{THUMB_H}")
            if size > THUMB_MAX_BYTES:
                problems.append(f"thumbnail is {size / 1e6:.1f} MB, over YouTube's 2 MB")

    if srt_path:
        sp = Path(srt_path)
        if not sp.exists():
            problems.append("subtitles missing")
        else:
            cues = sp.read_text(encoding="utf-8").strip().split("\n\n")
            report["subtitle_cues"] = len(cues)
            last = re.findall(r"(\d{2}):(\d{2}):(\d{2}),\d{3}\s*-->",
                              sp.read_text(encoding="utf-8"))
            if last:
                h, m, s = (int(x) for x in last[-1])
                end = h * 3600 + m * 60 + s
                report["subtitle_end_s"] = end
                # Cues should run to roughly the end of the narration.
                if end < duration * 0.85:
                    problems.append(
                        f"subtitles stop at {end / 60:.1f} min but the video runs "
                        f"{duration / 60:.1f} min")

    return report, problems


def write_report(path, report, problems):
    payload = {"checks": report, "problems": problems,
               "verdict": "PASS" if not problems else "ATTENTION"}
    Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    log.info("QC %s: %d problem(s)", payload["verdict"], len(problems))
    return payload
