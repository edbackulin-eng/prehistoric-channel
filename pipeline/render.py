"""Montage: graded stills + Ken Burns + crossfades, married to the narration bed.

Every scene moves. A still frame held for five seconds is the single clearest
signal of a low-effort channel, so each shot gets a slow push or pull with a drift
across the frame, and no two consecutive shots move the same way.
"""
import json
import subprocess
from pathlib import Path

from core import grade
from core.logger import get_logger

log = get_logger("render")

FPS = 30
W, H = grade.OUT_W, grade.OUT_H
XFADE_S = 0.5
ZOOM_RANGE = 1.32          # how far a push travels across a whole scene

# Each scene takes the next move in this cycle, so the motion never repeats twice
# running. (zoom direction, x drift, y drift) in normalised frame units.
MOVES = [
    ("in", 0.0, 0.0), ("out", 0.0, 0.0),
    ("in", 1.0, 0.0), ("out", -1.0, 0.0),
    ("in", 0.0, 1.0), ("out", 0.0, -1.0),
    ("in", -1.0, 0.5), ("out", 1.0, -0.5),
]


def _ffmpeg(args, label):
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", *args]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg {label} failed:\n{result.stderr[-2000:]}")
    return result


def probe_duration(path):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=nw=1:nk=1", str(path)],
        capture_output=True, text=True, check=True)
    return float(out.stdout.strip())


# A drawn card is mostly type, and type that drifts out of frame is just a bug.
# Cards get a slow centred push with no pan, shallow enough that the full card
# stays visible for the whole shot.
GENTLE_ZOOM = 1.05


def _gentle_filter(duration):
    """A barely-there push for frames whose content must stay fully visible."""
    d = max(0.1, duration)
    span = GENTLE_ZOOM - 1.0
    return (f"scale=w='{W}*(1+{span:.4f}*t/{d:.3f})':h=-2:eval=frame:flags=bicubic,"
            f"crop={W}:{H}")


def _kenburns_filter(move, duration):
    """A Ken Burns move for one shot: animated `scale`, then a fixed crop window.

    NOT zoompan. On a CPU like this one zoompan runs at about 0.05x realtime -
    measured, and effectively independent of input resolution - because it
    re-initialises its scaler every frame. Driving `scale` with eval=frame and
    cropping a fixed 1080p window out of the growing image is the same effect at
    about 1.24x realtime, a ~25x difference, and it is what makes a 20-minute
    video renderable here at all.

    `crop` cannot do the zoom itself: its w/h are evaluated once at configure time,
    when `t` does not yet exist, so only x and y can move per frame.
    """
    direction, dx, dy = move
    d = max(0.1, duration)
    span = ZOOM_RANGE - 1.0
    if direction == "in":
        zoom = f"{W}*(1+{span:.4f}*t/{d:.3f})"
    else:
        zoom = f"{W}*({ZOOM_RANGE:.4f}-{span:.4f}*t/{d:.3f})"
    # Drift across the slack the zoom leaves, staying clear of the very edge.
    x = f"(iw-ow)*(0.5+{0.9 * dx:.3f}*(t/{d:.3f}-0.5))"
    y = f"(ih-oh)*(0.5+{0.9 * dy:.3f}*(t/{d:.3f}-0.5))"
    return (f"scale=w='{zoom}':h=-2:eval=frame:flags=bicubic,"
            f"crop={W}:{H}:x='{x}':y='{y}'")


def _static_card_clip(shot, graded_path, clip, work_dir, index):
    """A card clip: the reveal sequence, then the settled card held still.

    The card itself never moves. Panning across type was what cropped the
    numbers, and a slow push on a flat graphic looks like a mistake rather than
    a choice, so the only motion is the figure arriving.
    """
    from pipeline import infographics

    spec = shot.get("card_spec")
    reveal_dir = Path(work_dir) / "reveal" / f"{index:03d}"
    duration = shot["duration"]
    if spec:
        kind, kwargs = spec
        frames, final = infographics.render_reveal(kind, kwargs, reveal_dir, "r")
        reveal_s = frames / infographics.REVEAL_FPS
        hold_s = max(0.1, duration - reveal_s)
        _ffmpeg([
            "-framerate", str(infographics.REVEAL_FPS),
            "-i", str(reveal_dir / "r_%04d.png"),
            "-loop", "1", "-framerate", str(FPS), "-t", f"{hold_s:.3f}",
            "-i", str(final),
            "-filter_complex",
            f"[0:v]scale={W}:{H}:flags=bicubic,fps={FPS},format=yuv420p[a];"
            f"[1:v]scale={W}:{H}:flags=bicubic,fps={FPS},format=yuv420p[b];"
            "[a][b]concat=n=2:v=1:a=0[v]",
            "-map", "[v]", "-c:v", "libx264", "-preset", "veryfast",
            "-crf", "20", "-r", str(FPS), str(clip),
        ], f"card reveal {index}")
        return
    # No spec available (a card rendered by an older run): hold it still.
    _ffmpeg([
        "-loop", "1", "-framerate", str(FPS), "-t", f"{duration:.3f}",
        "-i", str(graded_path),
        "-vf", f"scale={W}:{H}:flags=bicubic,format=yuv420p,"
               f"fade=t=in:st=0:d=0.5",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
        "-r", str(FPS), str(clip),
    ], f"static card {index}")


def build_scene_clips(shots, work_dir):
    """Grade each still and turn it into its own moving clip."""
    work_dir = Path(work_dir)
    (work_dir / "graded").mkdir(parents=True, exist_ok=True)
    clips = []
    for i, shot in enumerate(shots):
        graded_path = work_dir / "graded" / f"{i:03d}.png"
        grade.apply(shot["path"]).save(graded_path)
        clip = work_dir / f"clip_{i:03d}.mp4"
        move = MOVES[i % len(MOVES)]
        if shot.get("motion") == "static":
            # A card holds still and animates its own content instead.
            _static_card_clip(shot, graded_path, clip, work_dir, i)
            clips.append(clip)
            log.info("scene %d: %.2fs, card reveal", i, shot["duration"])
            continue
        if shot.get("motion") == "gentle":
            vf = _gentle_filter(shot["duration"])
            move = ("gentle", 0, 0)
        else:
            vf = _kenburns_filter(move, shot["duration"])
        _ffmpeg([
            "-loop", "1", "-framerate", str(FPS), "-t", f"{shot['duration']:.3f}",
            "-i", str(graded_path),
            "-vf", vf + ",format=yuv420p",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-r", str(FPS),
            str(clip),
        ], f"scene {i}")
        clips.append(clip)
        log.info("scene %d: %.2fs, move=%s", i, shot["duration"], move[0])
    return clips


# One ffmpeg call can only hold so many inputs and filter nodes before the graph
# becomes unmanageable, and a 20-minute video is over 200 clips. Fusing them in
# groups and then fusing the groups keeps every single call small.
CROSSFADE_GROUP = 16


def _crossfade_once(clips, out_path, xfade_s, crf):
    """Chain a handful of clips with crossfades in a single ffmpeg call."""
    if len(clips) == 1:
        _ffmpeg(["-i", str(clips[0]), "-c", "copy", str(out_path)], "single clip")
        return out_path
    inputs = []
    for c in clips:
        inputs += ["-i", str(c)]
    durations = [probe_duration(c) for c in clips]
    steps, prev, offset = [], "[0:v]", 0.0
    for i in range(1, len(clips)):
        offset += durations[i - 1] - xfade_s
        label = f"[x{i}]"
        steps.append(f"{prev}[{i}:v]xfade=transition=fade:"
                     f"duration={xfade_s}:offset={offset:.3f}{label}")
        prev = label
    _ffmpeg([*inputs, "-filter_complex", ";".join(steps), "-map", prev,
             "-c:v", "libx264", "-preset", "veryfast", "-crf", str(crf),
             "-pix_fmt", "yuv420p", "-r", str(FPS), str(out_path)], "crossfade")
    return out_path


def crossfade(clips, out_path, xfade_s=XFADE_S, work_dir=None, crf=20):
    """Crossfade any number of clips, fusing in groups when there are many.

    Each level re-encodes, so the intermediate CRF is kept a little tighter than
    the final one to stop two passes of generation loss showing.
    """
    clips = [Path(c) for c in clips]
    if len(clips) <= CROSSFADE_GROUP:
        _crossfade_once(clips, out_path, xfade_s, crf)
        log.info("crossfaded %d clips -> %.1fs", len(clips), probe_duration(out_path))
        return out_path

    work_dir = Path(work_dir or Path(out_path).parent) / "fuse"
    work_dir.mkdir(parents=True, exist_ok=True)
    level, current = 0, clips
    while len(current) > CROSSFADE_GROUP:
        level += 1
        merged = []
        for i in range(0, len(current), CROSSFADE_GROUP):
            group = current[i:i + CROSSFADE_GROUP]
            if len(group) == 1:
                merged.append(group[0])
                continue
            part = work_dir / f"L{level}_{i // CROSSFADE_GROUP:03d}.mp4"
            _crossfade_once(group, part, xfade_s, crf - 1)
            merged.append(part)
        log.info("fuse level %d: %d -> %d", level, len(current), len(merged))
        current = merged
    _crossfade_once(current, out_path, xfade_s, crf)
    log.info("crossfaded %d clips -> %.1fs", len(clips), probe_duration(out_path))
    return out_path


SAMPLE_RATE = 48000       # what YouTube wants; loudnorm otherwise emits 192k
TARGET_LUFS = -14.0

# Ducking, tuned by measurement rather than by the usual defaults. `level_sc` is
# the parameter that actually matters: the narration arrives as the key at well
# under the compressor's threshold, so without scaling the key up the compressor
# barely engages. At the documented-looking settings (threshold 0.05, ratio 9,
# level_sc 1) the bed drops by 1.2 dB - effectively no ducking at all. Scaling the
# key by 10 gives ~19.5 dB on normal speech, inside the -18 to -22 dB target, and
# the bed returns to full level in every pause.
# How far below the narration the music bed sits when nobody is speaking.
# Measured, not guessed: at 13 LU the bed reads as atmosphere on laptop
# speakers and is still there on headphones.
MUSIC_BELOW_VOICE_LU = 13.0

# A soft duck, measured at 6.7-7.8 dB on two segments of episode one. The
# previous setting (ratio=20, level_sc=10) measured 26.2 dB: with narration
# running almost continuously, that left the bed around -50 dB under speech,
# audible only in the one-second gaps between paragraphs.
DUCK = "threshold=0.05:ratio=3:level_sc=2:attack=20:release=400"


def _measure_loudness(inputs, graph, out_label):
    """First loudnorm pass: measure the mix so the second pass can hit the target.

    Single-pass loudnorm only estimates, and on short programme material it lands
    1-2 LU off. Measuring first and feeding the numbers back gets it exact.
    """
    args = []
    for src in inputs:
        args += ["-i", str(src)]
    measure = (f"{graph};[{out_label}]loudnorm=I={TARGET_LUFS}:TP=-1.5:LRA=11:"
               "print_format=json[m]")
    result = subprocess.run(
        ["ffmpeg", "-hide_banner", "-nostats", *args,
         "-filter_complex", measure, "-map", "[m]", "-f", "null", "-"],
        capture_output=True, text=True)
    text = result.stderr
    start = text.rfind("{")
    end = text.rfind("}")
    if start == -1 or end <= start:
        raise RuntimeError(f"loudnorm measurement produced no JSON:\n{text[-1500:]}")
    return json.loads(text[start:end + 1])


def _integrated_lufs(path):
    """Integrated loudness of a single file, for setting the bed relative to it."""
    result = subprocess.run(
        ["ffmpeg", "-hide_banner", "-nostats", "-i", str(path),
         "-af", f"loudnorm=I={TARGET_LUFS}:TP=-1.5:LRA=11:print_format=json",
         "-f", "null", "-"], capture_output=True, text=True)
    text = result.stderr
    start, end = text.rfind("{"), text.rfind("}")
    if start == -1 or end <= start:
        raise RuntimeError(
            f"could not measure {path}:\n{text[-800:]}")
    return float(json.loads(text[start:end + 1])["input_i"])


def bed_gain_db(narration, music, below=MUSIC_BELOW_VOICE_LU):
    """How much to attenuate this music so it sits `below` LU under the voice.

    A fixed -19 dB was tuned against one track and one narration, so any other
    track would land at its own arbitrary level - which matters now that the
    bed is chosen per episode by mood. Measuring both sides makes the result
    the same for every track.
    """
    voice, bed = _integrated_lufs(narration), _integrated_lufs(music)
    gain = (voice - below) - bed
    log.info("bed gain: voice %.1f LUFS, music %.1f LUFS -> %.1f dB "
             "(bed %.0f LU under voice)", voice, bed, gain, below)
    return gain


def mix_audio(narration, music, out_path, music_gain_db=None):
    """Narration over a gently ducked music bed, normalised to broadcast loudness.

    The bed is set relative to the measured loudness of the narration, then
    sidechaincompress eases it down a further ~7 dB while the narrator speaks.
    The point is a bed that is continuously present rather than one that only
    surfaces between paragraphs.
    """
    if music is not None and music_gain_db is None:
        music_gain_db = bed_gain_db(narration, music)
    if music is None:
        inputs = [narration]
        graph = f"[0:a]aresample={SAMPLE_RATE},aformat=channel_layouts=stereo[pre]"
    else:
        inputs = [narration, music]
        graph = (
            f"[1:a]aresample={SAMPLE_RATE},aformat=channel_layouts=stereo,"
            f"volume={music_gain_db}dB,aloop=loop=-1:size=2e9[bed];"
            f"[0:a]aresample={SAMPLE_RATE},aformat=channel_layouts=stereo,"
            "asplit=2[voice][key];"
            f"[bed][key]sidechaincompress={DUCK}[ducked];"
            "[voice][ducked]amix=inputs=2:duration=first:dropout_transition=0:"
            "normalize=0[pre]"
        )

    stats = _measure_loudness(inputs, graph, "pre")
    log.info("measured %s LUFS, TP %s, LRA %s", stats["input_i"],
             stats["input_tp"], stats["input_lra"])
    second = (
        f"{graph};[pre]loudnorm=I={TARGET_LUFS}:TP=-1.5:LRA=11:"
        f"measured_I={stats['input_i']}:measured_TP={stats['input_tp']}:"
        f"measured_LRA={stats['input_lra']}:measured_thresh={stats['input_thresh']}:"
        f"offset={stats['target_offset']}:linear=true,"
        f"aresample={SAMPLE_RATE}[out]"
    )
    args = []
    for src in inputs:
        args += ["-i", str(src)]
    _ffmpeg([*args, "-filter_complex", second, "-map", "[out]",
             "-ar", str(SAMPLE_RATE), "-ac", "2",
             "-c:a", "aac", "-b:a", "192k", str(out_path)], "audio mix")
    return out_path


def mux(video, audio, out_path):
    _ffmpeg(["-i", str(video), "-i", str(audio),
             "-map", "0:v", "-map", "1:a", "-c:v", "copy", "-c:a", "copy",
             "-shortest", str(out_path)], "mux")
    log.info("wrote %s (%.1fs)", Path(out_path).name, probe_duration(out_path))
    return out_path


def plan_scenes(image_paths, total_duration, min_s=4.0, max_s=7.0):
    """Spread the narration across the shots, 4-7s each, varied so the cutting
    rhythm does not become metronomic."""
    n = len(image_paths)
    base = total_duration / n
    if not (min_s <= base <= max_s):
        log.warning("%.2fs per shot is outside the %.0f-%.0fs target; "
                    "adjust the shot count", base, min_s, max_s)
    # Alternate slightly shorter and longer around the average.
    swing = min(0.9, (max_s - min_s) / 4)
    shots = []
    for i, path in enumerate(image_paths):
        delta = swing if i % 2 else -swing
        shots.append({"path": str(path),
                      "duration": max(min_s * 0.8, base + delta)})
    # Crossfades eat xfade_s per join; add it back so the montage still covers
    # the narration rather than finishing early.
    for s in shots[:-1]:
        s["duration"] += XFADE_S * (n - 1) / n
    return shots


def write_report(path, shots, video_path, audio_path):
    report = {
        "video": str(video_path),
        "duration_s": round(probe_duration(video_path), 2),
        "scenes": len(shots),
        "fps": FPS,
        "resolution": f"{W}x{H}",
        "scene_durations": [round(s["duration"], 2) for s in shots],
    }
    Path(path).write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
