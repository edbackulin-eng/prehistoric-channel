"""Put the edit together: shots, transitions, accents, and the light VHS layer.

Shots are rendered one by one to exact length, then joined by a plain concat, so
the picture can never drift from the narration. Transitions take turns:

  vhs   a hard cut through a fifth of a second of tracking noise and colour bleed
  rgb   a hard cut with the colour channels pulled apart and snapping back
  fade  a real crossfade, baked into the head of the incoming shot

and are all applied to the head of the incoming shot, so every shot keeps its own
length. The accents (punch-in on reveals, a shake on the last one, kinetic words
synced to Whisper's timestamps) and a light VHS layer go on in one final pass.
"""
import json
import subprocess
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter

from core import fonts
from core.logger import get_logger
from pipeline import anim, motion

log = get_logger("assemble")

W, H, FPS = 1920, 1080, 30
FADE_S = 0.35
GLITCH_S = 0.22


def _ff(*args):
    r = subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", *args],
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(r.stderr[-1500:])


def _x264():
    return ["-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p",
            "-r", str(FPS)]


# --- one shot -----------------------------------------------------------------

def render_shot(shot, frames, work, still_cache):
    """Render one planned shot to work/shot_NNN.mp4. Returns (path, sound cues)."""
    i, d = shot["move"], shot["end"] - shot["start"]
    if shot.get("transition") == "fade":
        d += FADE_S                        # the overlap eaten by the crossfade
    out = work / f"shot_{i:03d}.mp4"
    t = shot["type"]
    bg = frames.get(i)

    def prepared():
        if i not in still_cache:
            still_cache[i] = motion.prepare(bg, work / f"still_{i:03d}.jpg")
        return still_cache[i]

    if t == "flux":
        motion.kenburns(prepared(), out, d, index=i)
        return out, []
    if t == "stock":
        a = shot["asset"]
        if a["kind"] == "image":
            if i not in still_cache:
                still_cache[i] = motion.prepare(a["path"], work / f"still_{i:03d}.jpg")
            motion.kenburns(still_cache[i], out, d, index=i)
        else:
            motion.stock_clip(a["path"], out, d, a.get("duration", 0), seed=i)
        return out, []
    if t == "map":
        r = anim.worldmap(out, d, [tuple(x) for x in shot["places"]])
        return out, [{"t": round(shot["start"] + c["t"], 3), "kind": c["kind"]} for c in r["cues"]]
    if t == "annotate":
        r = anim.annotate(out, d, bg, circle=tuple(shot["circle"]))
    elif t == "browser":
        r = anim.browser(out, d, shot["address"], shot["heading"], shot["body"],
                         title=shot.get("title", "New tab"))
    elif t == "chat":
        r = anim.chat(out, d, [tuple(m) for m in shot["messages"]][:3],
                      header=shot.get("header", "Thread"))
    elif t == "notification":
        r = anim.notification(out, d, bg, shot["app"], shot["text"])
    elif t == "cursor":
        r = anim.cursor_click(out, d, bg)
    elif t == "counter":
        r = anim.counter(out, d, shot["label"], int(shot["end"]),
                         prefix=shot.get("prefix", ""), suffix=shot.get("suffix", ""))
    elif t == "timeline":
        r = anim.timeline(out, d, [tuple(e) for e in shot["events"]][:4])
    elif t == "quote":
        r = anim.quote(out, d, shot["text"])
    elif t == "censor":
        r = anim.censor(out, d, shot.get("heading", "file"), shot["lines"][:4], shot["redact"][:3])
    else:
        raise ValueError(f"unknown shot type {t}")
    return out, [{"t": round(shot["start"] + c["t"], 3), "kind": c["kind"]} for c in r["cues"]]


# --- transitions, baked into the head of the incoming shot ---------------------

def _glitch_filter(kind):
    w = GLITCH_S
    if kind == "vhs":
        # tracking noise, chroma bleed and a short horizontal tear
        return (f"rgbashift=rh=-14:bh=14:enable='lt(t,{w})',"
                f"noise=alls=55:allf=t:enable='lt(t,{w})',"
                f"eq=brightness=0.06:saturation=1.6:enable='lt(t,{w * 0.6})',"
                f"gblur=sigma=1.2:steps=1:enable='lt(t,{w})'")
    # rgb: channels pulled apart, snapping back over the window
    return (f"rgbashift=rh=-22:gh=0:bh=22:enable='lt(t,{w * 0.4})',"
            f"rgbashift=rh=-10:bh=10:enable='between(t,{w * 0.4},{w})'")


def apply_transitions(paths, shots, work):
    out = list(paths)
    for k in range(1, len(paths)):
        kind = shots[k].get("transition")
        if not kind:
            continue
        cur = work / f"tx_{k:03d}.mp4"
        if kind == "fade":
            prev_d = shots[k - 1]["end"] - shots[k - 1]["start"]
            if shots[k - 1].get("transition") == "fade":
                prev_d += FADE_S
            # the incoming shot was rendered FADE_S long; fade it over the
            # outgoing shot's last FADE_S, then trim that tail off the outgoing one
            _ff("-i", str(out[k - 1]), "-i", str(out[k]), "-filter_complex",
                f"[0:v]trim=start={prev_d - FADE_S:.3f},setpts=PTS-STARTPTS[a];"
                f"[a][1:v]xfade=transition=fade:duration={FADE_S}:offset=0[v]",
                "-map", "[v]", *_x264(), str(cur))
            trimmed = work / f"trim_{k - 1:03d}.mp4"
            _ff("-i", str(out[k - 1]), "-t", f"{prev_d - FADE_S:.3f}", *_x264(), str(trimmed))
            out[k - 1] = trimmed
        else:
            _ff("-i", str(out[k]), "-vf", _glitch_filter(kind), *_x264(), str(cur))
        out[k] = cur
    return out


def concat(paths, out, work):
    lst = work / "concat.txt"
    lst.write_text("".join(f"file '{Path(p).resolve().as_posix()}'\n" for p in paths),
                   encoding="utf-8")
    _ff("-f", "concat", "-safe", "0", "-i", str(lst), "-c", "copy", str(out))
    return out


# --- kinetic words ---------------------------------------------------------------

def kinetic_png(words, path, red_index=None):
    """1-3 words in Anton: white, one red, heavy black outline. Transparent PNG."""
    text = " ".join(w.upper().strip(".,!?;:\"'") for w in words)
    size, _ = fonts.fit(text, "title", W * 0.7, H * 0.26, 210)
    f = fonts.font("title", size)
    im = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    parts = text.split()
    red = len(parts) - 1 if red_index is None else red_index
    total = sum(f.getlength(p) for p in parts) + f.getlength(" ") * (len(parts) - 1)
    x = (W - total) / 2
    y = H * 0.62
    shadow = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    ImageDraw.Draw(shadow).rectangle((x - 40, y - 20, x + total + 40, y + size * 1.15), fill=(0, 0, 0, 120))
    im = Image.alpha_composite(im, shadow.filter(ImageFilter.GaussianBlur(30)))
    d = ImageDraw.Draw(im)
    for k, p in enumerate(parts):
        d.text((x, y), p, font=f, fill=(232, 44, 52) if k == red else (255, 255, 255),
               stroke_width=max(4, size // 22), stroke_fill=(0, 0, 0))
        x += f.getlength(p + " ")
    im.save(path)
    return path


# --- the final pass ----------------------------------------------------------------

def _window_sum(windows, rise=0.25):
    """An ffmpeg expression that is 0 outside the windows and eases to 1 inside."""
    terms = []
    for a, b in windows:
        terms.append(f"(between(t,{a:.3f},{b:.3f})*min(1,(t-{a:.3f})/{rise})*min(1,({b:.3f}-t)/{rise}))")
    return "+".join(terms) if terms else "0"


# One profile and level for every encode of a final picture: parts rendered on
# different machines are joined with -c copy, which needs identical streams.
H264 = ["-profile:v", "high", "-level:v", "4.1", "-g", "60"]


def finish(video, audio, accents, out, work, vhs=True, duration=None):
    """Punch-in, shake, kinetic words, VHS layer, and the mixed sound."""
    punch = _window_sum(accents.get("punch", []))
    shake = _window_sum(accents.get("shake", []), rise=0.05)
    z = f"(1+0.06*({punch}))"
    # The frame is scaled up for the punch and cropped back to size; the crop
    # centre is offset by a small fast sine during a shake.
    chain = [f"scale=w='trunc({W}*{z}/2)*2':h='trunc({H}*{z}/2)*2':eval=frame:flags=bicubic",
             f"crop={W}:{H}:x='(iw-{W})/2+9*({shake})*sin(t*71)':y='(ih-{H})/2+6*({shake})*cos(t*83)'"]
    if vhs:
        # Saturation is kept, not pulled down, and the vignette is light: the grade
        # (pipeline/grade.py) made the frames colourful and this pass must not undo it.
        chain += ["noise=alls=6:allf=t", "eq=saturation=1.05:contrast=1.05:brightness=0.015",
                  "vignette=angle=PI/8"]
    inputs = ["-i", str(video)] + (["-i", str(audio)] if audio else [])
    first = len(inputs) // 2                   # index of the first overlay input
    graph = f"[0:v]{','.join(chain)}[base]"
    last = "base"
    for n, k in enumerate(accents.get("kinetic", [])):
        png = kinetic_png(k["words"], work / f"kinetic_{n:02d}.png")
        inputs += ["-loop", "1", "-t", f"{k['end'] - k['start'] + 0.4:.3f}", "-i", str(png)]
        idx = first + n
        dur = k["end"] - k["start"]
        graph += (f";[{idx}:v]format=rgba,fade=t=in:st=0:d=0.12:alpha=1,"
                  f"fade=t=out:st={max(0.1, dur - 0.2):.3f}:d=0.2:alpha=1,"
                  f"setpts=PTS+{k['start']:.3f}/TB[k{n}];"
                  f"[{last}][k{n}]overlay=0:0:enable='between(t,{k['start']:.3f},{k['end']:.3f})'[o{n}]")
        last = f"o{n}"
    if vhs:
        lines = work / "scanlines.png"
        _scanlines(lines)
        inputs += ["-loop", "1", "-i", str(lines)]
        graph += f";[{last}][{first + len(accents.get('kinetic', []))}:v]overlay=0:0:shortest=1[v]"
        last = "v"
    # Capped: the film-grain noise of the VHS layer is incompressible, and left to
    # crf alone the hook came out at 42 Mbit/s (329 MB a minute). 12 Mbit/s is
    # above YouTube's own 1080p30 recommendation and keeps a 13-minute video
    # around 1.2 GB.
    sound_args = (["-map", "1:a", "-c:a", "aac", "-b:a", "192k", "-shortest"] if audio
                  else ["-an", "-t", f"{duration:.3f}"] if duration else ["-an"])
    _ff(*inputs, "-filter_complex", graph, "-map", f"[{last}]",
        "-c:v", "libx264", "-preset", "fast", "-crf", "20", "-maxrate", "12M",
        "-bufsize", "24M", *H264, "-pix_fmt", "yuv420p", "-r", str(FPS),
        *sound_args, "-movflags", "+faststart", str(out))
    return out


def _scanlines(path):
    if Path(path).exists():
        return path
    im = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    for y in range(0, H, 3):
        d.line((0, y, W, y), fill=(0, 0, 0, 28))
    im.save(path)
    return path
