"""Sound: effects synthesised in code, and the final mix with its checks.

Effects are generated, not downloaded, so there is nothing to license: keys and
clicks, a sub-bass thump, an analog glitch, a dark riser into each reveal, a deep
impact on it, and a low lamp hum under everything.

The mix, in this order:
  1. the voice alone, two-pass loudnorm;
  2. the music bed set 13 LU under the voice and ducked by 4-6 dB while the voice
     speaks, so it is present under speech and clearly heard in pauses;
  3. the effects bus;
  4. the sum, two-pass loudnorm to -14 LUFS with a -1 dBTP ceiling;
  5. a measurement. Anything outside the limits is corrected and measured again,
     and a mix that still fails raises - it is never handed to the uploader.
"""
import json
import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf

from core.logger import get_logger

log = get_logger("sound")

SR = 48000
TARGET_LUFS = -14.0
TRUE_PEAK = -1.0
LUFS_TOL = 0.5
VOICE_LUFS = -16.0
# 13 LU under the voice, inside the 12-14 asked for after the first hook: at 19 the
# music could not be heard at all.
MUSIC_UNDER_VOICE_LU = 13.0
# A gentle duck, 4-6 dB, so the music stays present under speech and is clearly
# there in every pause. Measured by measure_duck() below.
DUCK = "threshold=0.06:ratio=1.8:level_sc=1:attack=30:release=500"

_rng = np.random.default_rng(7)


# --- effects ---------------------------------------------------------------------

def _env(n, attack, decay):
    t = np.arange(n) / SR
    return np.minimum(1.0, t / max(attack, 1e-4)) * np.exp(-t / max(decay, 1e-4))


def _bandnoise(n, lo, hi):
    spec = np.fft.rfft(_rng.standard_normal(n))
    f = np.fft.rfftfreq(n, 1 / SR)
    spec[(f < lo) | (f > hi)] = 0
    x = np.fft.irfft(spec, n)
    return x / (np.abs(x).max() + 1e-9)


def key():
    n = int(SR * 0.05)
    hi = 2500 + _rng.uniform(-600, 600)
    return 0.5 * _bandnoise(n, hi * 0.6, hi * 2.2) * _env(n, 0.0005, 0.008)


def click():
    n = int(SR * 0.08)
    a = _bandnoise(n, 1500, 6000) * _env(n, 0.0003, 0.006)
    b = np.roll(_bandnoise(n, 900, 3500) * _env(n, 0.0003, 0.01), int(SR * 0.03))
    return 0.7 * a + 0.5 * b


# Everything below is meant to sound like a film, not a cartoon: nothing bright,
# nothing that sweeps up in pitch. The first set had a filtered-noise whoosh and a
# pitched "pop" that both read as cartoon effects; they are gone. What replaces
# them lives below 2 kHz and is felt as much as heard.

def _lowpass(x, cutoff):
    spec = np.fft.rfft(x)
    f = np.fft.rfftfreq(len(x), 1 / SR)
    spec *= 1 / (1 + (f / cutoff) ** 4)
    return np.fft.irfft(spec, len(x))


def blip():
    """A warm analog blip for a message arriving: a short low tone, no chirp."""
    n = int(SR * 0.16)
    t = np.arange(n) / SR
    tone = np.sin(2 * np.pi * 220 * t) + 0.35 * np.sin(2 * np.pi * 440 * t)
    return 0.30 * _lowpass(tone * _env(n, 0.004, 0.05), 1200)


def tick():
    n = int(SR * 0.025)
    return 0.22 * _lowpass(_bandnoise(n, 600, 4000) * _env(n, 0.0003, 0.004), 3000)


def thump(seconds=0.9):
    """A soft sub-bass hit: what a cut or an annotation lands on instead of a whoosh."""
    n = int(SR * seconds)
    t = np.arange(n) / SR
    f = 46 + 30 * np.exp(-t / 0.05)
    body = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t / 0.28)
    air = _lowpass(_bandnoise(n, 40, 900), 500) * np.exp(-t / 0.09)
    return 0.6 * body + 0.18 * air


def glitch(seconds=0.24):
    """An analog fault rather than a digital crunch: a tape wow with crackle."""
    n = int(SR * seconds)
    t = np.arange(n) / SR
    wow = np.sin(2 * np.pi * (90 + 40 * np.sin(2 * np.pi * 9 * t)) * t)
    crackle = (_rng.random(n) > 0.995) * _rng.standard_normal(n) * 2.0
    hiss = _lowpass(_bandnoise(n, 200, 6000), 2500) * 0.4
    return 0.85 * _lowpass(wow * 0.5 + crackle + hiss, 2200) * _env(n, 0.004, seconds * 0.5)


def riser(seconds=2.2):
    """A dark riser: low noise and a sub tone swelling, the pitch barely moving."""
    n = int(SR * seconds)
    t = np.linspace(0, 1, n)
    sub = np.sin(2 * np.pi * np.cumsum(38 + 22 * t) / SR)
    noise = _lowpass(_bandnoise(n, 60, 3000), 700) * (0.4 + 0.6 * t)
    return 0.34 * (0.7 * sub + 0.45 * noise) * t ** 2.2


def impact(seconds=2.4):
    """The reveal: a deep sub boom with a long tail."""
    n = int(SR * seconds)
    t = np.arange(n) / SR
    f = 34 + 50 * np.exp(-t / 0.07)
    body = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t / 0.9)
    hit = _lowpass(_bandnoise(n, 30, 1500), 700) * np.exp(-t / 0.06)
    return 0.95 * body + 0.30 * hit


def hum(seconds):
    """Lamp and mains hum under the whole film, very low: the room at 3 a.m."""
    n = int(SR * seconds)
    t = np.arange(n) / SR
    tone = (np.sin(2 * np.pi * 100 * t) + 0.5 * np.sin(2 * np.pi * 200 * t)
            + 0.25 * np.sin(2 * np.pi * 300 * t))
    flutter = 1 + 0.08 * np.sin(2 * np.pi * 0.13 * t)
    room = _lowpass(_rng.standard_normal(n), 400) * 0.6
    return 0.035 * (tone * flutter + room)


MAKERS = {"key": (key, 0.30), "click": (click, 0.50), "blip": (blip, 0.55),
          "tick": (tick, 0.35), "thump": (thump, 0.55), "glitch": (glitch, 0.45),
          "riser": (riser, 0.55), "impact": (impact, 0.85)}


def effects_track(cues, seconds):
    """One mono track with every cue placed at its time."""
    out = np.zeros(int(SR * (seconds + 2)), np.float32)
    last_key = -1.0
    for c in sorted(cues, key=lambda c: c["t"]):
        kind = c["kind"]
        if kind == "key":
            if c["t"] - last_key < 0.045:
                continue                            # cap the typing rate
            last_key = c["t"]
        maker, gain = MAKERS[kind]
        s = maker() * gain
        i = int(SR * c["t"])
        j = min(len(out), i + len(s))
        if 0 <= i < len(out):
            out[i:j] += s[: j - i]
    out = out[: int(SR * seconds)]
    out += hum(seconds)[: len(out)].astype(np.float32)
    return out


# --- loudness --------------------------------------------------------------------

def _run(args):
    r = subprocess.run(["ffmpeg", "-hide_banner", "-nostats", *args],
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(r.stderr[-1500:])
    return r.stderr


def _measure(path, target=TARGET_LUFS, tp=TRUE_PEAK):
    err = _run(["-i", str(path), "-af",
                f"loudnorm=I={target}:TP={tp}:LRA=11:print_format=json", "-f", "null", "-"])
    a, b = err.rfind("{"), err.rfind("}")
    return json.loads(err[a:b + 1])


def loudnorm2(src, dest, target, tp):
    """Two-pass loudnorm to an exact integrated loudness."""
    m = _measure(src, target, tp)
    _run(["-y", "-i", str(src), "-af",
          f"loudnorm=I={target}:TP={tp}:LRA=11:measured_I={m['input_i']}:"
          f"measured_TP={m['input_tp']}:measured_LRA={m['input_lra']}:"
          f"measured_thresh={m['input_thresh']}:offset={m['target_offset']}:linear=true,"
          f"aresample={SR}", "-ar", str(SR), "-ac", "2", str(dest)])
    return dest


def measure(path):
    """Integrated loudness and true peak of a finished file (EBU R128)."""
    err = _run(["-i", str(path), "-af", "ebur128=peak=true:framelog=quiet", "-f", "null", "-"])
    import re
    i = re.findall(r"I:\s*(-?[\d.]+)\s*LUFS", err)
    p = re.findall(r"Peak:\s*(-?[\d.]+)\s*dBFS", err)
    return float(i[-1]), float(p[-1])


def _lufs(path):
    return float(_measure(path, VOICE_LUFS, -1.5)["input_i"])


# --- the mix -----------------------------------------------------------------------

def mix(voice_wav, music_path, cues, seconds, out_path, work):
    work = Path(work)
    voice = loudnorm2(voice_wav, work / "voice_norm.wav", VOICE_LUFS, -1.5)

    fx = effects_track(cues, seconds)
    sf.write(work / "fx.wav", np.stack([fx, fx], 1), SR)

    inputs = ["-i", str(voice), "-i", str(work / "fx.wav")]
    if music_path:
        gain = (VOICE_LUFS - MUSIC_UNDER_VOICE_LU) - _lufs(music_path)
        inputs += ["-stream_loop", "-1", "-i", str(music_path)]
        graph = (f"[2:a]aresample={SR},aformat=channel_layouts=stereo,volume={gain:.2f}dB,"
                 f"atrim=0:{seconds:.3f}[bed];"
                 f"[0:a]asplit=2[v][key];[bed][key]sidechaincompress={DUCK}[ducked];"
                 f"[1:a]volume=-6dB[fx];"
                 f"[v][ducked][fx]amix=inputs=3:duration=first:normalize=0[pre]")
        log.info("music bed gain %.1f dB (bed %.0f LU under the voice)", gain, MUSIC_UNDER_VOICE_LU)
    else:
        graph = "[1:a]volume=-6dB[fx];[0:a][fx]amix=inputs=2:duration=first:normalize=0[pre]"
    _run(["-y", *inputs, "-filter_complex", graph, "-map", "[pre]", "-t", f"{seconds:.3f}",
          "-ar", str(SR), "-ac", "2", str(work / "premix.wav")])

    loudnorm2(work / "premix.wav", out_path, TARGET_LUFS, TRUE_PEAK - 0.3)
    return verify(out_path, work)


def verify(path, work):
    """Measure; correct once if outside the limits; refuse if still outside."""
    lufs, peak = measure(path)
    log.info("mix: %.1f LUFS, true peak %.1f dBTP", lufs, peak)
    if abs(lufs - TARGET_LUFS) <= LUFS_TOL and peak <= TRUE_PEAK:
        return {"lufs": lufs, "true_peak": peak, "corrected": False}
    fixed = Path(work) / "mix_fixed.wav"
    gain = TARGET_LUFS - lufs
    _run(["-y", "-i", str(path), "-af",
          f"volume={gain:.2f}dB,alimiter=limit={10 ** ((TRUE_PEAK - 0.4) / 20):.4f}:"
          f"level=disabled:attack=2:release=60", "-ar", str(SR), str(fixed)])
    lufs2, peak2 = measure(fixed)
    log.info("corrected: %.1f LUFS, true peak %.1f dBTP", lufs2, peak2)
    if abs(lufs2 - TARGET_LUFS) > LUFS_TOL or peak2 > TRUE_PEAK:
        raise RuntimeError(f"mix out of limits after correction: {lufs2:.1f} LUFS, "
                           f"{peak2:.1f} dBTP; not uploading")
    Path(fixed).replace(path)
    return {"lufs": lufs2, "true_peak": peak2, "corrected": True}


def measure_duck(voice_wav, music_path, work, gain_db=None):
    """How far the bed sits under the voice, and how much the duck takes off it.

    The ducked bed is rendered on its own and its level compared between spoken
    and silent stretches, using the voice's own silence map.
    """
    import re
    work = Path(work)
    gain = gain_db if gain_db is not None else (VOICE_LUFS - MUSIC_UNDER_VOICE_LU) - _lufs(music_path)
    voice = loudnorm2(voice_wav, work / "dv.wav", VOICE_LUFS, -1.5)
    dur = sf.info(str(voice)).duration
    _run(["-y", "-i", str(voice), "-stream_loop", "-1", "-i", str(music_path), "-filter_complex",
          f"[1:a]aresample={SR},aformat=channel_layouts=stereo,volume={gain:.2f}dB,atrim=0:{dur:.3f}[b];"
          f"[b][0:a]sidechaincompress={DUCK}[d]", "-map", "[d]", "-t", f"{dur:.3f}", str(work / "bed_ducked.wav")])
    _run(["-y", "-stream_loop", "-1", "-i", str(music_path), "-af",
          f"aresample={SR},aformat=channel_layouts=stereo,volume={gain:.2f}dB", "-t", f"{dur:.3f}",
          str(work / "bed_plain.wav")])
    err = _run(["-i", str(voice), "-af", "silencedetect=n=-40dB:d=0.35", "-f", "null", "-"])
    starts = [float(x) for x in re.findall(r"silence_start: ([\d.]+)", err)]
    ends = [float(x) for x in re.findall(r"silence_end: ([\d.]+)", err)]
    quiet = list(zip(starts, ends))

    def rms_db(path, mask_quiet):
        x, sr = sf.read(str(path))
        x = x.mean(axis=1) if x.ndim > 1 else x
        t = np.arange(len(x)) / sr
        inq = np.zeros(len(x), bool)
        for a, b in quiet:
            inq |= (t >= a + 0.15) & (t <= b - 0.05)
        sel = x[inq] if mask_quiet else x[~inq]
        return 20 * np.log10(np.sqrt(np.mean(sel ** 2)) + 1e-12)

    def lufs_of(path):
        return float(_measure(path, VOICE_LUFS, -1.5)["input_i"])

    v = lufs_of(voice)
    bed = lufs_of(work / "bed_plain.wav")
    # the duck is what the compressor takes off the same music while the voice speaks
    duck = rms_db(work / "bed_plain.wav", False) - rms_db(work / "bed_ducked.wav", False)
    in_pause = rms_db(work / "bed_ducked.wav", True) - rms_db(work / "bed_ducked.wav", False)
    return {"voice_lufs": round(v, 1), "bed_lufs": round(bed, 1),
            "bed_under_voice_lu": round(v - bed, 1), "duck_db": round(duck, 1),
            "bed_louder_in_pauses_db": round(in_pause, 1), "pauses_measured": len(quiet)}
