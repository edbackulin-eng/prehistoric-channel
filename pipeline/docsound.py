"""The sound of the film: narration, a music bed, atmosphere per scene, a few
serious effects, mixed and normalised to YouTube's -14 LUFS, true peak -1 dBTP.

Levels, by measurement (pyloudnorm, BS.1770), not by preset:
  - under speech the music sits 12-14 LU below the narration, so it is heard;
  - in the pauses it comes up by the duck depth, 4-6 dB;
  - the atmosphere sits well under both (AMB_BELOW_VOICE).
The duck is computed here rather than with sidechaincompress: a gain curve
from the narration's envelope (20 ms attack, 450 ms release), which makes its
depth exact and measurable.

Effects are synthesised, low and short - a sub boom, a distant rumble, a soft
air movement - never a cartoon swish or a stinger.
"""
import json
import subprocess
from pathlib import Path

import numpy as np
import pyloudnorm as pyln
import soundfile as sf

from core.logger import get_logger

log = get_logger("docsound")

ROOT = Path(__file__).resolve().parent.parent
SR = 48000
TARGET_LUFS, TRUE_PEAK = -14.0, -1.0
MUSIC_UNDER_VOICE = 13.0          # LU, measured under speech
DUCK_DB = 5.0
AMB_BELOW_VOICE = {"ocean": 20, "beach": 20, "harbour": 21, "underwater": 21, "forest": 22,
                   "swamp": 22, "wind": 22, "fire": 21, "room": 26}


def load(path, sr=SR):
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-ac", "2", "-ar", str(sr),
                          "-f", "f32le", "-"], capture_output=True).stdout
    return np.frombuffer(raw, np.float32).reshape(-1, 2).copy()


def lufs(x, meter=None):
    meter = meter or pyln.Meter(SR)
    try:
        return float(meter.integrated_loudness(x))
    except ValueError:
        return -70.0


def gain_db(x, db):
    return x * (10 ** (db / 20))


# -------------------------------------------------------------- synthesis
def _rng(seed):
    return np.random.default_rng(seed)


def _lowpass(x, hz):
    from scipy.signal import butter, sosfilt
    return sosfilt(butter(4, hz, "low", fs=SR, output="sos"), x, axis=0)


def _bandpass(x, lo, hi):
    from scipy.signal import butter, sosfilt
    return sosfilt(butter(2, [lo, hi], "band", fs=SR, output="sos"), x, axis=0)


def synth_bed(kind, seconds, seed=0):
    n = int(seconds * SR)
    r = _rng(seed)
    t = np.arange(n) / SR
    if kind == "wind":
        x = _bandpass(r.standard_normal((n, 2)), 120, 1800)
        gust = 0.55 + 0.45 * np.sin(2 * np.pi * 0.07 * t + 1.3) * np.sin(2 * np.pi * 0.023 * t)
        return (x * gust[:, None] * 0.25).astype(np.float32)
    if kind == "underwater":
        x = _lowpass(np.cumsum(r.standard_normal((n, 2)), axis=0) * 0.02, 380)
        x -= _lowpass(x, 20)
        swell = 0.75 + 0.25 * np.sin(2 * np.pi * 0.05 * t)[:, None]
        bubbles = np.zeros((n, 2))
        for _ in range(int(seconds * 0.6)):
            at = r.integers(0, max(1, n - SR))
            f = r.uniform(300, 900)
            ln = int(SR * r.uniform(0.04, 0.12))
            env = np.exp(-np.linspace(0, 6, ln))
            sweep = np.sin(2 * np.pi * np.cumsum(np.linspace(f, f * 1.6, ln)) / SR) * env * 0.05
            bubbles[at:at + ln, r.integers(0, 2)] += sweep
        return (x * swell * 0.6 + bubbles).astype(np.float32)
    raise ValueError(kind)


def boom(seconds=3.5, seed=1):
    n = int(seconds * SR)
    t = np.arange(n) / SR
    f = 55 * np.exp(-t * 0.35) + 28
    tone = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t * 1.4)
    noise = _lowpass(_rng(seed).standard_normal(n), 160) * np.exp(-t * 2.2) * 0.6
    x = tone + noise
    x[: int(0.01 * SR)] *= np.linspace(0, 1, int(0.01 * SR))
    return np.repeat((x / np.abs(x).max())[:, None], 2, 1).astype(np.float32)


def rumble(seconds=6.0, seed=2):
    n = int(seconds * SR)
    t = np.arange(n) / SR
    x = _lowpass(np.cumsum(_rng(seed).standard_normal((n, 2)), axis=0) * 0.01, 110)
    x -= _lowpass(x, 18)
    env = np.minimum(1, t / 1.5) * np.minimum(1, (seconds - t) / 2.0)
    return (x / np.abs(x).max() * env[:, None]).astype(np.float32)


def air(seconds=2.4, seed=3):
    """A slow rising breath of filtered noise: the fireball passing, far off."""
    n = int(seconds * SR)
    t = np.arange(n) / SR
    x = _rng(seed).standard_normal((n, 2))
    out = np.zeros_like(x)
    step = SR // 20
    for i in range(0, n, step):                       # sweep the band upward
        u = i / n
        lo, hi = 150 + 500 * u, 600 + 2600 * u
        seg = _bandpass(x[max(0, i - step):i + step], lo, hi)[-min(step, n - i):]
        out[i:i + len(seg)] = seg
    env = np.sin(np.pi * np.minimum(1, t / seconds)) ** 2
    return (out / np.abs(out).max() * env[:, None]).astype(np.float32)


# ------------------------------------------------------------------- mix
def envelope_gain(voice_mono, depth_db=DUCK_DB, attack=0.02, release=0.45, thresh_db=-38):
    hop = SR // 100
    frames = voice_mono[: len(voice_mono) // hop * hop].reshape(-1, hop)
    rms_db = 20 * np.log10(np.sqrt((frames ** 2).mean(1)) + 1e-9)
    active = (rms_db > thresh_db).astype(np.float32)
    # hold through the gaps between words
    hold = int(0.25 * 100)
    act = np.convolve(active, np.ones(hold), "same") > 0
    g = np.zeros(len(act))
    a_c, r_c = np.exp(-1 / (attack * 100)), np.exp(-1 / (release * 100))
    level = 0.0
    for i, on in enumerate(act):
        target = 1.0 if on else 0.0
        c = a_c if target > level else r_c
        level = target + (level - target) * c
        g[i] = level
    gain = 10 ** (-depth_db * g / 20)
    gain = np.repeat(gain, hop)
    out = np.ones(len(voice_mono), np.float32)
    out[: len(gain)] = gain
    out[len(gain):] = gain[-1] if len(gain) else 1
    return out, np.repeat(act, hop)


def build(plan, narration_wav, music_path, amb_lib, out_wav, offset_s=0.0, length_s=None):
    """Mix the film's sound into out_wav (48 kHz stereo float). Returns a report.
    offset_s/length_s cut a stretch (for the 30-second voice samples)."""
    total = length_s or plan["total_s"]
    n = int(total * SR)
    meter = pyln.Meter(SR)
    voice = load(narration_wav)
    voice = voice[int(offset_s * SR): int(offset_s * SR) + n]
    voice = np.pad(voice, ((0, max(0, n - len(voice))), (0, 0)))
    v_l = lufs(voice, meter)

    # music: loop if needed, fade in and out
    mus = load(music_path)
    reps = int(np.ceil((n + offset_s * SR) / len(mus))) + 1
    mus = np.concatenate([mus] * reps)[int(offset_s * SR): int(offset_s * SR) + n]
    fade_in = np.minimum(1, np.arange(n) / (SR * (3.0 if offset_s == 0 else 0.5)))
    fade_out = np.minimum(1, (n - np.arange(n)) / (SR * 3.0))
    mus = mus * (fade_in * fade_out)[:, None]
    duck, active = envelope_gain(voice.mean(1))
    # set the bed so that, ducked, it sits MUSIC_UNDER_VOICE under the speech
    speech = np.zeros(n, bool)
    speech[: min(n, len(active))] = active[:n].astype(bool)
    m_speech = lufs(mus[speech], meter) if speech.any() else lufs(mus, meter)
    g = (v_l - MUSIC_UNDER_VOICE + DUCK_DB) - m_speech
    mus = gain_db(mus, g) * np.pad(duck, (0, max(0, n - len(duck))), constant_values=1)[:n, None]

    # atmosphere per scene, crossfaded
    amb = np.zeros((n, 2), np.float32)
    cache = {}
    for s in plan["shots"]:
        kind = s.get("amb")
        if not kind:
            continue
        a, b = int((s["start"] - offset_s) * SR), int((s["end"] - offset_s) * SR)
        a, b = max(0, a - SR), min(n, b + SR)          # one second of overlap each side
        if b <= a:
            continue
        if kind not in cache:
            entry = amb_lib["beds"].get(kind, {})
            if "same_as" in entry:
                entry = amb_lib["beds"][entry["same_as"]]
            if "synth" in entry:
                bed = synth_bed(entry["synth"], 60, seed=len(cache))
            else:
                bed = load(ROOT / "assets" / "ambience" / entry["file"])
            bed = gain_db(bed, (v_l - AMB_BELOW_VOICE.get(kind, 22)) - lufs(bed, meter))
            cache[kind] = bed
        bed = cache[kind]
        seg = np.resize(bed[(a * 7) % max(1, len(bed) - 1):], (b - a, 2)) if len(bed) else np.zeros((b - a, 2))
        ramp = np.ones(b - a, np.float32)
        r = min(SR * 2, (b - a) // 2)
        ramp[:r] = np.linspace(0, 1, r)
        ramp[-r:] = np.linspace(1, 0, r)
        amb[a:b] += seg * ramp[:, None]

    # effects, from the plan
    fx = np.zeros((n, 2), np.float32)
    def put(sig, t, db):
        i = int((t - offset_s) * SR)
        if 0 <= i < n:
            m = min(len(sig), n - i)
            fx[i:i + m] += gain_db(sig[:m], v_l + db - lufs(np.tile(sig, (3, 1)), meter))
    for s in plan["shots"]:
        if s.get("id") == "asteroid_sky":
            put(air(3.0), s["start"] + 0.8, -10)
        if s.get("anim") == "map" and "chicxulub" in s.get("places", []):
            put(boom(4.0), s["start"] + 0.55, -6)
        if s.get("id") == "impact_forest":
            put(rumble(min(7.0, s["end"] - s["start"] + 1)), s["start"], -12)
        if s.get("title"):
            put(boom(3.0, seed=5), s["start"] + 0.5, -12)
        if s["text"].startswith("It is a coelacanth") and s["kind"] == "recon":
            put(boom(2.5, seed=9), s["start"] + 0.3, -16)

    mix = voice + mus + amb + fx
    sf.write(str(out_wav) + ".pre.wav", mix, SR, subtype="FLOAT")
    _loudnorm(str(out_wav) + ".pre.wav", out_wav)
    for _ in range(3):                 # loudnorm lands within ~0.6 LU; close the gap exactly
        m = measure(out_wav)
        err = TARGET_LUFS - m["integrated_lufs"]
        if abs(err) <= 0.2 and m["true_peak_dbtp"] <= TRUE_PEAK:
            break
        tmp = str(out_wav) + ".fix.wav"
        subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(out_wav), "-af",
                        f"volume={err:.2f}dB,aresample={SR * 4},"
                        f"alimiter=limit={10 ** ((TRUE_PEAK - 0.6) / 20):.4f}:attack=2:release=60:level=false,"
                        f"aresample={SR}", "-c:a", "pcm_f32le", tmp], check=True)
        Path(tmp).replace(out_wav)
    rep = {"voice_lufs_pre": round(v_l, 2), "music_gain_db": round(g, 2),
           "music_under_voice_in_speech_lu": round(v_l - lufs(mus[speech], meter), 2),
           "music_under_voice_in_pauses_lu": round(v_l - lufs(mus[~speech], meter), 2)
           if (~speech).sum() > SR * 3 else None,
           "duck_db_set": DUCK_DB,
           "duck_db_measured": round(float(20 * np.log10(duck[: n][~speech[: len(duck)]].max() /
                                                         duck[: n][speech[: len(duck)]].min())), 2),
           **measure(out_wav)}
    log.info("mix: %s", rep)
    return rep


def _loudnorm(src, dst):
    m = subprocess.run(["ffmpeg", "-hide_banner", "-nostats", "-i", src, "-af",
                        f"loudnorm=I={TARGET_LUFS}:TP={TRUE_PEAK - 0.3}:LRA=11:print_format=json",
                        "-f", "null", "-"], capture_output=True, text=True).stderr
    j = json.loads(m[m.rfind("{"): m.rfind("}") + 1])
    af = (f"loudnorm=I={TARGET_LUFS}:TP={TRUE_PEAK - 0.3}:LRA=11:measured_I={j['input_i']}:"
          f"measured_TP={j['input_tp']}:measured_LRA={j['input_lra']}:"
          f"measured_thresh={j['input_thresh']}:offset={j['target_offset']}:linear=true,"
          f"alimiter=limit={10 ** ((TRUE_PEAK - 0.4) / 20):.4f}:level=false,aresample={SR}")
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", src, "-af", af,
                    "-ar", str(SR), "-c:a", "pcm_f32le", str(dst)], check=True)


def measure(path):
    r = subprocess.run(["ffmpeg", "-hide_banner", "-nostats", "-i", str(path), "-af",
                        "ebur128=peak=true", "-f", "null", "-"], capture_output=True, text=True).stderr
    tail = r[r.rfind("Summary:"):]
    def grab(key):
        import re
        m = re.search(key + r":\s+(-?[\d.]+)", tail)
        return float(m.group(1)) if m else None
    return {"integrated_lufs": grab("I"), "true_peak_dbtp": grab("Peak"), "lra": grab("LRA")}
