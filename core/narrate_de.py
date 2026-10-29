"""German documentary narration with Piper (de_DE-thorsten-high), on the CPU.

Same contract as core/narrate.py (Kokoro, English): reads the delivery markup,
calibrates its pace over the whole programme, and returns the audio with every
line's start and end, so the edit cuts to the words without an alignment pass.

German runs to longer words than English, so the same documentary pace is
fewer words a minute: about 128, where the English channel reads 150.

Piper (GPL-3.0 engine) and the Thorsten voice (dataset and model CC0) run in a
few seconds per minute of audio on a GitHub runner; no GPU is involved.
"""
import re
import time
from pathlib import Path

import numpy as np

from core import delivery, speech_de
from core.logger import get_logger

log = get_logger("narrate_de")

ROOT = Path(__file__).resolve().parent.parent
MODEL = ROOT / "assets" / "models" / "piper" / "de_DE-thorsten-high.onnx"
VOICE = "piper_thorsten"
TARGET_WPM = 128.0
SPEED_MIN, SPEED_MAX = 0.68, 1.15
SLOW_LEAD = (0.6, 0.9)
SENTENCE_TAIL_S = 0.32
QUESTION_EXTRA_S = 0.12
_SENT = re.compile(r"(?<=[.!?])\s+")


def load():
    from piper import PiperVoice
    return PiperVoice.load(str(MODEL))


def _lead_for(line):
    n = len(line.text.split())
    lo, hi = SLOW_LEAD
    return round(lo + (hi - lo) * min(1.0, n / 10), 2)


def _trim(a, sr, thr=0.008):
    idx = np.nonzero(np.abs(a) > thr)[0]
    if not len(idx):
        return a
    return a[max(0, idx[0] - int(0.02 * sr)): idx[-1] + int(0.05 * sr)]


def sentence(voice, text, speed, lex=None):
    from piper import SynthesisConfig
    cfg = SynthesisConfig(length_scale=1.0 / speed)
    chunks = [c.audio_float_array for c in voice.synthesize(speech_de.for_piper(text, lex), cfg)]
    return _trim(np.concatenate(chunks).astype(np.float32), voice.config.sample_rate)


def render(voice, lines, scale, lex=None):
    sr = voice.config.sample_rate
    lex = lex if lex is not None else speech_de.lexicon()
    chunks, timeline, t = [], [], 0.0
    for line in lines:
        pause = line.pause_before
        if line.kind == "slow":
            pause = max(pause, _lead_for(line))
        if pause:
            chunks.append(np.zeros(int(sr * pause), np.float32))
            t += pause
        speed = min(SPEED_MAX, max(SPEED_MIN, line.speed * scale))
        start = t
        for s in [s for s in _SENT.split(line.text) if s.strip()]:
            a = sentence(voice, s, speed, lex)
            chunks.append(a)
            t += len(a) / sr
            tail = SENTENCE_TAIL_S + (QUESTION_EXTRA_S if s.rstrip().endswith("?") else 0)
            chunks.append(np.zeros(int(sr * tail), np.float32))
            t += tail
        timeline.append({"text": line.text, "kind": line.kind, "section": line.section, "start": round(start, 3),
                         "end": round(t, 3), "speed": round(speed, 3)})
    return np.concatenate(chunks), timeline


def wpm(script, seconds):
    return len(delivery.plain_text(script).split()) / (seconds / 60)


def synthesize(voice, script, target_wpm=TARGET_WPM):
    """(audio, sample_rate, timeline, report)."""
    sr = voice.config.sample_rate
    lex = speech_de.lexicon()
    lines = delivery.parse(script)
    # Thorsten reads briskly: at Piper's own speed he is near 150 words a minute
    scale, passes, wall = 0.82, 0, 0.0
    # a long script is calibrated on its first lines, then read once in full
    # (three full passes of 25 minutes would not fit a runner's job)
    probe = lines if len(lines) <= 40 else lines[:30]
    probe_script = script if len(lines) <= 40 else " ".join(l.text for l in probe)
    for _ in range(3):
        t0 = time.perf_counter()
        audio, timeline = render(voice, probe, scale, lex)
        wall += time.perf_counter() - t0
        passes += 1
        got = wpm(probe_script, len(audio) / sr)
        if abs(got - target_wpm) < 2.5:
            break
        scale = min(1.25, max(0.66, scale * (target_wpm / got) ** 1.15))
    if probe is not lines:
        t0 = time.perf_counter()
        audio, timeline = render(voice, lines, scale, lex)
        wall += time.perf_counter() - t0
        passes += 1
    seconds = len(audio) / sr
    audio = audio / max(1e-6, float(np.abs(audio).max())) * 0.89
    report = {"voice": VOICE, "engine": "piper de_DE-thorsten-high", "seconds": round(seconds, 2),
              "wpm": round(wpm(script, seconds), 1), "speed_scale": round(scale, 3),
              "cpu_wall_s": round(wall, 1), "passes": passes,
              # one pass is what an episode costs once the pace is known
              "cpu_s_per_audio_min": round(wall / passes / (seconds / 60), 2)}
    log.info("%s: %.1f s at %.1f wpm (scale %.3f), %.1f s on the CPU", VOICE, seconds, report["wpm"], scale, wall)
    return audio, sr, timeline, report
