"""Render a marked-up fragment with Kokoro, speed varying line by line.

    python scripts/voice_sample_kokoro.py voice/sample_cicada.txt am_michael am_onyx
"""
import re
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from core import delivery, text  # noqa: E402
from core.voice import PAUSE_S, SAMPLE_RATE, _trim_silence, peak_normalize  # noqa: E402

MODELS = Path(r"C:\whatif-channel\assets\models")
TARGET_WPM = 170
# Speaking rate measured for each voice at the nominal delivery speeds on this
# fragment. Each voice is scaled toward the target, then held inside the
# 0.95-1.12 band the delivery is allowed to use.
MEASURED_WPM = {"am_michael": 166, "am_onyx": 146}
SPEED_MIN, SPEED_MAX = 0.95, 1.12
SENTENCE_TAIL_S = 0.20          # breath between sentences inside one line
_SENT = re.compile(r"(?<=[.!?])\s+")


def render(kokoro, script, voice):
    scale = TARGET_WPM / MEASURED_WPM.get(voice, TARGET_WPM)
    out, spoken_s = [], 0.0
    for line in delivery.parse(script):
        if line.pause_before:
            out.append(np.zeros(int(SAMPLE_RATE * line.pause_before), dtype=np.float32))
        for sentence in [s for s in _SENT.split(line.text) if s.strip()]:
            speed = min(SPEED_MAX, max(SPEED_MIN, line.speed * scale))
            audio, sr = kokoro.create(sentence, voice=voice, speed=speed, lang="en-us")
            audio = _trim_silence(np.asarray(audio, dtype=np.float32))
            spoken_s += len(audio) / sr
            out.append(audio)
            tail = SENTENCE_TAIL_S + (0.12 if sentence.rstrip().endswith("?") else 0.0)
            out.append(np.zeros(int(SAMPLE_RATE * tail), dtype=np.float32))
    return peak_normalize(np.concatenate(out)), spoken_s


def main():
    from kokoro_onnx import Kokoro
    script = Path(sys.argv[1]).read_text(encoding="utf-8")
    words = text.word_count(delivery.plain_text(script))
    kokoro = Kokoro(str(MODELS / "kokoro-v1.0.onnx"), str(MODELS / "voices-v1.0.bin"))
    for voice in sys.argv[2:]:
        audio, spoken = render(kokoro, script, voice)
        path = ROOT / "samples" / f"B_kokoro_{voice}.wav"
        sf.write(path, audio, SAMPLE_RATE)
        total = len(audio) / SAMPLE_RATE
        print(f"{voice}: {total:.1f}s total, {words} words, "
              f"{words / (spoken / 60):.0f} wpm while speaking, "
              f"{words / (total / 60):.0f} wpm overall -> {path.name}")


if __name__ == "__main__":
    main()
