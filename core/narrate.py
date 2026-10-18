"""Documentary narration with Kokoro, read from the delivery markup.

The channel speaks at about 150 words a minute over the whole programme,
pauses included - the pace of a nature documentary, not of an explainer. Each
voice reads at its own natural rate, so the speed is calibrated per voice: one
pass at the markup's nominal speeds, then every speed is scaled so the total
lands on the target, and the narration is rendered again.

Latin and place names that Kokoro's espeak front end gets wrong are not
respelled (respellings change the stress unpredictably); their IPA from
voice/pronunciations.json is spliced straight into the phoneme string.

Returns the audio and, for every line, where it starts and ends, so the edit
can cut to the narration without a forced alignment pass.
"""
import json
import re
from pathlib import Path

import numpy as np

from core import delivery
from core.logger import get_logger
from core.voice import SAMPLE_RATE, _trim_silence, peak_normalize

log = get_logger("narrate")

ROOT = Path(__file__).resolve().parent.parent
VOICES = {"bm_lewis": "en-gb", "am_fenrir": "en-us"}
VOICE = "bm_lewis"
TARGET_WPM = 150.0
SPEED_MIN, SPEED_MAX = 0.78, 1.12
SLOW_LEAD = (0.6, 0.9)
SENTENCE_TAIL_S = 0.30
QUESTION_EXTRA_S = 0.12
_SENT = re.compile(r"(?<=[.!?])\s+")


def lexicon():
    data = json.loads((ROOT / "voice" / "pronunciations.json").read_text(encoding="utf-8"))
    return {k: v for k, v in data.items() if not k.startswith("_")}


def _word_re(lex):
    words = sorted(lex, key=len, reverse=True)
    return re.compile(r"(?<![\w-])(" + "|".join(map(re.escape, words)) + r")(?![\w-])")


def phonemes(kokoro, text, lang, lex=None):
    """The sentence as Kokoro phonemes, with lexicon words in their own IPA."""
    lex = lex if lex is not None else lexicon()
    out = []
    for i, part in enumerate(_word_re(lex).split(text)):
        if i % 2:
            out.append(lex[part]["ipa"][lang])
        elif part.strip():
            out.append(kokoro.tokenizer.phonemize(part, lang).strip())
    ph = " ".join(p for p in out if p)
    return re.sub(r"\s+([.,;:!?])", r"\1", ph)


def respell(text, lex=None):
    """Text-only engines (Chatterbox) get the phonetic respelling instead."""
    lex = lex if lex is not None else lexicon()
    return _word_re(lex).sub(lambda m: lex[m.group(1)]["respell"], text)


def _lead_for(line):
    n = len(line.text.split())
    lo, hi = SLOW_LEAD
    return round(lo + (hi - lo) * min(1.0, n / 10), 2)


def _render(kokoro, lines, voice, scale, lex):
    lang = VOICES[voice]
    chunks, timeline, t = [], [], 0.0
    for line in lines:
        pause = line.pause_before
        if line.kind == "slow":
            pause = max(pause, _lead_for(line))
        if pause:
            chunks.append(np.zeros(int(SAMPLE_RATE * pause), dtype=np.float32))
            t += pause
        speed = min(SPEED_MAX, max(SPEED_MIN, line.speed * scale))
        start = t
        for sentence in [s for s in _SENT.split(line.text) if s.strip()]:
            ph = phonemes(kokoro, sentence, lang, lex)
            audio, _ = kokoro.create(ph, voice=voice, speed=speed, lang=lang, is_phonemes=True)
            audio = _trim_silence(np.asarray(audio, dtype=np.float32))
            chunks.append(audio)
            t += len(audio) / SAMPLE_RATE
            tail = SENTENCE_TAIL_S + (QUESTION_EXTRA_S if sentence.rstrip().endswith("?") else 0)
            chunks.append(np.zeros(int(SAMPLE_RATE * tail), dtype=np.float32))
            t += tail
        timeline.append({"text": line.text, "kind": line.kind, "start": round(start, 3),
                         "end": round(t, 3), "speed": round(speed, 3)})
    return np.concatenate(chunks), timeline


def wpm(script, seconds):
    return len(delivery.plain_text(script).split()) / (seconds / 60)


def synthesize(kokoro, script, voice=VOICE, target_wpm=TARGET_WPM):
    """(audio, timeline, report). Two passes: measure the voice, then hit the pace."""
    lex = lexicon()
    lines = delivery.parse(script)
    scale = 0.92
    for _ in range(3):
        audio, timeline = _render(kokoro, lines, voice, scale, lex)
        got = wpm(script, len(audio) / SAMPLE_RATE)
        if abs(got - target_wpm) < 2.5:
            break
        # pauses do not scale with speed, so correct a little past the ratio
        scale = min(1.25, max(0.8, scale * (target_wpm / got) ** 1.15))
    seconds = len(audio) / SAMPLE_RATE
    report = {"voice": voice, "seconds": round(seconds, 2),
              "wpm": round(wpm(script, seconds), 1), "speed_scale": round(scale, 3)}
    log.info("%s: %.1f s at %.1f wpm (scale %.3f)", voice, seconds, report["wpm"], scale)
    return peak_normalize(audio), timeline, report


def load_kokoro():
    from kokoro_onnx import Kokoro
    models = ROOT / "assets" / "models"
    return Kokoro(str(models / "kokoro-v1.0.onnx"), str(models / "voices-v1.0.bin"))
