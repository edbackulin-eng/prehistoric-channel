"""Narrate the opening of a script (about a minute) and time every word.

    python automation/narrate_hook.py cicada-3301 [seconds]

Writes queue/<slug>/build/hook.wav, hook_lines.json (where each marked-up line
starts and ends) and hook_words.json (Whisper word timestamps, for kinetic text).
"""
import json
import sys
from pathlib import Path

import soundfile as sf

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from core import narrate  # noqa: E402
from core.voice import SAMPLE_RATE  # noqa: E402

WPM = 170


def opening(script, seconds):
    """Lines from the top until about `seconds` of speech, ending on a full line."""
    budget = seconds * WPM / 60
    out, words = [], 0
    for line in script.splitlines():
        out.append(line)
        if not line.strip().startswith("["):
            words += len(line.lstrip("<> ").split())
        if words >= budget * 0.97:          # the pauses fill the rest of the minute
            break
    return "\n".join(out)


def words_with_times(wav_path):
    from faster_whisper import WhisperModel
    model = WhisperModel("small.en", device="cpu", compute_type="int8")
    segments, _ = model.transcribe(str(wav_path), word_timestamps=True, vad_filter=False)
    return [{"word": w.word.strip(), "start": round(w.start, 3), "end": round(w.end, 3)}
            for s in segments for w in s.words]


def main(slug, seconds=60):
    q = ROOT / "queue" / slug
    out = q / "build"
    out.mkdir(parents=True, exist_ok=True)
    hook = opening((q / "script.txt").read_text(encoding="utf-8"), seconds)
    (out / "hook_script.txt").write_text(hook + "\n", encoding="utf-8")
    audio, timeline = narrate.synthesize(narrate.load_kokoro(), hook)
    sf.write(out / "hook.wav", audio, SAMPLE_RATE)
    (out / "hook_lines.json").write_text(json.dumps(timeline, indent=1), encoding="utf-8")
    words = words_with_times(out / "hook.wav")
    (out / "hook_words.json").write_text(json.dumps(words, indent=1), encoding="utf-8")
    print(f"{len(audio) / SAMPLE_RATE:.1f}s narrated, {len(timeline)} lines, "
          f"{len(words)} words timed")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], float(sys.argv[2]) if len(sys.argv) > 2 else 60))
