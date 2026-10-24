"""Voice sample A: Chatterbox (MIT) on a Kaggle GPU.

    python kaggle/chatterbox_job.py voice/sample_cicada.txt

Chatterbox's built-in default voice only - no reference audio is passed, so
nobody's voice is cloned. The fragment is spoken line by line from the same
delivery markup Kokoro reads. Every chunk is transcribed back with Whisper and
redrawn when words are missing or added; the chunks are then matched in loudness
and in broad spectral balance so the joins do not show.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "kaggle"))
from core import delivery  # noqa: E402

SLUG = "closedtabs-voice-sample"
BUILD = ROOT / "kaggle" / "build_voice"
RESULTS = ROOT / "kaggle" / "results_voice"

SOURCE = r'''
import subprocess, sys, json, re, difflib, pathlib
# chatterbox-tts pins torch 2.6, while Kaggle ships a newer torch with a
# torchvision built for it. Installing chatterbox alone downgrades torch and
# leaves torchvision mismatched ("operator torchvision::nms does not exist"),
# which then breaks transformers. So the three are installed as one matched set.
subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                "torch==2.6.0", "torchvision==0.21.0", "torchaudio==2.6.0",
                "chatterbox-tts", "openai-whisper", "pyloudnorm"], check=True)
import numpy as np, torch, torchaudio, whisper, pyloudnorm as pyln
from scipy.signal import fftconvolve
from chatterbox.tts import ChatterboxTTS

OUT = pathlib.Path("/kaggle/working"); OUT.mkdir(exist_ok=True)
tts = ChatterboxTTS.from_pretrained(device="cuda")
SR = tts.sr
asr = whisper.load_model("base.en", device="cuda")
meter = pyln.Meter(SR)
TARGET_LUFS = -20.0
TRIES = 4

NUMBER_WORDS = set("zero one two three four five six seven eight nine ten eleven twelve "
                   "thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty "
                   "thirty forty fifty sixty seventy eighty ninety hundred thousand million "
                   "first second third fourth fifth sixth seventh eighth ninth tenth".split())

def words(s):
    # Whisper writes numbers as digits ("January 4, 2012") however they were
    # spoken, so numbers are left out of the comparison on both sides; a line
    # that differed only there was being redrawn four times for nothing.
    toks = re.findall(r"[a-z0-9']+", s.lower())
    return [t for t in toks if not re.fullmatch(r"[0-9]+(st|nd|rd|th)?", t)
            and t not in NUMBER_WORDS]

def score(want, got):
    return difflib.SequenceMatcher(None, words(want), words(got)).ratio()

def numbers_to_words(s):
    return (s.replace("January 4th, 2012", "January fourth, twenty twelve")
             .replace("4th", "fourth"))

report, chunks = [], []
for i, line in enumerate(JOB["lines"]):
    text = numbers_to_words(line["text"])
    # Chatterbox has no speed control. Lower cfg_weight gives a slower, more
    # deliberate read; the exaggeration stays in the 0.6-0.7 band asked for.
    cfg = {"fast": 0.55, "base": 0.45, "slow": 0.30}[line["kind"]]
    exag = {"fast": 0.70, "base": 0.65, "slow": 0.60}[line["kind"]]
    best = None
    for attempt in range(TRIES):
        torch.manual_seed(1000 + i * 10 + attempt)
        wav = tts.generate(text, exaggeration=exag, cfg_weight=cfg).squeeze(0).cpu().numpy()
        heard = asr.transcribe(wav.astype(np.float32) if SR == 16000 else
                               torchaudio.functional.resample(torch.tensor(wav), SR, 16000).numpy(),
                               language="en")["text"]
        s = score(text, heard)
        if best is None or s > best[0]:
            best = (s, wav, heard, attempt + 1)
        if s >= 0.97:
            break
    s, wav, heard, tries = best
    report.append({"line": i, "want": text, "heard": heard.strip(), "match": round(s, 3), "tries": tries})
    # trim leading/trailing silence
    env = np.abs(wav) > 0.01
    if env.any():
        a, b = np.argmax(env), len(env) - np.argmax(env[::-1])
        wav = wav[max(0, a - int(0.02 * SR)): b + int(0.04 * SR)]
    chunks.append((line, wav))

# Loudness: every chunk to the same integrated loudness.
def to_lufs(w):
    try:
        return pyln.normalize.loudness(w, meter.integrated_loudness(w), TARGET_LUFS)
    except Exception:
        return w
chunks = [(l, to_lufs(w)) for l, w in chunks]

# Timbre: a gentle matching EQ toward the average spectrum of all chunks, at most
# +-3 dB, so one chunk drawn slightly brighter or duller does not stand out.
def spectrum(w, n=2048):
    frames = [w[k:k + n] for k in range(0, max(1, len(w) - n), n // 2)]
    return np.mean([np.abs(np.fft.rfft(f * np.hanning(len(f)), n)) for f in frames if len(f) == n], axis=0) + 1e-9
ref = np.mean([spectrum(w) for _, w in chunks], axis=0)
smooth = lambda x: np.convolve(x, np.ones(31) / 31, mode="same")
matched = []
for l, w in chunks:
    gain = np.clip(smooth(ref / spectrum(w)), 10 ** (-3 / 20), 10 ** (3 / 20))
    kernel = np.fft.fftshift(np.fft.irfft(gain, 2048)) * np.hanning(2048)
    matched.append((l, fftconvolve(w, kernel, mode="same").astype(np.float32)))

parts = []
for l, w in matched:
    if l["pause_before"]:
        parts.append(np.zeros(int(SR * l["pause_before"]), dtype=np.float32))
    parts.append(w)
    parts.append(np.zeros(int(SR * 0.18), dtype=np.float32))
audio = np.concatenate(parts)
audio = audio / max(1e-6, np.abs(audio).max()) * 10 ** (-1 / 20)
torchaudio.save(str(OUT / "A_chatterbox.wav"), torch.tensor(audio).unsqueeze(0), SR)
(OUT / "report.json").write_text(json.dumps({"seconds": len(audio) / SR, "lines": report}, indent=1))
print(json.dumps(report, indent=1))
'''


def main():
    import push
    script = Path(sys.argv[1]).read_text(encoding="utf-8")
    job = {"lines": [l.__dict__ for l in delivery.parse(script)]}
    BUILD.mkdir(parents=True, exist_ok=True)
    cell = f"JOB = {json.dumps(job, indent=1)}\n{SOURCE}"
    nb = {"cells": [{"cell_type": "code", "metadata": {}, "execution_count": None,
                     "outputs": [], "source": cell.splitlines(keepends=True)}],
          "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                      "name": "python3"}},
          "nbformat": 4, "nbformat_minor": 5}
    (BUILD / "notebook.ipynb").write_text(json.dumps(nb, indent=1), encoding="utf-8")
    os.environ["FLUX_MIRROR"] = "0"
    meta = push._metadata(push.username(), SLUG, SLUG)
    (BUILD / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    push.wait_for_free_gpu()
    out = subprocess.run([sys.executable, "-m", "kaggle", "kernels", "push", "-p", str(BUILD)],
                         capture_output=True, text=True)
    print(out.stdout, out.stderr)
    ref = f"{push.username()}/{SLUG}"
    ok = push.wait(ref, poll_s=30, timeout_s=2 * 3600)
    push.fetch(ref, RESULTS)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
