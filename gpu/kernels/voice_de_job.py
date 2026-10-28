"""German voice samples: Chatterbox Multilingual (MIT) on a Kaggle GPU.

Two of the three samples the channel compares (the third, Piper, runs on a
GitHub runner's CPU - core/narrate_de.py):
  A  Chatterbox Multilingual conditioned on Thorsten Mueller's voice: 11.4 s
     from the Thorsten-Voice dataset (CC0), assets/voice/thorsten_ref.wav;
  C  Chatterbox Multilingual with the German voice Resemble AI's own demo uses
     (mtl_prompts/de_f1.flac); its licence is not stated, so it is for this
     comparison only. If it cannot be fetched, the model's built-in voice.

Every line is read with language_id="de", heard back by Whisper (small, in
German) and redrawn when words go missing, up to four times. Generation and
checking are timed per sample, so the report can say what a minute of German
narration costs on this GPU.
"""
from pathlib import Path

SLUG = "prehistoric-voice"
REF_URL = ("https://raw.githubusercontent.com/edbackulin-eng/prehistoric-channel/main/"
           "assets/voice/thorsten_ref.wav")
DEMO_DE_URL = "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/de_f1.flac"

SOURCE = r'''
import subprocess, sys, json, re, difflib, pathlib, time, urllib.request
# chatterbox-tts pins torch 2.6; torch, torchvision and torchaudio go in as one
# matched set, or torchvision breaks transformers (as in chatterbox_job.py).
subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                "torch==2.6.0", "torchvision==0.21.0", "torchaudio==2.6.0",
                "chatterbox-tts==0.1.7", "openai-whisper", "pyloudnorm", "num2words"], check=True)
_mark("install done")
import numpy as np, torch, torchaudio, whisper, pyloudnorm as pyln
from num2words import num2words
from chatterbox.mtl_tts import ChatterboxMultilingualTTS
_need_gpu()

OUT = pathlib.Path("/kaggle/working"); OUT.mkdir(exist_ok=True)
report = {"errors": [], "variants": {}, "gpu": torch.cuda.get_device_name(0)}
t0 = time.time()
tts = ChatterboxMultilingualTTS.from_pretrained(device="cuda")
SR = tts.sr
asr = whisper.load_model("small", device="cuda")
report["load_s"] = round(time.time() - t0, 1)
import copy
BUILT_IN = copy.deepcopy(tts.conds)        # the model's own voice, for a sample without a reference
meter = pyln.Meter(SR)
TRIES = 4

def words(s):
    s = re.sub(r"\d+", lambda m: num2words(int(m.group(0)), lang="de"), s.lower())
    return re.findall(r"[a-zäöüß]+", s)

def score(want, got):
    return difflib.SequenceMatcher(None, words(want), words(got)).ratio()

def fetch(url, path):
    try:
        urllib.request.urlretrieve(url, path)
        return str(path)
    except Exception as e:
        report["errors"].append(f"fetch {url}: {type(e).__name__}: {e}")
        return None

for v in JOB["variants"]:
    key = v["key"]
    vdir = OUT / key; vdir.mkdir(exist_ok=True)
    _mark(f"voice {key}")
    ref = fetch(v["ref_url"], vdir / ("ref" + pathlib.Path(v["ref_url"]).suffix)) if v.get("ref_url") else None
    if v.get("ref_url") and ref is None and v.get("required"):
        report["errors"].append(f"{key}: the reference voice could not be fetched; sample skipped")
        continue
    rep = {"label": v["label"], "reference": v.get("ref_url") if ref else "built-in voice",
           "lines": [], "gen_s": 0.0, "asr_s": 0.0}
    if ref:
        tts.prepare_conditionals(ref, exaggeration=0.5)
    else:
        tts.conds = copy.deepcopy(BUILT_IN)
    parts = []
    for i, line in enumerate(JOB["lines"]):
        text = line["speak"]
        # documentary delivery: a lower cfg weight reads slower and steadier
        cfg = {"fast": 0.50, "base": 0.42, "slow": 0.32}[line["kind"]]
        exag = {"fast": 0.55, "base": 0.50, "slow": 0.45}[line["kind"]]
        best = None
        for attempt in range(TRIES):
            torch.manual_seed(2000 + i * 10 + attempt)
            t1 = time.time()
            wav = tts.generate(text, language_id="de", exaggeration=exag, cfg_weight=cfg).squeeze(0).cpu().numpy()
            torch.cuda.synchronize(); rep["gen_s"] += time.time() - t1
            t1 = time.time()
            heard = asr.transcribe(torchaudio.functional.resample(torch.tensor(wav), SR, 16000).numpy(),
                                   language="de")["text"]
            rep["asr_s"] += time.time() - t1
            s = score(text, heard)
            if best is None or s > best[0]:
                best = (s, wav, heard, attempt + 1)
            if s >= 0.95:
                break
        s, wav, heard, tries = best
        env = np.abs(wav) > 0.01
        if env.any():
            a, b = np.argmax(env), len(env) - np.argmax(env[::-1])
            wav = wav[max(0, a - int(0.02 * SR)): b + int(0.05 * SR)]
        try:
            wav = pyln.normalize.loudness(wav, meter.integrated_loudness(wav), -20.0).astype(np.float32)
        except Exception:
            pass
        torchaudio.save(str(vdir / f"line_{i:02d}.wav"), torch.tensor(wav).unsqueeze(0), SR)
        rep["lines"].append({"line": i, "want": text, "heard": heard.strip(), "match": round(s, 3),
                             "tries": tries, "seconds": round(len(wav) / SR, 2)})
        if line["pause_before"]:
            parts.append(np.zeros(int(SR * line["pause_before"]), np.float32))
        parts.append(wav)
        parts.append(np.zeros(int(SR * 0.12), np.float32))
    sample = np.concatenate(parts)
    torchaudio.save(str(vdir / "sample.wav"), torch.tensor(sample).unsqueeze(0), SR)
    speech = sum(l["seconds"] for l in rep["lines"])
    rep.update({"gen_s": round(rep["gen_s"], 1), "asr_s": round(rep["asr_s"], 1),
                "speech_s": round(speech, 1), "sample_s": round(len(sample) / SR, 1),
                "gpu_s_per_audio_min": round((rep["gen_s"] + rep["asr_s"]) / (speech / 60), 1),
                "gen_s_per_audio_min": round(rep["gen_s"] / (speech / 60), 1),
                "redraws": sum(l["tries"] - 1 for l in rep["lines"]),
                "min_match": min(l["match"] for l in rep["lines"])})
    report["variants"][key] = rep
    print(key, {k: v for k, v in rep.items() if k != "lines"}, flush=True)

report["total_s"] = round(time.time() - t0, 1)
report["lines"] = len(JOB["lines"])
(OUT / "report.json").write_text(json.dumps(report, indent=1, ensure_ascii=False))
_WD.cancel()
print(json.dumps({k: v for k, v in report.items() if k != "variants"}), flush=True)
'''


def job(slug):
    """The lines of the opening (about 40 s) as each engine reads them."""
    import sys
    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root))
    from core import delivery, speech_de
    script = (root / "episodes" / slug / "script.txt").read_text(encoding="utf-8")
    lines = sample_lines(delivery.parse(script))
    lex = speech_de.lexicon()
    return {"lines": [{"text": l.text, "speak": speech_de.for_text_engine(l.text, lex), "kind": l.kind,
                       "pause_before": l.pause_before} for l in lines],
            "variants": [{"key": "A", "label": "Chatterbox Multilingual, Thorsten reference (CC0)",
                          "ref_url": REF_URL, "required": True},
                         {"key": "C", "label": "Chatterbox Multilingual, standard German demo voice",
                          "ref_url": DEMO_DE_URL}],
            "budget_s": 1500}


def sample_lines(lines, seconds=40.0, wpm=150.0):
    """The opening lines that fill about `seconds`: words at the speaking rate
    within a line (pauses excluded), plus every pause before a line."""
    out, t = [], 0.0
    for l in lines:
        t += l.pause_before + len(l.text.split()) / (wpm / 60) + 0.3
        out.append(l)
        if t >= seconds - 2:
            break
    return out
