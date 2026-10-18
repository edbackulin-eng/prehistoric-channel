"""Build a documentary episode: prepare, render parts, merge.

    python automation/build_doc.py prepare <slug> [--res 2160p]
    python automation/build_doc.py part    <slug> <k> [--res 2160p]
    python automation/build_doc.py merge   <slug> [--res 2160p]
    python automation/build_doc.py samples <slug> [--res 2160p]

prepare  narration (Kokoro, deterministic), the edit plan, real clips fetched
         from their sources, the 2.5D layers fetched from the Kaggle kernel's
         output; everything lands in work/<slug>/inputs.
part k   renders work/<slug>/parts/part_k.mp4 (GitHub Actions runs one per job).
merge    joins the parts with -c copy, mixes the sound, muxes, checks.
samples  three 30-second fragments of the opening, one per voice, on the
         same picture: each line of each voice laid where the edit cuts to it.
"""
import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "kaggle"))

from core.logger import get_logger  # noqa: E402

log = get_logger("build")
MAIN_VOICE = "bm_lewis"
RECON_KERNEL = "prehistoric-recon"
SAMPLE_S = 30.0


def W(slug):
    return ROOT / "work" / slug


def narration(slug, voice=MAIN_VOICE):
    from core import narrate
    out = W(slug) / "inputs" / f"narration_{voice}.wav"
    meta = out.with_suffix(".json")
    if out.exists() and meta.exists():
        return out, json.loads(meta.read_text())
    script = (ROOT / "episodes" / slug / "script.txt").read_text(encoding="utf-8")
    audio, timeline, rep = narrate.synthesize(narrate.load_kokoro(), script, voice)
    out.parent.mkdir(parents=True, exist_ok=True)
    sf.write(out, audio, 24000)
    data = {"timeline": timeline, "report": rep}
    meta.write_text(json.dumps(data, indent=1), encoding="utf-8")
    return out, data


def fetch_media(slug, plan):
    from pipeline import footage
    media = W(slug) / "inputs" / "media"
    media.mkdir(parents=True, exist_ok=True)
    for s in plan["shots"]:
        if s["kind"] != "real":
            continue
        c = s["clip"]
        ext = Path(c["url"].split("?")[0]).suffix or ".mp4"
        cached = ROOT / "cache" / "media" / f"{c['id']}{ext}"
        dest = media / f"{c['id']}{ext}"
        if dest.exists():
            continue
        if cached.exists():
            shutil.copy(cached, dest)
        else:
            footage.download(c["url"], dest)
            time.sleep(2)
        log.info("media %s", dest.name)
    return media


def fetch_layers(slug):
    dest = W(slug) / "inputs" / "layers"
    local = ROOT / "kaggle" / "results_recon"
    if (local / "report.json").exists():
        src = local
    else:
        import push
        src = W(slug) / "inputs" / "kaggle"
        push.fetch(f"{push.username()}/{RECON_KERNEL}", src)
    dest.mkdir(parents=True, exist_ok=True)
    for f in list((src / "up").glob("*.jpg")) + list((src / "layers").glob("*")):
        tgt = dest / f.name
        if not tgt.exists():
            shutil.copy(f, tgt)
    return dest


def prepare(slug, res):
    from pipeline import docplan
    wav, data = narration(slug)
    seconds = sf.info(str(wav)).duration
    plan = docplan.build(slug, data["timeline"], seconds, parts=int(ARGS.parts))
    plan["voice"] = MAIN_VOICE
    plan["narration"] = data["report"]
    (W(slug) / "plan.json").write_text(json.dumps(plan, indent=1, ensure_ascii=False), encoding="utf-8")
    fetch_media(slug, plan)
    fetch_layers(slug)
    log.info("plan: %d shots, %d parts, %.1f s", len(plan["shots"]), len(plan["parts"]), plan["total_s"])
    return plan


def part(slug, k, res):
    from pipeline import docrender
    plan = json.loads((W(slug) / "plan.json").read_text(encoding="utf-8"))
    inputs = {"media": W(slug) / "inputs" / "media", "layers": W(slug) / "inputs" / "layers"}
    return docrender.render_part(plan, k, inputs, W(slug) / "tmp" / f"p{k}_{res}", W(slug) / f"parts_{res}",
                                 res=res)


def _mux(picture, wav, out):
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(picture), "-i", str(wav),
                    "-map", "0:v", "-map", "1:a", "-c:v", "copy", "-c:a", "aac", "-b:a", "256k",
                    "-ar", "48000", "-shortest", "-movflags", "+faststart", str(out)], check=True)


def merge(slug, res):
    from pipeline import docrender, docsound
    plan = json.loads((W(slug) / "plan.json").read_text(encoding="utf-8"))
    parts_dir = W(slug) / f"parts_{res}"
    files = [parts_dir / f"part_{k}.mp4" for k in range(len(plan["parts"]))]
    missing = [f.name for f in files if not f.exists()]
    if missing:
        raise RuntimeError(f"parts missing: {missing}")
    t0 = time.monotonic()
    picture = docrender.join(files, W(slug) / f"picture_{res}.mp4")
    amb = json.loads((ROOT / "assets" / "ambience" / "library.json").read_text(encoding="utf-8"))
    music = ROOT / "assets" / "music" / "lost-frontier.mp3"
    wav = W(slug) / "mix.wav"
    sound = docsound.build(plan, W(slug) / "inputs" / f"narration_{MAIN_VOICE}.wav", music, amb, wav)
    out = W(slug) / f"video_{res}.mp4"
    _mux(picture, wav, out)
    info = docrender.probe(out)
    reports = [json.loads((parts_dir / f"part_{k}.json").read_text()) for k in range(len(files))]
    rep = {"res": res, "video": str(out), "duration_s": round(info["duration"], 2),
           "planned_s": plan["total_s"], "size": f"{info['w']}x{info['h']}", "fps": info["fps"],
           "merge_s": round(time.monotonic() - t0, 1), "parts": reports, "sound": sound,
           "size_mb": round(out.stat().st_size / 1e6, 1)}
    (W(slug) / f"report_{res}.json").write_text(json.dumps(rep, indent=1), encoding="utf-8")
    log.info("video %s: %.1f s, %s", out.name, info["duration"], sound)
    return rep


def _voice_track(timeline_main, lines_audio, sr, total):
    """Lay each line's audio at the main voice's line start."""
    out = np.zeros(int(total * sr), np.float32)
    for l, a in zip(timeline_main, lines_audio):
        i = int(l["start"] * sr)
        if i >= len(out):
            break
        m = min(len(a), len(out) - i)
        out[i:i + m] += a[:m]
    return out


def samples(slug, res):
    """Three 30 s fragments: bm_lewis, am_fenrir, Chatterbox, on one picture."""
    from core import narrate
    from pipeline import docsound
    plan = json.loads((W(slug) / "plan.json").read_text(encoding="utf-8"))
    main_tl = json.loads((W(slug) / "inputs" / f"narration_{MAIN_VOICE}.json").read_text())["timeline"]
    picture = W(slug) / f"picture_{res}.mp4"
    clip = W(slug) / f"opening_{res}.mp4"
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(picture), "-t", str(SAMPLE_S),
                    "-c", "copy", str(clip)], check=True)
    amb = json.loads((ROOT / "assets" / "ambience" / "library.json").read_text(encoding="utf-8"))
    music = ROOT / "assets" / "music" / "lost-frontier.mp3"
    keep = [l for l in main_tl if l["start"] < SAMPLE_S]
    voices = {}
    k = narrate.load_kokoro()
    from core import delivery
    script = (ROOT / "episodes" / slug / "script.txt").read_text(encoding="utf-8")
    lines = delivery.parse(script)[:len(keep)]
    for v in ("bm_lewis", "am_fenrir"):
        lex = narrate.lexicon()
        _, _, rep = narrate.synthesize(k, script, v)        # its own calibrated pace
        scale = rep["speed_scale"]
        audio = []
        for line in lines:
            a, _ = narrate._render(k, [line.__class__(line.text, line.kind, line.speed, 0.0)], v, scale, lex)
            audio.append(a)
        voices[f"kokoro_{v}"] = (audio, 24000, rep)
    cb = sorted((ROOT / "kaggle" / "results_voice").glob("line_*.wav"))
    if cb:
        audio = [sf.read(str(f), dtype="float32")[0] for f in cb]
        sr = sf.info(str(cb[0])).samplerate
        voices["chatterbox"] = (audio, sr, {"voice": "chatterbox default"})
    out_dir = ROOT / "test"
    out_dir.mkdir(exist_ok=True)
    reps = {}
    for name, (audio, sr, rep) in voices.items():
        track = _voice_track(keep, audio, sr, SAMPLE_S + 0.5)
        vw = W(slug) / f"sample_{name}_voice.wav"
        sf.write(vw, track, sr)
        mix = W(slug) / f"sample_{name}_mix.wav"
        snd = docsound.build(plan, vw, music, amb, mix, length_s=SAMPLE_S)
        out = out_dir / f"voice_sample_{name}.mp4"
        _mux(clip, mix, out)
        words = sum(len(l["text"].split()) for l in keep)
        speech_s = sum(len(a) / sr for a in audio)
        reps[name] = {"file": str(out), "words": words, "speech_s": round(speech_s, 1),
                      "wpm_in_30s": round(words / (SAMPLE_S / 60), 1), "voice": rep, "sound": snd}
        log.info("sample %s", reps[name])
    (W(slug) / "samples_report.json").write_text(json.dumps(reps, indent=1), encoding="utf-8")
    return reps


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd")
    ap.add_argument("slug")
    ap.add_argument("k", nargs="?", type=int)
    ap.add_argument("--res", default="2160p")
    ap.add_argument("--parts", default="6")
    ARGS = ap.parse_args()
    if ARGS.cmd == "prepare":
        prepare(ARGS.slug, ARGS.res)
    elif ARGS.cmd == "part":
        part(ARGS.slug, ARGS.k, ARGS.res)
    elif ARGS.cmd == "merge":
        merge(ARGS.slug, ARGS.res)
    elif ARGS.cmd == "samples":
        samples(ARGS.slug, ARGS.res)
