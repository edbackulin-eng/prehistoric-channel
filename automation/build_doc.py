"""Build a documentary episode: prepare, render parts, merge.

    python automation/build_doc.py prepare <slug> [--res 2160p]
    python automation/build_doc.py part    <slug> <k> [--res 2160p]
    python automation/build_doc.py merge   <slug> [--res 2160p]
    python automation/build_doc.py samples <slug> [--res 2160p]

prepare  German narration (Piper, Thorsten, on the CPU), the 2.5D layers from
         the Kaggle kernel's output, the species check of every candidate
         picture (pipeline/speciescheck.py), the edit plan built from what
         passed, the real clips fetched, German subtitles; everything lands in
         work/<slug>/inputs.
part k   renders work/<slug>/parts/part_k.mp4 (GitHub Actions runs one per job).
merge    joins the parts with -c copy, mixes the sound, muxes, checks.
samples  the three German voice samples, about 40 s of the same opening text,
         each normalised to -14 LUFS: A Chatterbox Multilingual with Thorsten's
         voice as reference, B Piper Thorsten, C Chatterbox Multilingual's
         standard German voice; with what a minute of each costs.
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

from core.logger import get_logger  # noqa: E402

log = get_logger("build")
MAIN_VOICE = "piper_thorsten"
RECON_KERNEL = "prehistoric-recon"
SAMPLE_S = 40.0
SAMPLES = {"A": "voice_A_chatterbox_thorsten", "B": "voice_B_piper_thorsten", "C": "voice_C_chatterbox_standard"}


def W(slug):
    return ROOT / "work" / slug


def narration(slug, voice=MAIN_VOICE):
    from core import narrate_de
    out = W(slug) / "inputs" / f"narration_{voice}.wav"
    meta = out.with_suffix(".json")
    if out.exists() and meta.exists():
        return out, json.loads(meta.read_text())
    script = (ROOT / "episodes" / slug / "script.txt").read_text(encoding="utf-8")
    audio, sr, timeline, rep = narrate_de.synthesize(narrate_de.load(), script)
    out.parent.mkdir(parents=True, exist_ok=True)
    sf.write(out, audio, sr)
    data = {"timeline": timeline, "report": rep}
    meta.write_text(json.dumps(data, indent=1, ensure_ascii=False), encoding="utf-8")
    return out, data


def _split(text, limit):
    """Split text into pieces of at most `limit` characters, at the punctuation
    (else the space) nearest the middle, so neither half is a stray word."""
    if len(text) <= limit:
        return [text]
    mid = len(text) / 2
    cuts = [i + 1 for i, ch in enumerate(text[:-1]) if ch in ".,:;?!" and text[i + 1] == " "]
    cuts = [c for c in cuts if 0.25 * len(text) <= c <= 0.75 * len(text)] or         [i for i, ch in enumerate(text) if ch == " "]
    c = min(cuts, key=lambda i: abs(i - mid))
    return _split(text[:c].strip(), limit) + _split(text[c:].strip(), limit)


def subtitles(slug, timeline, max_chars=42):
    """German subtitles (SRT) from the narration timeline: each line in cues of
    at most two rows of max_chars, timed by their share of the letters."""
    def ts(t):
        ms = int(round(t * 1000))
        h, ms = divmod(ms, 3600000)
        m, ms = divmod(ms, 60000)
        sec, ms = divmod(ms, 1000)
        return f"{h:02d}:{m:02d}:{sec:02d},{ms:03d}"
    cues = []
    for l in timeline:
        chunks = []
        for c in _split(l["text"], 2 * max_chars):
            # a chunk whose rows would not fit in two becomes two cues
            chunks += [c] if len(_split(c, max_chars)) <= 2 else _split(c, max(max_chars, len(c) // 2 + 8))
        total = sum(len(c) for c in chunks)
        t = l["start"]
        for c in chunks:
            d = (l["end"] - l["start"]) * len(c) / total
            cues.append((t, t + d, "\n".join(_split(c, max_chars)[:2] if len(_split(c, max_chars)) <= 2
                                             else _split(c, len(c) - 1))))
            t += d
    srt = "".join(f"{i}\n{ts(a)} --> {ts(b)}\n{text}\n\n" for i, (a, b, text) in enumerate(cues, 1))
    out = W(slug) / "inputs" / "subtitles_de.srt"           # travels with the inputs to the merge
    out.write_text(srt, encoding="utf-8")
    return out


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


def modal_get(slug, what, dest):
    """Copy /out/<slug>/<what> from the Modal Volume "prehistoric-out" into dest.
    Returns the folder that holds the files, or None if Modal has nothing yet."""
    dest = Path(dest)
    tmp = dest.parent / f"_modal_{what}"
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True, exist_ok=True)
    r = subprocess.run(["modal", "volume", "get", "--force", "prehistoric-out", f"{slug}/{what}", str(tmp)],
                       capture_output=True, text=True)
    if r.returncode:
        log.warning("modal volume get %s/%s failed: %s", slug, what, r.stderr[-400:])
        return None
    reports = list(tmp.rglob("report.json"))
    root = reports[0].parent if reports else tmp
    if dest.exists():
        shutil.rmtree(dest)
    shutil.move(str(root), str(dest))
    shutil.rmtree(tmp, ignore_errors=True)
    return dest


def fetch_layers(slug):
    """The 2.5D layers and thumbnail bases made on Modal (gpu/modal_app.py)."""
    dest = W(slug) / "inputs" / "layers"
    src = ROOT / "gpu" / "results_recon"
    if not (src / "report.json").exists():
        src = modal_get(slug, "recon", W(slug) / "inputs" / "modal_recon")
        if src is None:
            log.warning("no reconstructions yet; parts that need them will fail")
            return dest
    dest.mkdir(parents=True, exist_ok=True)
    shutil.copy(src / "report.json", W(slug) / "inputs" / "kaggle_recon_report.json")
    for f in list((src / "up").glob("*.jpg")) + list((src / "layers").glob("*")):
        tgt = dest / f.name
        if not tgt.exists():
            shutil.copy(f, tgt)
    if src != ROOT / "gpu" / "results_recon":
        shutil.rmtree(src, ignore_errors=True)      # the layers are all that travel on
    return dest


def fetch_voice(slug):
    """Chatterbox Multilingual's German sample lines (A/, C/) into gpu/results_voice."""
    dest = ROOT / "gpu" / "results_voice"
    if list(dest.glob("*/line_*.wav")):
        return dest
    return modal_get(slug, "voice", dest)


def prepare(slug, res):
    from pipeline import docplan, speciescheck
    wav, data = narration(slug)
    seconds = sf.info(str(wav)).duration
    layers = fetch_layers(slug)
    picks = speciescheck.check(slug, layers, W(slug) / "inputs", data["timeline"])
    plan = docplan.build(slug, data["timeline"], seconds, parts=int(ARGS.parts), picks=picks)
    plan["voice"] = MAIN_VOICE
    plan["narration"] = data["report"]
    (W(slug) / "plan.json").write_text(json.dumps(plan, indent=1, ensure_ascii=False), encoding="utf-8")
    subtitles(slug, data["timeline"])
    fetch_media(slug, plan)
    fetch_voice(slug)
    from pipeline import maskcheck
    mc = maskcheck.check(layers, W(slug) / "inputs" / "mask_check.json")
    log.info("mask check: %s", {k: v.get("kept", v.get("verdict")) for k, v in mc["masks"].items()}
             or mc.get("skipped"))
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
    test = ROOT / "test"
    test.mkdir(exist_ok=True)
    for f in ("subtitles_de.srt", "species_check.json"):
        if (W(slug) / "inputs" / f).exists():
            shutil.copy(W(slug) / "inputs" / f, test / f)
    log.info("video %s: %.1f s, %s", out.name, info["duration"], sound)
    return rep


def loudnorm(src, dst, lufs=-14.0, tp=-1.0):
    """Two-pass EBU R128 normalisation to `lufs` integrated, true peak <= tp."""
    import re
    r = subprocess.run(["ffmpeg", "-hide_banner", "-i", str(src), "-af",
                        f"loudnorm=I={lufs}:TP={tp}:LRA=11:print_format=json", "-f", "null", "-"],
                       capture_output=True, text=True)
    m = json.loads(re.findall(r"\{[^{}]+\}", r.stderr)[-1])
    af = (f"loudnorm=I={lufs}:TP={tp}:LRA=11:measured_I={m['input_i']}:measured_TP={m['input_tp']}:"
          f"measured_LRA={m['input_lra']}:measured_thresh={m['input_thresh']}:offset={m['target_offset']}:"
          "linear=true")
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(src), "-af", af,
                    "-ar", "48000", "-ac", "1", str(dst)], check=True)
    r = subprocess.run(["ffmpeg", "-hide_banner", "-nostats", "-i", str(dst), "-af", "ebur128=peak=true",
                        "-f", "null", "-"], capture_output=True, text=True)
    i = re.findall(r"I:\s+(-?[0-9.]+) LUFS", r.stderr)
    p = re.findall(r"Peak:\s+(-?[0-9.]+) dBFS", r.stderr)
    return {"integrated_lufs": float(i[-1]) if i else None, "true_peak_dbtp": float(p[-1]) if p else None}


def samples(slug, res):
    """The three German voice samples (about 40 s, same text), each at -14 LUFS,
    with what a minute of narration costs in each engine and the 25-minute forecast."""
    import time
    sys.path.insert(0, str(ROOT / "gpu" / "kernels"))
    import voice_de_job
    from core import delivery, narrate_de
    script = (ROOT / "episodes" / slug / "script.txt").read_text(encoding="utf-8")
    lines = voice_de_job.sample_lines(delivery.parse(script), SAMPLE_S)
    out_dir = ROOT / "test"
    out_dir.mkdir(exist_ok=True)
    reps = {"text": [l.text for l in lines], "words": sum(len(l.text.split()) for l in lines)}

    # B: Piper on this runner's CPU, at the pace the whole narration was calibrated to
    meta = json.loads((W(slug) / "inputs" / f"narration_{MAIN_VOICE}.json").read_text(encoding="utf-8"))
    voice = narrate_de.load()
    t0 = time.perf_counter()
    audio, _ = narrate_de.render(voice, lines, meta["report"]["speed_scale"])
    wall = time.perf_counter() - t0
    sr = voice.config.sample_rate
    raw = W(slug) / "sample_B_raw.wav"
    sf.write(raw, audio, sr)
    per_min = wall / (len(audio) / sr / 60)
    full = meta["report"]
    reps["B"] = {"engine": "Piper 1.8 (GPL-3.0), voice de_DE-thorsten-high (CC0)", "device": "GitHub runner CPU",
                 "seconds": round(len(audio) / sr, 1), "cpu_s": round(wall, 1),
                 "cpu_s_per_audio_min": round(per_min, 1),
                 "full_narration": {k: full.get(k) for k in ("seconds", "wpm", "cpu_wall_s", "passes",
                                                               "cpu_s_per_audio_min")},
                 "forecast_25min": {"cpu_min_one_pass": round(per_min * 25 / 60, 1),
                                    "cpu_min_with_pace_calibration": round(per_min * 25 * 3 / 60, 1),
                                    "gpu_h": 0}}

    # A and C: Chatterbox Multilingual on Kaggle (gpu/kernels/voice_de_job.py)
    vdir = ROOT / "gpu" / "results_voice"
    kr = json.loads((vdir / "report.json").read_text(encoding="utf-8")) if (vdir / "report.json").exists() else {}
    for key in ("A", "C"):
        v = (kr.get("variants") or {}).get(key)
        if not v or not (vdir / key / "sample.wav").exists():
            reps[key] = {"missing": True}
            continue
        g = v["gpu_s_per_audio_min"]
        reps[key] = {"engine": "Chatterbox Multilingual 0.1.7 (MIT)", "device": kr.get("gpu", "Kaggle GPU"),
                     "label": v["label"], "reference": v["reference"], "seconds": v["sample_s"],
                     "gpu_s": round(v["gen_s"] + v["asr_s"], 1), "gen_s": v["gen_s"], "whisper_check_s": v["asr_s"],
                     "gpu_s_per_audio_min": g, "gen_s_per_audio_min": v["gen_s_per_audio_min"],
                     "redraws": v["redraws"], "min_match": v["min_match"], "model_load_s": kr.get("load_s"),
                     "forecast_25min": {"gpu_min": round(g * 25 / 60 + (kr.get("load_s") or 0) / 60, 1),
                                        "gpu_h": round((g * 25 + (kr.get("load_s") or 0)) / 3600, 2)}}
    for key, name in SAMPLES.items():
        src = W(slug) / "sample_B_raw.wav" if key == "B" else vdir / key / "sample.wav"
        if not src.exists():
            continue
        out = out_dir / f"{name}.wav"
        reps[key]["file"] = str(out)
        reps[key]["loudness"] = loudnorm(src, out)
        subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(out), "-c:a", "aac",
                        "-b:a", "192k", str(out.with_suffix(".m4a"))], check=True)
        log.info("sample %s: %s", key, reps[key]["loudness"])
    (W(slug) / "samples_report.json").write_text(json.dumps(reps, indent=1, ensure_ascii=False), encoding="utf-8")
    shutil.copy(W(slug) / "samples_report.json", out_dir / "voice_samples_report.json")
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
