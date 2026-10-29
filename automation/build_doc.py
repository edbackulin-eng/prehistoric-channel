"""Build a documentary episode: prepare, render parts, merge.

    python automation/build_doc.py prepare <slug> [--res 2160p]
    python automation/build_doc.py part    <slug> <k> [--res 2160p]
    python automation/build_doc.py merge   <slug> [--res 2160p]

prepare  German narration (Piper, Thorsten, on the CPU), the 2.5D layers and the
         upscaled LTX-Video clips from the runners' CPU matrix, the species
         check of every candidate picture (pipeline/speciescheck.py), the edit
         plan built from what passed (hook, moving shots, at most three museum
         pictures), the real clips fetched, German subtitles; everything lands
         in work/<slug>/inputs.
part k   renders work/<slug>/parts/part_k.mp4 (GitHub Actions runs one per job).
merge    joins the parts with -c copy, mixes the sound (music by section),
         muxes, checks; the finished episode goes to work/<slug>/release/.
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


def fetch_media(slug, plan, shots=None, local_only=False):
    """The real clips: in prepare only the LTX clips (local files), in each part
    job the clips of its own shots - a 25-minute episode has ~150 clips, too
    much to pass through one artifact."""
    from pipeline import footage
    media = W(slug) / "inputs" / "media"
    media.mkdir(parents=True, exist_ok=True)
    for s in (plan["shots"] if shots is None else shots):
        if s["kind"] != "real":
            continue
        if local_only and not s["clip"]["url"].startswith("file:"):
            continue
        c = s["clip"]
        ext = Path(c["url"].split("?")[0]).suffix or ".mp4"
        cached = ROOT / "cache" / "media" / f"{c['id']}{ext}"
        dest = media / f"{c['id']}{ext}"
        if dest.exists():
            continue
        if c["url"].startswith("file:"):           # an LTX-Video clip, upscaled on the runners
            shutil.copy(c["url"][5:], dest)
        elif cached.exists():
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
    """The 2.5D layers: made on the runners' CPUs (pipeline/layers_cpu.py, the
    render's "layers" matrix) from the FLUX frames Kaggle made; failing that, an
    older all-GPU kernel output or Modal's."""
    dest = W(slug) / "inputs" / "layers"
    cpu = W(slug) / "layers_cpu"
    if (cpu / "up").exists() and any((cpu / "up").glob("*.jpg")):
        from pipeline import layers_cpu
        dest.mkdir(parents=True, exist_ok=True)
        for f in list((cpu / "up").glob("*.jpg")) + list((cpu / "layers").glob("*")):
            shutil.copy(f, dest / f.name)
        summary = layers_cpu.merge_reports(slug)
        (W(slug) / "inputs" / "cpu_layers_report.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
        rp = ROOT / "gpu" / "results_recon" / "report.json"
        if rp.exists():
            shutil.copy(rp, W(slug) / "inputs" / "kaggle_recon_report.json")
        log.info("layers from the CPU matrix: %d frames, %s s a FLUX frame", summary["frames"],
                 summary["per_flux_frame_s"].get("total_s"))
        return dest
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


def animated(slug):
    """The LTX-Video clips that came through: FLUX frame id -> clip."""
    from pipeline import docrender
    out = {}
    for p in sorted((W(slug) / "layers_cpu" / "clips").glob("*.mp4")):
        try:
            i = docrender.probe(p)
            out[p.stem] = {"path": str(p), "w": i["w"], "h": i["h"], "duration": round(i["duration"], 2)}
        except Exception as e:                                       # noqa: BLE001
            log.warning("clip %s unreadable: %s", p.name, e)
    return out


def prepare(slug, res):
    from pipeline import docplan, speciescheck
    wav, data = narration(slug)
    seconds = sf.info(str(wav)).duration
    layers = fetch_layers(slug)
    picks = speciescheck.check(slug, layers, W(slug) / "inputs", data["timeline"])
    plan = docplan.build(slug, data["timeline"], seconds, parts=int(ARGS.parts), picks=picks,
                         animated=animated(slug))
    plan["voice"] = MAIN_VOICE
    plan["narration"] = data["report"]
    (W(slug) / "plan.json").write_text(json.dumps(plan, indent=1, ensure_ascii=False), encoding="utf-8")
    subtitles(slug, data["timeline"])
    fetch_media(slug, plan, local_only=True)
    tb = W(slug) / "inputs" / "thumb_bases"                 # FLUX bases for the thumbnails, made at merge
    tb.mkdir(parents=True, exist_ok=True)
    for f in (ROOT / "gpu" / "results_recon" / "raw").glob("thumb_*.jpg"):
        shutil.copy(f, tb / f.name)
    from pipeline import maskcheck
    mc = maskcheck.check(layers, W(slug) / "inputs" / "mask_check.json")
    log.info("mask check: %s", {k: v.get("kept", v.get("verdict")) for k, v in mc["masks"].items()}
             or mc.get("skipped"))
    log.info("plan: %d shots, %d parts, %.1f s, %s", len(plan["shots"]), len(plan["parts"]), plan["total_s"],
             plan["checks"])
    return plan


def part(slug, k, res):
    from pipeline import docrender
    plan = json.loads((W(slug) / "plan.json").read_text(encoding="utf-8"))
    a, b = plan["parts"][k]
    fetch_media(slug, plan, plan["shots"][a:b])
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
    music = docsound.score(plan, slug)                       # a track per section, by mood
    wav = W(slug) / "mix.wav"
    sound = docsound.build(plan, W(slug) / "inputs" / f"narration_{MAIN_VOICE}.wav", music, amb, wav)
    plan["music"] = [m["title"] for m in music["tracks"]]
    (W(slug) / "plan.json").write_text(json.dumps(plan, indent=1, ensure_ascii=False), encoding="utf-8")
    out = W(slug) / f"video_{res}.mp4"
    _mux(picture, wav, out)
    info = docrender.probe(out)
    reports = [json.loads((parts_dir / f"part_{k}.json").read_text()) for k in range(len(files))]
    rep = {"res": res, "video": str(out), "duration_s": round(info["duration"], 2),
           "planned_s": plan["total_s"], "size": f"{info['w']}x{info['h']}", "fps": info["fps"],
           "merge_s": round(time.monotonic() - t0, 1), "parts": reports, "sound": sound,
           "size_mb": round(out.stat().st_size / 1e6, 1)}
    (W(slug) / f"report_{res}.json").write_text(json.dumps(rep, indent=1), encoding="utf-8")
    rel = W(slug) / "release"                               # what the upload (or the buffer) takes
    rel.mkdir(exist_ok=True)
    for f in ("subtitles_de.srt", "species_check.json", "species_grid.jpg", "cpu_layers_report.json"):
        if (W(slug) / "inputs" / f).exists():
            shutil.copy(W(slug) / "inputs" / f, rel / f)
    shutil.move(str(out), rel / "video.mp4")
    rep["video"] = str(rel / "video.mp4")
    (W(slug) / f"report_{res}.json").write_text(json.dumps(rep, indent=1), encoding="utf-8")
    log.info("video %s: %.1f s, %s", out.name, info["duration"], sound)
    return rep


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
