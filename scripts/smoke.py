"""Smoke test: the whole documentary chain on a few seconds, in a few minutes.

    python scripts/smoke.py

Run by .github/workflows/smoke.yml on every push to dev; only a passing run
moves the code to main.
  1. every module imports;
  2. two German lines are narrated (Piper Thorsten, numbers spoken as words,
     pronunciation lexicon applied); the species check's pass rule, the
     plan's handling of a dropped scene and the subtitles are exercised;
  3. a two-part plan of drawn shots and a 2.5D still is rendered at 640x360,
     each part on its own, and the parts are joined with -c copy;
  4. the sound is mixed and measured: -14 LUFS (+-0.5), true peak <= -1 dBTP;
  5. the joined picture is as long as planned and has no black stretch.
Exits non-zero, naming the failed check.
"""
import importlib
import json
import pkgutil
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "gpu" / "kernels"))
sys.path.insert(0, str(ROOT / "gpu"))
W = ROOT / "work" / "smoke"
failures = []


def check(name, ok, detail=""):
    print(f"[{'ok' if ok else 'FAIL'}] {name} {detail}", flush=True)
    if not ok:
        failures.append(f"{name}: {detail}")


def main():
    shutil.rmtree(W, ignore_errors=True)
    (W / "layers").mkdir(parents=True)
    for pkg in ("core", "pipeline", "automation"):
        for m in pkgutil.iter_modules([str(ROOT / pkg)]):
            try:
                importlib.import_module(f"{pkg}.{m.name}")
            except Exception as e:                               # noqa: BLE001
                check(f"import {pkg}.{m.name}", False, repr(e)[:200])
    check("imports", not failures)

    # the GPU code Modal runs is assembled from the kernels; it must build clean
    sys.path.insert(0, str(ROOT / "gpu"))
    import modal_app
    src = modal_app.recon_source("smoke")
    vsrc = modal_app.voice_source("smoke")
    check("modal sources", "/kaggle/" not in src + vsrc and "flux4bit" in src, f"{len(src)}+{len(vsrc)} chars")

    # a kernel is judged by its output, not by Kaggle's COMPLETE (25.10)
    import kaggle_once
    for label, rep, want, expect in (
            ("0 of 14 images", {"images": [], "errors": ["flux: 401"]}, 14, False),
            ("13 of 14, no errors", {"images": [{}] * 13, "errors": []}, 14, True),
            ("14 of 14 but an error", {"images": [{}] * 14, "errors": ["layers: x"]}, 14, False)):
        d = W / "once" / label.replace(" ", "_")
        d.mkdir(parents=True, exist_ok=True)
        (d / "report.json").write_text(json.dumps(rep), encoding="utf-8")
        ok, why = kaggle_once.validate("recon", d, want)
        check(f"kernel output check: {label}", ok == expect, why)

    # the 2.5D mask rules: one clear subject passes, scraps and empty fail
    import recon_job
    src = recon_job.SOURCE
    g = {"np": np, "cv2": cv2, "Image": None}
    exec(src[src.index("_SEG = {}"): src.index("# ------------------------------------------- 3. depth")], g)
    blob = np.zeros((792, 1408), np.float32)
    cv2.ellipse(blob, (700, 400), (300, 160), 0, 0, 360, 1.0, -1)
    scraps = np.zeros_like(blob)
    for i in range(12):
        cv2.circle(scraps, (100 + i * 110, 200 + (i % 3) * 180), 30, 1.0, -1)
    check("mask: one subject passes", g["mask_checks"](blob)["ok"])
    check("mask: scattered scraps fail", not g["mask_checks"](scraps)["ok"])
    check("mask: empty fails", not g["mask_checks"](np.zeros_like(blob))["ok"])
    # Kaggle runs FLUX and nothing else; the CPU layers apply the same mask rules
    fsrc = recon_job.flux_source()
    compile("JOB = {}\n" + fsrc, "flux_only", "exec")
    check("GPU job is FLUX only", "esrgan" not in fsrc.lower() and "big-lama" not in fsrc
          and "new_session" not in fsrc and "depth-estimation" not in fsrc and "RAW = OUT" in fsrc)
    from pipeline import layers_cpu
    check("CPU mask rules match", layers_cpu.mask_checks(blob)["ok"] and not layers_cpu.mask_checks(scraps)["ok"])

    from core import narrate_de, speech_de
    spoken = speech_de.for_text_engine("East London, Südafrika. Der 22. Dezember 1938.")
    check("numbers in words", "zweiundzwanzigste Dezember neunzehnhundertachtunddreißig" in spoken
          and "Iest London" in spoken, spoken)
    check("lexicon to Piper", "[[ kalˈʊmnɛː ]]" in speech_de.for_piper("Latimeria chalumnae"),
          speech_de.for_piper("Latimeria chalumnae"))
    voice = narrate_de.load()
    text = "> Smith gab dem Fisch den Namen Latimeria chalumnae.\n< Es ist ein Quastenflosser."
    audio, sr, timeline, rep = narrate_de.synthesize(voice, text)
    wav = W / "narration.wav"
    sf.write(wav, audio, sr)
    check("narration (Piper)", rep["seconds"] > 2 and len(timeline) == 2, json.dumps(rep))

    # the species check's rule, on answers as Gemini gives them
    from pipeline import speciescheck as sc
    good = {"species_match": True, "features": {"a": "yes", "b": "yes", "c": "not visible"},
            "forbidden_seen": [], "fits_narration": True, "off_topic_objects": [], "people": False,
            "text_or_logo": False}
    check("species: right animal passes", sc.passes(good, "latimeria")[0] is True)
    check("species: a wrong feature fails", sc.passes({**good, "features": {"a": "yes", "b": "no"}}, "latimeria")[0]
          is False)
    check("species: a bird's head fails", sc.passes({**good, "forbidden_seen": ["a bird's head"]}, "quetzalcoatlus")[0]
          is False)
    check("relevance: candles fail", sc.passes({**good, "off_topic_objects": ["candles"]})[0] is False)
    check("archive: people are fine", sc.passes({**good, "people": True, "off_topic_objects": ["people"]},
                                                "latimeria", archive=True)[0] is True)

    # a beat nothing passed for is dropped and its neighbour runs on; nothing is shown twice
    from pipeline import docplan
    tl = [{"text": t, "kind": "base", "start": a, "end": b} for t, a, b in
          (("Vor sechsundsechzig Millionen Jahren.", 0.0, 6.0), ("Er schlug an der Küste ein.", 6.0, 11.0),
           ("Etwa drei Viertel.", 11.0, 16.0), ("Die großen Dinosaurier.", 16.0, 20.0))]
    picks = {"asteroid_sky": ["asteroid_sky__s1"], "impact_forest": [], "edmontosaurus_shore": ["edmontosaurus_shore__s2"]}
    real = docplan.ROOT
    shots_spec = json.loads((real / "episodes" / "living-fossils" / "shots.json").read_text(encoding="utf-8"))
    tmp_ep = W / "ep"
    (tmp_ep / "episodes" / "smoke").mkdir(parents=True)
    (tmp_ep / "episodes" / "smoke" / "shots.json").write_text(
        json.dumps({"lines": shots_spec["lines"][:4]}, ensure_ascii=False), encoding="utf-8")
    (tmp_ep / "episodes" / "smoke" / "footage.json").write_text("{}", encoding="utf-8")
    docplan.ROOT = tmp_ep
    try:
        pl = docplan.build("smoke", tl, 20.0, parts=2, picks=picks)
    finally:
        docplan.ROOT = real
    imgs = [x.get("image") for x in pl["shots"] if x.get("image")]
    check("plan: dropped scene covered", "impact_forest__s0" not in imgs and len(imgs) == len(set(imgs))
          and abs(pl["shots"][-1]["end"] - pl["total_s"]) < 0.01, str([(x["kind"], x.get("image"), x["start"], x["end"])
                                                                       for x in pl["shots"]]))
    from automation import build_doc
    (W / "inputs").mkdir(exist_ok=True)
    build_doc.W = lambda slug: W
    srt = build_doc.subtitles("smoke", tl).read_text(encoding="utf-8")
    check("subtitles", srt.startswith("1\n00:00:00,000 --> ") and "Dinosaurier" in srt, srt[:80])

    # a synthetic still with a foreground disc, as the Kaggle job would leave it
    img = np.zeros((2376, 4224, 3), np.uint8)
    img[:] = (40, 90, 130)
    cv2.circle(img, (2100, 1300), 500, (200, 180, 150), -1)
    alpha = np.zeros((2376, 4224), np.uint8)
    cv2.circle(alpha, (2100, 1300), 500, 255, -1)
    cv2.imwrite(str(W / "layers" / "smoke__s0.jpg"), img)
    cv2.imwrite(str(W / "layers" / "smoke__s0_bg.jpg"), np.full_like(img, (40, 90, 130)))
    cv2.imwrite(str(W / "layers" / "smoke__s0_alpha.png"), alpha)
    total = max(8.0, rep["seconds"] + 1)
    a, b = total / 2, total
    shots = [
        {"kind": "anim", "anim": "timeline", "start": 0.0, "end": a / 2, "transition": "cut", "text": "x", "line": 0},
        {"kind": "recon", "image": "smoke__s0", "fx": "marine_snow", "move": "push_in", "start": a / 2, "end": a,
         "transition": "fade", "text": "x", "line": 0, "label": ["Coelacanth", "Latimeria chalumnae"]},
        {"kind": "anim", "anim": "map", "places": ["east_london"], "start": a, "end": b, "transition": "cut",
         "text": "It is a coelacanth.", "line": 1, "amb": "underwater"},
    ]
    plan = {"slug": "smoke", "total_s": b, "narration_s": rep["seconds"], "xfade_s": 0.6, "shots": shots,
            "parts": [[0, 2], [2, 3]]}
    from pipeline import docrender, docsound
    docrender.SIZES["smoke"] = (640, 360)
    parts = []
    for p in range(2):
        r = docrender.render_part(plan, p, {"media": W, "layers": W / "layers"}, W / f"tmp{p}", W / "parts",
                                  res="smoke")
        parts.append(W / "parts" / f"part_{p}.mp4")
        check(f"part {p}", abs(r["seconds"] - r["expected_s"]) < 0.15, f"{r['seconds']} vs {r['expected_s']}")
    joined = docrender.join(parts, W / "joined.mp4")
    d = docrender.probe(joined)["duration"]
    check("join -c copy", abs(d - b) < 0.2, f"{d:.2f} s vs {b:.2f} s")
    lum = []
    for t in np.arange(0.2, d - 0.2, 0.5):
        out = __import__("subprocess").run(["ffmpeg", "-v", "error", "-ss", f"{t:.2f}", "-i", str(joined),
                                            "-frames:v", "1", "-vf", "format=gray,scale=64:-2", "-f", "rawvideo",
                                            "-"], capture_output=True).stdout
        lum.append(np.frombuffer(out, np.uint8).mean() if out else 0)
    check("no black", min(lum) > 12, f"min luma {min(lum):.0f}")

    amb = json.loads((ROOT / "assets" / "ambience" / "library.json").read_text(encoding="utf-8"))
    music = ROOT / "assets" / "music" / "lost-frontier.mp3"
    s = docsound.build(plan, wav, music, amb, W / "mix.wav")
    check("loudness", abs(s["integrated_lufs"] + 14) <= 0.5, str(s["integrated_lufs"]))
    check("true peak", s["true_peak_dbtp"] <= -1.0, str(s["true_peak_dbtp"]))
    print("\nFAILED:\n  " + "\n  ".join(failures) if failures else "\nsmoke test passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
