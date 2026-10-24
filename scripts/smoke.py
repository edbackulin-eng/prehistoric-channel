"""Smoke test: the whole documentary chain on a few seconds, in a few minutes.

    python scripts/smoke.py

Run by .github/workflows/smoke.yml on every push to dev; only a passing run
moves the code to main.
  1. every module imports;
  2. the opening line is narrated (Kokoro, pronunciation lexicon applied);
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

    from core import narrate
    k = narrate.load_kokoro()
    text = "> Smith named the fish Latimeria chalumnae.\n< It is a coelacanth."
    audio, timeline, rep = narrate.synthesize(k, text, "bm_lewis")
    wav = W / "narration.wav"
    sf.write(wav, audio, 24000)
    check("narration", rep["seconds"] > 2, json.dumps(rep))
    ph = narrate.phonemes(k, "Latimeria chalumnae", "en-gb")
    check("lexicon", "kəlˈʌmniː" in ph, ph)

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
