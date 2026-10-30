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
sys.path.insert(0, str(ROOT / "scripts"))
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

    # the channel's rules load, and the kernels compile with their watchdog in front
    sys.path.insert(0, str(ROOT / "gpu"))
    cfg = json.loads((ROOT / "config" / "channel.json").read_text(encoding="utf-8"))
    check("rules", cfg["language"] == "de-DE" and cfg["voice"]["engine"] == "piper" and cfg["buffer"] == 2)
    import animate_job
    import watchdog
    compile("JOB = {}\n" + watchdog.WATCHDOG + "\n" + animate_job.SOURCE, "animate", "exec")
    check("LTX kernel compiles", "LTXImageToVideoPipeline" in animate_job.SOURCE)

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
    # the JOB line each kernel starts with must run as Python, booleans included
    for name, job in (("animate", {"frames": [{"id": "x__s0", "prompt": "ä", "motion": "m"}], "steps": 30}),
                      ("recon", {"prompts": {"a": "b"}, "flag": True, "none": None})):
        g2 = {}
        try:
            exec(kaggle_once.job_line(job), g2)
            check(f"kernel JOB line runs ({name})", g2["JOB"] == job)
        except Exception as e:                                   # noqa: BLE001
            check(f"kernel JOB line runs ({name})", False, repr(e)[:200])
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
    # the schedule: Wednesday and Saturday, 17:00 in Berlin (15:00 or 16:00 UTC)
    from datetime import datetime, timezone
    from zoneinfo import ZoneInfo
    from pipeline import youtube
    sl = youtube.slots(4, now=datetime(2026, 10, 29, 12, 0, tzinfo=timezone.utc))
    berlin = [x.astimezone(ZoneInfo("Europe/Berlin")) for x in sl]
    check("slots Wed/Sat 17:00 Berlin", all(b.weekday() in (2, 5) and b.hour == 17 for b in berlin)
          and len({b.date() for b in berlin}) == 4, str([b.isoformat() for b in berlin]))
    kept = youtube.own_tags(["Terra X", "terra-x Doku", "ZDF", "BBC Earth", "Jurassic Park", "Urzeit", "Quastenflosser",
                             "Brückenechse", "Arterien", "Welten der Urzeit"])
    check("tags without other brands", kept == ["Urzeit", "Quastenflosser", "Brückenechse", "Arterien",
                                                "Welten der Urzeit"], str(kept))
    # one episode, one video (the duplicate of 30.10), against a fake YouTube
    import smoke_upload
    for name, ok, detail in smoke_upload.run(W):
        check(name, ok, detail)
    # sections: the markup carries them, the music follows them by mood
    from core import delivery
    secs = [l.section for l in delivery.parse("[section hook]\n> A.\n[section story]\nB.\n[section finale]\nC.")]
    check("sections", secs == ["hook", "story", "finale"], str(secs))
    from automation import write_episode as we
    check("script rules", we.MIN_WORDS == 22 * 128 and we.MAX_WORDS == 27 * 128
          and we.has_hook(we.normalise("[Hook]\n> A.\n[Geschichte]\nB."))
          and not we.has_hook(we.normalise("> A.\n[section story]\nB.")))
    from core import gemini as gm
    fence = chr(96) * 3
    check("Gemma's JSON read from its text", gm._json_from(fence + 'json' + chr(10) + '{"lines": ["a"]}' + chr(10) + fence)
          == {"lines": ["a"]} and gm._json_from('Hier: {"a": 1} fertig') == {"a": 1})
    from core import textllm as tlm
    from core import groq as gq
    check("text: Groq (not the other channels' model), then Gemma, then flash-lite",
          tlm._C["groq_enabled"] and gq.MODEL != "openai/gpt-oss-120b" and tlm.PRIMARY.startswith("gemma")
          and tlm.FALLBACK == "gemini-flash-lite-latest")
    real_ask = tlm.ask
    tlm.ask = lambda *a, **k: {"lines": ["eins", "zwei"]}               # the editor answering two lines for three
    try:
        kept = we.edit(["a", "b", "c"])
        tlm.ask = lambda *a, **k: {"lines": ["1: A", "2: B"]}
        told = we.edit(["a", "[pause 0.8]"])
    finally:
        tlm.ask = real_ask
    check("editor keeps the line count, and pauses", kept == ["a", "b", "c"] and told == ["A", "[pause 0.8]"],
          f"{kept} {told}")
    check("answers read as object or bare list", gm.pick([1, 2], "animals") == [1, 2]
          and gm.pick({"animals": [3]}, "animals") == [3] and gm.pick({"tiere": [4]}, "animals") == [4]
          and gm.pick({"x": 1}, "animals") == [])
    # the quota ledger and the budget: counted per model per Pacific day, a step that
    # would overrun its model waits for the next day
    from core import llm_usage
    from automation import llm_budget
    real_dir, real_def = llm_usage.DIR, llm_budget.DEFERRED
    llm_usage.DIR, llm_budget.DEFERRED = W / "llm_usage", W / "deferred.json"
    try:
        for _ in range(3):
            llm_usage.record("gemini-flash-lite-latest", 100, 20, "species_check")
        u = llm_usage.used("gemini-flash-lite-latest")
        lim = llm_usage.CFG["llm_quota"]["models"]["gemini-flash-lite-latest"]["rpd"]
        check("quota ledger", u["requests"] == 3 and u["in_tokens"] == 300
              and llm_usage.left("gemini-flash-lite-latest")["requests"] == int(lim * 0.85) - 3, str(u))
        check("a step too big for today is deferred", llm_budget.image_ok(10 ** 6)[0] is False)
        llm_budget.defer("smoke-ep", "test")
        d = llm_budget.deferred_now()
        check("deferred until the next Pacific day", d and d["slug"] == "smoke-ep")
    finally:
        llm_usage.DIR, llm_budget.DEFERRED = real_dir, real_def
        llm_usage._RUN.clear()                 # the test's requests are not the channel's
    from core import textllm
    rel = textllm.relevant([{"title": "A", "text": "Der Quastenflosser lebt in Höhlen.\n\n" + "Füllwort " * 40 +
                             "\n\nAmmoniten starben am Ende der Kreidezeit aus, zusammen mit den Dinosauriern."}],
                           "Ammoniten Kreidezeit", 200)
    check("sources trimmed to what the request needs", "Ammoniten" in rel and len(rel) <= 260, rel)
    from automation import plan_episode as pe
    check("abstract frames are not drawn", pe.abstract("dna_helix_abstract", "glowing DNA strands")
          and pe.abstract("molecular_evolution_chart", "x") and not pe.abstract("ancient_lungfish_swamp",
                                                                              "a prehistoric lungfish swimming"))
    spx = {"nautilus": {"living": True, "name": "Perlboot (Nautilus pompilius)"}, "mosasaur": {"living": False, "name": "Mosasaurus"}}
    check("a living animal is never drawn", pe.living_named("nautilus_pinhole_eye", "eye of a nautilus", spx) == "nautilus"
          and pe.living_named("mosasaur_hunt", "a mosasaur hunting", spx) is None)
    sh = "[section hook]\n> Ein Rätsel.\n< Wer überlebte?\n[section story]\nEin Rätsel.\nDie Geschichte beginnt.\n"
    check("hook is stripped, and its copies with it", we.strip_hook(sh).strip() ==
          "[section story]\nDie Geschichte beginnt.", repr(we.strip_hook(sh)))
    from pipeline import docsound
    sc = docsound.score({"total_s": 90, "sections": [{"name": "hook", "start": 0, "end": 30},
                                                      {"name": "climax", "start": 30, "end": 60},
                                                      {"name": "finale", "start": 60, "end": 90}]}, "smoke")
    check("music by mood", [t["mood"] for t in sc["tracks"]] == ["tension", "epic", "calm"]
          and len({t["title"] for t in sc["tracks"]}) == 3)
    # the hook: short moving shots, 1.5-3 s
    hook_tl = [{"text": "Ein Rätsel aus der Tiefe.", "kind": "fast", "section": "hook", "start": 0.0, "end": 9.0},
               {"text": "Es lebt noch heute.", "kind": "base", "section": "story", "start": 9.2, "end": 15.0}]
    hook_spec = {"lines": [{"at": "Ein Rätsel", "n": 0, "section": "hook", "cut": True, "beats": [
        {"kind": "real", "need": "h1"}, {"kind": "real", "need": "h2"}]},
        {"at": "Es lebt", "n": 1, "section": "story", "beats": [{"kind": "real", "need": "h3"}]}]}
    clip = lambda i: {"id": i, "url": "x", "width": 1920, "height": 1080, "duration": 20, "source": "Pixabay"}
    hook_foot = {f"h{k}": {**clip(f"c{k}"), "alternates": [clip(f"c{k}b")]} for k in (1, 2, 3)}
    (tmp_ep / "episodes" / "hook").mkdir(parents=True)
    (tmp_ep / "episodes" / "hook" / "shots.json").write_text(json.dumps(hook_spec), encoding="utf-8")
    (tmp_ep / "episodes" / "hook" / "footage.json").write_text(json.dumps(hook_foot), encoding="utf-8")
    docplan.ROOT = tmp_ep
    try:
        hp = docplan.build("hook", hook_tl, 15.0, parts=1)
    finally:
        docplan.ROOT = real
    hs = [x for x in hp["shots"] if x.get("section") == "hook"]
    check("hook shots 1.5-3 s, moving", hs and all(1.5 <= x["end"] - x["start"] <= 3.05 and x["kind"] == "real"
                                                   for x in hs), str([(x["start"], x["end"]) for x in hs]))
    # the watchdog hands a built episode in the buffer to upload.yml once a token exists
    from automation import release, watchdog
    calls = []
    saved = (release.rows, youtube.available, watchdog.gh)
    release.rows = lambda: [{"slug": "smoke-ep", "status": "built", "run_id": "123"},
                            {"slug": "old-ep", "status": "scheduled", "publish_at": "2026-11-04T16:00:00Z"}]
    youtube.available = lambda: True
    watchdog.gh = lambda *a, **k: calls.append(a)
    try:
        acts = []
        watchdog.check_uploads(acts)
    finally:
        release.rows, youtube.available, watchdog.gh = saved
    check("watchdog starts the upload of a buffered episode",
          len(calls) == 1 and "upload.yml" in calls[0] and "slug=smoke-ep" in calls[0] and "run_id=123" in calls[0],
          str(calls))
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
