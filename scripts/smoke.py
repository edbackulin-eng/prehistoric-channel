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
import subprocess
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

    # owner 06.11: the GPU work starts as the free window opens, not after a blind 4 h 10 wait
    from datetime import datetime, timezone
    U = timezone.utc
    opens = {label: kaggle_once.next_open(t, 0.6) for label, t in (
        ("Fri 13:10", datetime(2026, 11, 6, 13, 10, tzinfo=U)), ("Fri 19:50", datetime(2026, 11, 6, 19, 50, tzinfo=U)),
        ("Sat 23:00", datetime(2026, 11, 7, 23, 0, tzinfo=U)), ("Wed 01:00", datetime(2026, 11, 11, 1, 0, tzinfo=U)))}
    check("GPU window opens: What If days 17:30, Wed and Sun 04:00, never into 20:00",
          opens["Fri 13:10"] == datetime(2026, 11, 6, 17, 30, tzinfo=U)
          and opens["Fri 19:50"] == datetime(2026, 11, 7, 17, 30, tzinfo=U)
          and opens["Sat 23:00"] == datetime(2026, 11, 8, 4, 0, tzinfo=U)
          and opens["Wed 01:00"] == datetime(2026, 11, 11, 4, 0, tzinfo=U), str(opens))
    from automation import gpu_pending
    saved_gp = (gpu_pending.PENDING, gpu_pending.NEXT_OPEN, gpu_pending.dispatch)
    gpu_pending.PENDING, gpu_pending.NEXT_OPEN = W / "gpu_pending.json", W / "next_open.txt"
    sent = []
    gpu_pending.dispatch = lambda repo, wf, inputs: sent.append((wf, inputs))
    try:
        now_r = gpu_pending.relay("r", "ltx-extra.yml", {"slug": "x", "relay": "1"})
        gpu_pending.NEXT_OPEN.write_text("2026-11-06T17:30:00Z", encoding="utf-8")
        later = gpu_pending.relay("r", "ltx-extra.yml", {"slug": "x", "relay": "2"})
        gpu_pending.relay("r", "ltx-extra.yml", {"slug": "x", "relay": "3"})          # replaces, not adds
        early = gpu_pending.take_due(datetime(2026, 11, 6, 17, 0, tzinfo=U))
        due = gpu_pending.take_due(datetime(2026, 11, 6, 17, 12, tzinfo=U))
        check("GPU relay: at once when busy; when the window opens later, 20 min before it, once",
              now_r == "relayed now" and len(sent) == 1 and later.startswith("pending") and not early
              and len(due) == 1 and due[0]["inputs"]["relay"] == "3" and due[0]["at"] == "2026-11-06T17:10:00Z"
              and gpu_pending.load() == [], f"{now_r} | {later} | {early} | {due}")
    finally:
        gpu_pending.PENDING, gpu_pending.NEXT_OPEN, gpu_pending.dispatch = saved_gp

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
    from pipeline import fragments
    imgs = [x.get("image") for x in pl["shots"] if x.get("image")]
    frags = [fragments.key(x) for x in pl["shots"] if x.get("image")]
    # no fragment twice: a still may come back in another framing (pipeline/fragments.py, owner 04.11)
    check("plan: dropped scene covered", "impact_forest__s0" not in imgs and len(frags) == len(set(frags))
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
    # the QA gate before every upload, and the hook rule (the owner's review of 31.10)
    sys.path.insert(0, str(ROOT / "automation"))
    import smoke_qa
    for name, ok, detail in smoke_qa.run():
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
    # a living animal real video cannot fill: its stand-ins (FLUX/LTX of the same species) where
    # real video has run out, real first, none more than twice; a line without beats goes on
    lat = lambda need: {"kind": "real", "need": need, "species": "latimeria"}    # noqa: E731
    sb_spec = {"lines": [
        {"at": "Eins", "n": 0, "beats": [lat("r1"), {"kind": "recon", "id": "sa", "species": "latimeria",
                                                   "standby": True}]},
        {"at": "Zwei", "n": 1, "beats": [lat("r2")]}, {"at": "Drei", "n": 2, "beats": []},
        {"at": "Vier", "n": 3, "beats": [lat("r3")]}]}
    sb_tl = [{"text": t, "kind": "base", "section": "story", "start": 8.4 * i, "end": 8.4 * (i + 1)}
             for i, t in enumerate(("Eins.", "Zwei.", "Drei.", "Vier."))]
    # short clips (8 s): their fragments run out, and the stand-ins have to come in
    sb_foot = {k: {**clip(v), "duration": 8} for k, v in (("r1", "x1"), ("r2", "x2"), ("r3", "x3"))}
    (tmp_ep / "episodes" / "standby").mkdir(parents=True)
    (tmp_ep / "episodes" / "standby" / "shots.json").write_text(json.dumps(sb_spec), encoding="utf-8")
    (tmp_ep / "episodes" / "standby" / "footage.json").write_text(json.dumps(sb_foot), encoding="utf-8")
    saved_dp = (docplan.ROOT, docplan.HOOK_WINDOW_S, docplan.REPEAT_GAP_S)
    docplan.ROOT, docplan.HOOK_WINDOW_S, docplan.REPEAT_GAP_S = tmp_ep, 0.0, 5.0
    try:
        sp = docplan.build("standby", sb_tl, 33.6, parts=1,
                           picks={"r1": ["x1"], "r2": [], "r3": [], "sa": ["sa__s0", "sa__s1"]})
    finally:
        docplan.ROOT, docplan.HOOK_WINDOW_S, docplan.REPEAT_GAP_S = saved_dp
    ids = [(x.get("clip") or {}).get("id") or x.get("image") for x in sp["shots"]]
    frags = [fragments.key(x) for x in sp["shots"]]
    check("living animal: real first, then its stand-ins, no fragment more than twice",
          ids[0] == "x1" and {"sa__s0", "sa__s1"} <= set(ids) and all(frags.count(i) <= 2 for i in frags)
          and all(x.get("species") == "latimeria" for x in sp["shots"]) and any(x["line"] == 2 for x in sp["shots"]),
          str([(x["line"], i, x["start"], x["end"]) for x, i in zip(sp["shots"], ids)]))
    # a wow moment every 30-45 s: where the film runs long without one, a tense picture or an LTX
    # creature takes the shot there; an LTX clip stays on screen as long as it runs (owner 02.11)
    w_spec = {"lines": [{"at": f"L{i}", "n": i, "beats": [{"kind": "real", "need": f"n{i}"}] +
                         ([{"kind": "recon", "id": "trex", "species": None}] if i == 1 else []) +
                         ([{"kind": "recon", "id": f"roar{t}", "species": None} for t in "abcd"] if i == 5 else [])}
                        for i in range(6)]}
    w_tl = [{"text": f"L{i}.", "kind": "base", "section": "story", "start": 60.0 * i, "end": 60.0 * (i + 1)}
            for i in range(6)]
    w_foot = {f"n{i}": {**clip(f"c{i}_0"), "alternates": [clip(f"c{i}_{k}") for k in range(1, 20)]} for i in range(6)}
    (tmp_ep / "episodes" / "wow").mkdir(parents=True)
    (tmp_ep / "episodes" / "wow" / "shots.json").write_text(json.dumps(w_spec), encoding="utf-8")
    (tmp_ep / "episodes" / "wow" / "footage.json").write_text(json.dumps(w_foot), encoding="utf-8")
    (tmp_ep / "episodes" / "wow" / "recon.json").write_text(json.dumps({"wow": [f"roar{t}" for t in "abcd"]}), encoding="utf-8")
    w_picks = {**{f"n{i}": [f"c{i}_{k}" for k in range(20)] for i in range(6)}, "trex": ["trex__s0"],
               **{f"roar{t}": [f"roar{t}__s0", f"roar{t}__s1"] for t in "abcd"}}
    saved_dp = (docplan.ROOT, docplan.HOOK_WINDOW_S)
    docplan.ROOT, docplan.HOOK_WINDOW_S = tmp_ep, 0.0
    try:
        wp = docplan.build("wow", w_tl, 360.0, parts=1, picks=w_picks,
                           animated={"trex__s0": {"path": "t.mp4", "w": 768, "h": 448, "duration": 6.7}})
    finally:
        docplan.ROOT, docplan.HOOK_WINDOW_S = saved_dp
    wows = [(round(x["start"]), x.get("image") or (x.get("clip") or {}).get("id")) for x in wp["shots"] if x.get("wow")]
    trex = [x for x in wp["shots"] if x.get("need") == "ltx-trex__s0"]
    early = [i for t, i in wows if t < 150 and str(i).startswith("roar")]
    check("wow moments put in where the film ran long without one; the LTX clip runs its length",
          len(early) >= 2 and trex and trex[0]["end"] - trex[0]["start"] >= 6.0
          and max(t for t, _ in wows if t < 150) > 100 and wows[0][0] <= 45,
          str((wows, [(x["start"], x["end"]) for x in trex], wp["checks"].get("wow_max_gap_s"))))
    check("fragments: a piece of real footage keeps its need (the render names it)",
          all(x.get("need") for x in wp["shots"] if x["kind"] == "real"),
          str([(x["start"], (x.get("clip") or {}).get("id")) for x in wp["shots"] if x["kind"] == "real" and not x.get("need")][:5]))
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
    from automation import gpu_pending
    saved_w = (gpu_pending.PENDING, watchdog.gh, watchdog.now)
    gpu_pending.PENDING = W / "gpu_pending_w.json"
    calls = []
    watchdog.gh = lambda *a, **k: calls.append(a)
    try:
        gpu_pending.add("episode.yml", {"slug": "megalodon", "relay": "4"},
                        datetime(2026, 11, 8, 4, 0, tzinfo=timezone.utc))
        watchdog.now = lambda: datetime(2026, 11, 8, 3, 20, tzinfo=timezone.utc)
        acts = []
        waits = watchdog.check_gpu_pending(acts)
        watchdog.now = lambda: datetime(2026, 11, 8, 3, 41, tzinfo=timezone.utc)
        started = watchdog.check_gpu_pending(acts)
    finally:
        gpu_pending.PENDING, watchdog.gh, watchdog.now = saved_w
    check("watchdog: a build waiting for the GPU window is busy, and is started 20 min before it opens",
          waits is True and started is False and len(calls) == 1 and "episode.yml" in calls[0]
          and "slug=megalodon" in calls[0] and "relay=4" in calls[0], str(calls))
    # 01.11-02.11: one step failed 21 runs in a row, rerun forever, a letter each time. Now: 3 reruns of
    # the same step a day, then the episode waits for the next day and ONE letter goes out
    from automation import llm_budget
    t0 = datetime(2026, 11, 2, 9, 0, tzinfo=timezone.utc)
    fails = [{"id": 900 + k, "run_attempt": 1, "status": "completed", "conclusion": "failure",
              "created_at": "2026-11-02T08:00:00Z", "run_started_at": "2026-11-02T08:00:00Z",
              "updated_at": "2026-11-02T08:00:00Z"} for k in range(6)]
    calls, deferred, letters = [], [], []
    saved = (watchdog.runs, watchdog.gh, watchdog.failed_step, watchdog.failed_log, watchdog.email, watchdog.now,
             llm_budget.defer, llm_budget.deferred_now)
    cur = {}
    watchdog.runs = lambda wf, n=15: [cur["run"]] if wf == "episode.yml" else []
    watchdog.gh = lambda *a, **k: calls.append(a)
    watchdog.failed_step = lambda rid: "build / FLUX on Kaggle"
    watchdog.failed_log = lambda rid: "  SLUG: lebende-fossilien\n82 FLUX frames asked for\n##[error]exit code 1"
    watchdog.email = lambda key, subject, body, mem: letters.append((key, subject))
    watchdog.now = lambda: t0
    llm_budget.defer = lambda slug, why: deferred.append((slug, why))
    llm_budget.deferred_now = lambda: deferred[-1] if deferred else None
    try:
        mem, acts = {"emailed": []}, []
        for r in fails:                                   # every new failure of the same step that day
            cur["run"] = r
            watchdog.check_runs(mem, acts)
            watchdog.check_runs(mem, acts)                # the next pass sees the same failure: nothing new
        reruns = [c for c in calls if c[:2] == ("run", "rerun")]
    finally:
        (watchdog.runs, watchdog.gh, watchdog.failed_step, watchdog.failed_log, watchdog.email, watchdog.now,
         llm_budget.defer, llm_budget.deferred_now) = saved
    check("watchdog: 3 reruns of one step a day, then deferred to tomorrow and one letter",
          len(reruns) == 3 and deferred and deferred[0][0] == "lebende-fossilien" and len({k for k, _ in letters}) == 1,
          str((len(reruns), deferred[:1], letters)))
    # a verdict once given is never asked again, nor its clip fetched (the reruns ate Gemini's quota)
    from pipeline import speciescheck as scq
    ep_v = tmp_ep / "episodes" / "verdicts"
    ep_v.mkdir(parents=True)
    (ep_v / "shots.json").write_text(json.dumps({"lines": [{"at": "Eins", "n": 0, "beats": [
        {"kind": "real", "need": "r1", "species": "latimeria"}]}]}), encoding="utf-8")
    (ep_v / "footage.json").write_text(json.dumps({"r1": {**clip("k1"), "alternates": [clip("k2")]}}), encoding="utf-8")
    (ep_v / "species_verdicts.json").write_text(json.dumps({"k1": {**good, "species_match": False},
                                                            "k2": good}), encoding="utf-8")
    saved = (scq.ROOT, scq.ask, scq._frames, scq._key)
    asked = []
    scq.ROOT, scq._key = tmp_ep, lambda: "k"
    scq.ask = lambda *a, **k: asked.append(a) or good
    scq._frames = lambda *a, **k: asked.append("download") or []
    try:
        pk = scq.check("verdicts", W / "none", W / "vcheck", kinds=("real",))
    finally:
        scq.ROOT, scq.ask, scq._frames, scq._key = saved
    check("species check: cached verdicts, no request and no download", pk.get("r1") == ["k2"] and not asked,
          str((pk, asked)))
    # the GPU: the stand-ins fit the frame cap, and a job must end before Closed Tabs' 20:00
    sys.path.insert(0, str(ROOT / "gpu"))
    import kaggle_once
    check("Kaggle: the cap is the episode's 60 frames", kaggle_once.MAX_FLUX_FRAMES == 60,
          str(kaggle_once.MAX_FLUX_FRAMES))
    # the CLI itself starts (03.11: kaggle 2.2.5 died on import without six and every run waited 4 h)
    cli = subprocess.run([sys.executable, "-c", "import kaggle.api.kaggle_api_extended"], capture_output=True, text=True)
    check("Kaggle: the CLI imports", cli.returncode == 0, (cli.stderr.strip().splitlines() or [""])[-1][:200])
    check("Kaggle: a CLI crash is told from a busy Kaggle",
          kaggle_once.cli_broken("Traceback (most recent call last):\n  File x\nModuleNotFoundError: No module named 'six'")
          and not kaggle_once.cli_broken("ref,title,author,lastRunTime\n"))
    # the stand-ins of a living animal come out of the 60 frames: in place of landscapes first, the
    # animated extinct animals stay (owner 02.11); a plan over the cap (82 frames) is planned again
    import plan_episode as pe2
    sb = tmp_ep / "episodes" / "sbcap"
    sb.mkdir(parents=True)
    lines_txt = ["Der Quastenflosser schwimmt langsam durch die dunkle Tiefe vor den Komoren und sucht "
                 "zwischen den Felsen nach Beute, so wie seine Vorfahren es schon vor Jahrmillionen taten, "
                 f"Nummer {i}." for i in range(20)]
    (sb / "script.txt").write_text("[section story]\n" + "\n".join(lines_txt) + "\n", encoding="utf-8")
    (sb / "meta.json").write_text(json.dumps({"minutes_est": 6.0}), encoding="utf-8")
    (sb / "species.json").write_text(json.dumps({"latimeria": {"name": "Quastenflosser", "living": True,
                                                               "features": ["a"], "forbidden": ["b"]}}), encoding="utf-8")
    land_ids = [f"ocean_sky_{k}" for k in range(4)]
    ext_ids = [f"mawsonia_swim_{k}" for k in range(8)]
    prompts = {**{f: "a calm ocean under a stormy sky" for f in land_ids},
               **{f: "an extinct Mawsonia fish swimming" for f in ext_ids}}
    (sb / "recon.json").write_text(json.dumps({
        "prompts": prompts, "species": {f: None for f in prompts}, "thumbs": {"thumb_a": "x", "thumb_b": "x",
                                                                              "thumb_c": "x"},
        "animate": {**{f: "m" for f in ext_ids}, land_ids[0]: "m"}}), encoding="utf-8")
    (sb / "shots.json").write_text(json.dumps({"lines": [
        {"at": "Der", "n": i, "beats": [{"kind": "real", "need": f"v{i}", "species": "latimeria"}] +
         ([{"kind": "recon", "id": (land_ids + ext_ids)[i], "species": None}] if i < 12 else [])}
        for i in range(20)]}), encoding="utf-8")
    (sb / "footage.json").write_text(json.dumps({f"v{i}": clip(f"q{i}") for i in range(20)}), encoding="utf-8")
    saved = (pe2.ROOT, pe2.textllm.ask)
    pe2.ROOT = tmp_ep

    def no_llm(*a, **k):
        raise RuntimeError("no LLM in the smoke test")
    pe2.textllm.ask = no_llm
    try:
        pe2.standby("sbcap")
        r1 = json.loads((sb / "recon.json").read_text(encoding="utf-8"))
        over = dict(r1)                        # the plan of 31.10: stand-ins on top, 82 frames
        over["prompts"] = {**r1["prompts"], **{f"standby_latimeria_x{k}": "p" for k in range(25)}}
        over["standby"] = {**r1["standby"], **{f"standby_latimeria_x{k}": "latimeria" for k in range(25)}}
        (sb / "recon.json").write_text(json.dumps(over), encoding="utf-8")
        pe2.standby("sbcap")
        r2 = json.loads((sb / "recon.json").read_text(encoding="utf-8"))
    finally:
        pe2.ROOT, pe2.textllm.ask = saved
    n_frames = lambda r: 2 * len(r["prompts"]) + 2 * len(r["thumbs"])            # noqa: E731
    shots_after = json.loads((sb / "shots.json").read_text(encoding="utf-8"))["lines"]
    beat_ids = {b.get("id") for e in shots_after for b in e["beats"] if b["kind"] == "recon"}
    check("stand-ins inside the 60 FLUX frames: landscapes give way, animated extinct animals stay",
          n_frames(r1) <= 60 and n_frames(r2) <= 60 and r1["standby"] and all(f in r1["prompts"] for f in ext_ids)
          and not any(f in r1["prompts"] for f in land_ids[1:]) and set(r2["prompts"]) <= beat_ids | set(r2["thumbs"])
          | set(r2["prompts"]) and not any(f.startswith("standby_latimeria_x") for f in r2["prompts"]),
          str((n_frames(r1), n_frames(r2), len(r1["standby"]), r1.get("standby_report"))))
    mon = datetime(2026, 11, 2, 18, 0, tzinfo=timezone.utc)
    check("Kaggle: a job that would run past 20:00 UTC waits, a short one starts",
          kaggle_once.in_their_hours(mon, 2.5) and not kaggle_once.in_their_hours(mon, 0.9)
          and kaggle_once.in_their_hours(datetime(2026, 11, 2, 17, 0, tzinfo=timezone.utc), 0.5)
          and not kaggle_once.in_their_hours(datetime(2026, 11, 4, 9, 0, tzinfo=timezone.utc), 3.0),
          str([kaggle_once.in_their_hours(mon, 2.5), kaggle_once.in_their_hours(mon, 0.9)]))
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
    # 04.11: the render takes n frames and stops; the parallax report must be there all the same
    from pipeline import parallax
    pimg = np.zeros((360, 640, 3), np.uint8)
    pa = np.zeros((360, 640), np.uint8)
    pa[100:250, 200:400] = 255
    parallax.frames.last_report = None
    for _, _f in zip(range(3), parallax.frames(pimg, pimg.copy(), pa, "push_in", 1.0, 10, 320, 180)):
        pass
    check("parallax: its report comes with the first frame", (parallax.frames.last_report or {}).get("layered") is True,
          str(parallax.frames.last_report))
    # a scene the checker could not judge (quota, frame missing) is unknown, never "dropped"
    from pipeline import speciescheck
    v = speciescheck.passes({"error": "HTTP 429"}, "prehistoric_scene")
    check("species check: a 429 is no verdict", v[0] is None, str(v))
    print("\nFAILED:\n  " + "\n  ".join(failures) if failures else "\nsmoke test passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
