"""Smoke test of the QA gate (automation/qa.py) on hand-made plans: a good one passes,
each of the owner's complaints of 31.10 fails its check. Returns (name, ok, detail) rows."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _shot(t0, t1, kind="real", species="coelacanth", need=None, clip=None, amb="underwater", line=0, topic=None):
    s = {"start": t0, "end": t1, "kind": kind, "species": species, "amb": amb, "line": line, "topic": topic}
    if kind == "real":
        s["clip"] = {"id": clip or need or f"c{t0:.0f}"}
    if need:
        s["need"] = need
    return s


# owner 07.11: a good film is the world around the animal - in every 40 s the main animal 12 s,
# another animal of the line 8 s, people and history 8 s, a map 4 s, the prehistoric world 4 s,
# the sea 4 s - and a dramatic scene (an LTX hunt) in every chapter
BODY = [("main", 3), ("other", 2), ("people", 2), ("map", 1), ("world", 1), ("env", 1)]


def _body_shot(t, kind, i):
    if kind == "main":
        return _shot(t, t + 4.0, clip=f"b{i}", line=1)
    if kind == "other":
        return _shot(t, t + 4.0, clip=f"o{i}", species="shark", line=1)
    if kind == "people":
        s = _shot(t, t + 4.0, kind="photo", species=None, amb="room", line=1, topic="museum scientist with fossil")
        s["image"] = f"p{i}"
        return s
    if kind == "map":
        return {"start": t, "end": t + 4.0, "kind": "anim", "anim": "map", "places": [f"place{i}"], "line": 1,
                "amb": "room", "species": None}
    if kind == "world":
        s = _shot(t, t + 4.0, kind="recon", species=None, line=1, topic="carboniferous fern swamp")
        s["image"] = f"w{i}__s0"
        return s
    s = _shot(t, t + 4.0, species=None, clip=f"e{i}", line=1, topic="deep sea cave underwater")
    return s


def good_plan():
    shots, t, i = [], 0.0, 0
    while t < 30:                                   # the hook: 2.5 s cuts, moving animals, LTX first
        need = (f"ltx-tyrannosaurus_{i}__s0" if i < 2 else f"ltx-f{i}") if i < 8 else None
        shots.append(_shot(t, t + 2.5, need=need, clip=need or f"h{i}"))
        t, i = t + 2.5, i + 1
    shots[0]["scene"] = True
    k = 0
    while t < 300:                                  # the body: every source once
        kind = [x for x, n in BODY for _ in range(n)][k % 10]
        s = _body_shot(t, kind, i)
        if k % 10 == 0:                             # an LTX hunt opens every 40 s
            s.update({"need": f"ltx-hunt{i}__s0", "scene": True})
            s["clip"] = {"id": f"ltx-hunt{i}__s0"}
        shots.append(s)
        t, i, k = t + 4.0, i + 1, k + 1
    return {"total_s": t, "shots": shots, "checks": {"first_minute_moving_share": 1.0}}


def good_sound(minutes=5):
    pm = [{"minute": m, "voice_lufs": -18.0, "music_lufs_under_voice": -40.0, "music_lufs": -36.0, "under_db": 22.0,
           "background_lufs_under_voice": -40.5, "background_under_db": 22.5} for m in range(minutes)]
    sec = minutes * 60
    curve = {"voice": [-20.0] * sec, "music": [-42.0] * sec, "amb": [-45.0] * sec}
    return {"hook_hit": True, "true_peak_dbtp": -1.4, "integrated_lufs": -14.0,
            "levels": {"per_minute": pm, "music_silent_s": 0, "curve_dbfs": curve, "amb_spans": []}}


def run(slug="lebende-fossilien"):
    from automation import qa
    out = []
    spec = {0: {"coelacanth"}, 1: {"coelacanth", "shark"}}
    qa.line_species = lambda s: spec                        # no shots.json needed
    qa.recon_prompts = lambda s: {}
    qa.recon_wow = lambda s: ["tense_mawsonia"]
    thumbs = {"choice": {"best": "thumbnail_a.jpg", "judge_pass": True}}

    # a 5-minute test film of made-up clips: its length and licences are judged on their own below
    qa.duration_ok, qa.license_ok = (lambda total: (True, "smoke")), (lambda plan, slug: (True, "smoke"))

    def run_checks(plan, sound=None, th=thumbs):
        return {c["name"]: c for c in qa.checks_of(plan, sound or good_sound(), th, slug)}
    c = run_checks(good_plan())
    out.append(("QA: a good film passes", all(x["ok"] for x in c.values()),
                str({k: v["detail"] for k, v in c.items() if not v["ok"]})))
    # owner 12.11: only real faults block; the rest is a score against config qa.targets
    import importlib
    q2 = importlib.reload(qa)
    snd = good_sound()
    for k in ("voice", "music", "amb"):
        snd["levels"]["curve_dbfs"][k][100:105] = [-90.0] * 5
    out.append(("QA 12.11: 5 s of silence blocks", q2.sound_ok(snd)[0] is False and q2.sound_ok(good_sound())[0],
                q2.sound_ok(snd)[1]))
    out.append(("QA 12.11: a length outside the range blocks", q2.duration_ok(300.0)[0] is False
                and q2.duration_ok(25 * 60.0)[0], q2.duration_ok(300.0)[1]))
    lic = q2.license_ok({"shots": [{"kind": "real", "clip": {"id": "nolicence-1"}}], "music": []}, "no-such-episode")
    out.append(("QA 12.11: a clip with no licence blocks", lic[0] is False, lic[1]))
    chk = [{"name": "pace", "ok": False, "detail": "", "mean_shot_s": 6.0},
           {"name": "subject", "ok": False, "detail": "", "landscape_share": 0.3},
           {"name": "viewer", "ok": False, "detail": ""}, {"name": "sound", "ok": True, "detail": ""}]
    sc = q2.score(chk)
    out.append(("QA 12.11: pace, landscapes, viewer are scores, not gates",
                not any(c["blocking"] for c in chk if c["name"] != "sound") and chk[0]["ok"] and not chk[1]["ok"]
                and 0 < sc < 1, str([(c["name"], c["ok"], c.get("target")) for c in chk])))
    qa.duration_ok, qa.license_ok = (lambda total: (True, "smoke")), (lambda plan, slug: (True, "smoke"))
    qa.line_species = lambda s: spec
    qa.recon_prompts = lambda s: {}
    qa.recon_wow = lambda s: ["tense_mawsonia"]
    p = good_plan()
    p["shots"][2]["kind"], p["shots"][2]["clip"] = "photo", None
    out.append(("QA: a still in the hook fails", not run_checks(p)["hook"]["ok"], run_checks(p)["hook"]["detail"]))
    # owner 07.11: the last hook shot running past 30 s is judged by its own length (2.22 s from 29.32 s)
    p = good_plan()
    shots, t, i = [], 0.0, 0
    while t < 29.0:                                 # 2.22 s cuts: the 14th runs from 28.86 s to 31.08 s
        need = (f"ltx-tyrannosaurus_{i}__s0" if i < 2 else f"ltx-f{i}") if i < 8 else None
        shots.append(_shot(t, t + 2.22, need=need, clip=need or f"h{i}"))
        t, i = t + 2.22, i + 1
    body = [s for s in p["shots"] if s["start"] >= 30]
    shift = t - body[0]["start"]
    for s in body:
        s["start"], s["end"] = s["start"] + shift, s["end"] + shift
    p["shots"] = shots + body
    out.append(("QA: a 2.22 s shot across 30 s is no bad cut", "cuts outside 1.5-3 s: none" in run_checks(p)["hook"]["detail"],
                run_checks(p)["hook"]["detail"]))
    p = good_plan()
    p["shots"][3]["end"] = p["shots"][3]["start"] + 1.0
    out.append(("QA: a 1.0 s cut in the hook still fails", not run_checks(p)["hook"]["ok"], run_checks(p)["hook"]["detail"]))
    p = good_plan()
    for s in p["shots"]:
        s["need"] = None
    out.append(("QA: fewer than 8 LTX clips fail", not run_checks(p)["ltx"]["ok"], run_checks(p)["ltx"]["detail"]))
    p = good_plan()
    for s in p["shots"][40:]:
        s.update({"species": None, "topic": "lonely ocean waves dramatic lighting"})
    r = run_checks(p)["subject"]
    out.append(("QA: the animal off screen, sea instead, fails", not r["ok"], r["detail"]))
    p = good_plan()
    p["shots"][30]["species"] = "frog"
    r = run_checks(p)["subject"]
    out.append(("QA: an animal the line does not name fails (owner 07.11)", not r["ok"], r["detail"]))
    snd = good_sound()
    snd["levels"]["music_silent_s"] = 4
    out.append(("QA: music silent fails", not run_checks(good_plan(), snd)["music"]["ok"],
                run_checks(good_plan(), snd)["music"]["detail"]))
    snd = good_sound()
    snd["levels"]["per_minute"][2].update({"background_under_db": 18.5})
    out.append(("QA: music + atmosphere under 20 dB below the voice in one minute fails (owner 07.11)",
                not run_checks(good_plan(), snd)["background"]["ok"], run_checks(good_plan(), snd)["background"]["detail"]))
    snd = good_sound()
    for m in snd["levels"]["per_minute"]:
        m.pop("background_under_db")
    out.append(("QA: no measurement of the whole background fails", not run_checks(good_plan(), snd)["background"]["ok"], ""))
    p = good_plan()
    p["shots"][50].update({"species": None, "topic": "historic scientific archives room", "amb": "ocean",
                           "clip": {"id": "archive1"}})
    snd = good_sound()
    snd["levels"]["amb_spans"] = [{"start": p["shots"][50]["start"], "end": p["shots"][50]["end"], "amb": "ocean",
                                   "shots": [50]}]
    out.append(("QA: sea sound under an archive picture fails", not run_checks(p, snd)["ambience"]["ok"],
                run_checks(p, snd)["ambience"]["detail"]))
    snd = good_sound()
    snd["levels"].pop("amb_spans")
    out.append(("QA: no record of the atmosphere's stretches fails", not run_checks(good_plan(), snd)["ambience"]["ok"], ""))
    # owner 07.11: diversity - the main animal at most 40 % of the film, 3 kinds in every chapter
    p = good_plan()
    for s in p["shots"][12:]:
        if s.get("kind") != "anim":
            s.update({"species": "coelacanth", "topic": None})
    r = run_checks(p)["diversity"]
    out.append(("QA: the main animal nearly all the time fails diversity", not r["ok"], r["detail"]))
    out.append(("QA: the good film's mix passes diversity", run_checks(good_plan())["diversity"]["ok"],
                run_checks(good_plan())["diversity"]["detail"]))
    # owner 07.11: a dramatic scene in the hook and in every chapter
    p = good_plan()
    for s in p["shots"]:
        if s["start"] >= 210:
            s.pop("scene", None)
            if str(s.get("need", "")).startswith("ltx-hunt"):
                s["need"] = s["need"].replace("hunt", "swim")
    r = run_checks(p)["scenes"]
    out.append(("QA: a chapter without a dramatic scene fails", not r["ok"], r["detail"]))
    p = good_plan()
    for s in p["shots"]:
        if str(s.get("need", "")).startswith("ltx-tyrannosaurus"):
            s["need"] = "ltx-ammonite__s0"
    out.append(("QA: fewer than 2 dinosaur LTX clips fail", not run_checks(p)["dinosaurs"]["ok"],
                run_checks(p)["dinosaurs"]["detail"]))
    # owner 06.11: one LTX dinosaur and one FLUX dinosaur the render moved with its 2.5D parallax pass;
    # the same FLUX still without the render's move does not count
    p = good_plan()
    p["shots"][1]["need"] = "ltx-ammonite__s0"
    for s in p["shots"][40:43]:
        s.update({"kind": "recon", "image": "cretaceous_trex_stride__s0", "id": "cretaceous_trex_stride", "species": None})
        s.pop("clip", None)
    unmoved = qa.checks_of(p, good_sound(), thumbs, slug)
    moved_parts = [{"clips": [{"i": i, "kind": "recon", "layered": True, "amplitude": 0.04} for i in range(40, 43)]}]
    moved = qa.checks_of(p, good_sound(), thumbs, slug, moved_parts)
    d_un = next(c for c in unmoved if c["name"] == "dinosaurs")
    d_mv = next(c for c in moved if c["name"] == "dinosaurs")
    out.append(("QA: a FLUX dinosaur counts only when the 2.5D parallax moved it", not d_un["ok"] and d_mv["ok"],
                d_mv["detail"]))
    for s in p["shots"][40:43]:                     # the LTX dinosaur's own still is not a second dinosaur
        s.update({"image": "tyrannosaurus_0__s0", "id": "tyrannosaurus_0"})
    p["shots"][1]["need"] = "ltx-ammonite__s0"
    p["shots"][0]["need"] = "ltx-tyrannosaurus_0__s0"
    again = next(c for c in qa.checks_of(p, good_sound(), thumbs, slug, moved_parts) if c["name"] == "dinosaurs")
    out.append(("QA: an LTX dinosaur's still is not counted twice", not again["ok"], again["detail"]))
    p = good_plan()                     # the body as FLUX stills: moving only where the render moved them
    for s in p["shots"][12:]:
        s.update({"kind": "recon", "image": "mawsonia__s0", "id": "mawsonia", "species": None, "topic": "mawsonia"})
        s.pop("clip", None)
    r = qa.checks_of(p, good_sound(), thumbs, slug)
    still_fail = not next(c for c in r if c["name"] == "motion")["ok"]
    parts = [{"clips": [{"i": i, "kind": "recon", "layered": True, "amplitude": 0.04} for i in range(len(p["shots"]))]}]
    r2 = qa.checks_of(p, good_sound(), thumbs, slug, parts)
    out.append(("QA: animals in motion - stills count only with their 2.5D move", still_fail and
                next(c for c in r2 if c["name"] == "motion")["ok"], next(c for c in r2 if c["name"] == "motion")["detail"]))
    # owner 07.11: wow is a dramatic scene only - tense stills and cuts do not make one
    for s in p["shots"]:
        s.pop("scene", None)
    for s in p["shots"][12::9]:
        s["id"] = "tense_mawsonia"
    r3 = qa.checks_of(p, good_sound(), thumbs, slug, parts)
    out.append(("QA: tense stills are no wow scenes", not next(c for c in r3 if c["name"] == "scenes")["ok"],
                next(c for c in r3 if c["name"] == "scenes")["detail"]))
    # owner 07.11: the view checks - perceptual-hash repeats (a crop of one still is the same source)
    from automation import qa_view
    from PIL import Image, ImageDraw
    import tempfile
    tmp = Path(tempfile.mkdtemp())
    shots, frames = [], {}
    for i in range(6):
        im = Image.new("L", (160, 90), 30)
        ImageDraw.Draw(im).ellipse((10 + i * 22, 10, 50 + i * 22, 80), fill=230)
        f = tmp / f"f{i}.jpg"
        im.save(f)
        frames[i] = f
        shots.append({"start": i * 200.0, "end": i * 200.0 + 4, "kind": "recon", "image": f"still{i}__s0"})
    ok_all = qa_view.repeats(shots, frames)[0]
    shots[3]["image"] = "still1__s0#2"                      # a crop of still1, 400 s later
    shots[4]["start"], shots[4]["end"] = 640.0, 644.0
    frames[4] = frames[1]                                    # the same picture under another name
    bad = qa_view.repeats(shots, frames)
    out.append(("QA: a crop of a still and a look-alike frame are repeats (perceptual hash)", ok_all and not bad[0],
                bad[1]))
    out.append(("QA: no judged thumbnail fails", not run_checks(good_plan(), th={"choice": {"best": None}})["thumbnail"]["ok"], ""))
    # the hook rule for every episode
    import write_episode as we
    yes = {"concrete_scene": True, "question": True, "not_all": True, "facts_supported": True}
    owner = ["22. Dezember 1938.", "Ein Fischer zieht etwas aus dem Indischen Ozean, das es seit 66 Millionen Jahren "
             "nicht mehr geben dürfte.", "Wie konnte dieser Fisch überleben, als die Dinosaurier starben?",
             "Und er ist nicht der Einzige."]
    out.append(("hook rule: the owner's hook passes", we.hook_rule(owner, yes)[0], str(we.hook_rule(owner, yes))))
    sie = owner[:3] + ["Erfahren Sie, wie er überlebte."]
    out.append(("hook rule: 'Erfahren Sie' fails", not we.hook_rule(sie, yes)[0], str(we.hook_rule(sie, yes))))
    return out
