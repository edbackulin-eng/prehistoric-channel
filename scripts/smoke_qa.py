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


def good_plan():
    shots, t, i = [], 0.0, 0
    while t < 30:                                   # the hook: 2.5 s cuts, moving animals, LTX first
        need = f"ltx-f{i}" if i < 8 else None
        shots.append(_shot(t, t + 2.5, need=need, clip=need or f"h{i}"))
        t, i = t + 2.5, i + 1
    while t < 300:                                  # the body: 4 s shots of the animal, every clip once
        shots.append(_shot(t, t + 4.0, clip=f"b{i}", line=1))
        t, i = t + 4.0, i + 1
    return {"total_s": t, "shots": shots, "checks": {"first_minute_moving_share": 1.0}}


def good_sound(minutes=5):
    pm = [{"minute": m, "voice_lufs": -18.0, "music_lufs_under_voice": -40.0, "music_lufs": -36.0, "under_db": 22.0}
          for m in range(minutes)]
    return {"hook_hit": True, "levels": {"per_minute": pm, "music_silent_s": 0, "curve_dbfs": {}}}


def run(slug="lebende-fossilien"):
    from automation import qa
    out = []
    spec = {0: {"coelacanth"}, 1: {"coelacanth"}}
    qa.line_species = lambda s: spec                        # no shots.json needed
    thumbs = {"choice": {"best": "thumbnail_a.jpg", "judge_pass": True}}

    def run_checks(plan, sound=None, th=thumbs):
        return {c["name"]: c for c in qa.checks_of(plan, sound or good_sound(), th, slug)}
    c = run_checks(good_plan())
    out.append(("QA: a good film passes", all(x["ok"] for x in c.values()),
                str({k: v["detail"] for k, v in c.items() if not v["ok"]})))
    p = good_plan()
    p["shots"][2]["kind"], p["shots"][2]["clip"] = "photo", None
    out.append(("QA: a still in the hook fails", not run_checks(p)["hook"]["ok"], run_checks(p)["hook"]["detail"]))
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
    for s in p["shots"][30:40]:
        s["clip"] = {"id": "same"}
    out.append(("QA: a clip repeated within 3 min fails", not run_checks(p)["pace"]["ok"], run_checks(p)["pace"]["detail"]))
    snd = good_sound()
    snd["levels"]["per_minute"][2].update({"under_db": 15.0})
    snd["levels"]["music_silent_s"] = 4
    out.append(("QA: music too loud in one minute, or silent, fails", not run_checks(good_plan(), snd)["music"]["ok"],
                run_checks(good_plan(), snd)["music"]["detail"]))
    p = good_plan()
    p["shots"][50].update({"species": None, "topic": "historic scientific archives room", "amb": "ocean"})
    out.append(("QA: sea sound under an archive picture fails", not run_checks(p)["ambience"]["ok"],
                run_checks(p)["ambience"]["detail"]))
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
