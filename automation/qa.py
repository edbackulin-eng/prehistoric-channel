"""The QA gate before an upload: what the owner checked by hand on 31.10, measured.

    python automation/qa.py <slug> [--res 2160p]      # merge job, after the thumbnails

Writes work/<slug>/release/qa.json ({"pass": bool, "checks": [...]}) - the upload
refuses to run without a passing one (automation/release.py) - and the episode's
report in qa/<slug>/ (committed): README.md with every number, contact.jpg (a
frame every 5 s), sound.png (voice and music, per second and per minute), the
three thumbnails. Exit 0 when it passes, 1 when not.

Checks (config/channel.json "qa"):
  hook        the first 30 s: only moving pictures with an animal (LTX clips or
              real animal footage), at least 3 of them, a cut every 1.5-3 s, the
              sound hit at the start
  ltx         at least 8 LTX-Video clips in the film
  subject     wherever the narration tells of an animal, that animal is on
              screen >= 60 % of the time; sea, sky and landscape without an
              animal <= 15 % of the film; prehistoric animals in motion
  pace        the average shot 3-5 s (hook 1.5-3 s); no clip more than twice,
              and never twice within 3 minutes
  music       music under the whole film (no silence), in every minute 20-24 dB
              under the voice and never louder than it
  ambience    the sea only under sea pictures
  first_min   >= 70 % moving pictures in the first minute
  thumbnail   one thumbnail chosen that the Gemini judge passed (make_thumbs.py)
  motion      prehistoric animals (living fossils too) in motion >= 40 % of the film:
              LTX clips, real footage of the episode's animals, FLUX animals whose
              2.5D parallax move the render made (owner 02.11)
  dinosaurs   at least 2 dinosaurs in motion: LTX clips of dinosaurs, or FLUX dinosaurs the
              render moved with its 2.5D parallax (owner 06.11)
  wow         a wow moment (an LTX creature, real footage of an animal, a tense
              picture the plan marked) at least every 45 s
"""
import argparse
import json
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from core.logger import get_logger  # noqa: E402
from pipeline import docplan  # noqa: E402

log = get_logger("qa")
CFG = json.loads((ROOT / "config" / "channel.json").read_text(encoding="utf-8")).get("qa", {})
HOOK_S = 30.0
HOOK_CUT = (1.5, 3.0)
HOOK_MOVING_MIN = 3
LTX_MIN = int(CFG.get("ltx_min", 8))
SUBJECT_MIN = 0.60
LANDSCAPE_MAX = 0.15
PACE = (3.0, 5.0)
REPEAT_MAX, REPEAT_GAP_S = 2, 180.0
UNDER = (20.0, 24.0)
FIRST_MINUTE_MOVING = 0.70
TOL = 0.05
MOTION_MIN = float(CFG.get("prehistoric_motion_min", 0.40))
DINO_LTX_MIN = int(CFG.get("dino_ltx_min", 2))
WOW_MAX_S = float(CFG.get("wow_max_s", 45))


def moving(s):
    return s.get("kind") == "real" and bool(s.get("clip"))


def ltx(s):
    return str(s.get("need", "")).startswith("ltx-")


def animal(s):
    return bool(s.get("species")) or ltx(s)


def clip_key(s):
    """The fragment on screen: a stretch of a clip or a framing of a still (pipeline/fragments.py)."""
    from pipeline import fragments
    return fragments.key(s)


def line_species(slug):
    """Narration line index -> the animals its beats name (episodes/<slug>/shots.json)."""
    spec = json.loads((ROOT / "episodes" / slug / "shots.json").read_text(encoding="utf-8"))["lines"]
    out = {}
    for i, e in enumerate(spec):
        n = e.get("n", i)
        out[n] = {b["species"] for b in e["beats"] if b.get("species")}
    return out


def recon_prompts(slug):
    p = ROOT / "episodes" / slug / "recon.json"
    return json.loads(p.read_text(encoding="utf-8")).get("prompts", {}) if p.exists() else {}


def recon_wow(slug):
    p = ROOT / "episodes" / slug / "recon.json"
    return json.loads(p.read_text(encoding="utf-8")).get("wow", []) if p.exists() else []


def parallax_by_shot(parts):
    """Shot index -> the render's 2.5D report of that still (report_<res>.json "parts")."""
    out = {}
    for part in parts or []:
        for c in (part or {}).get("clips", []):
            out[c.get("i")] = c
    return out


def checks_of(plan, sound, thumbs, slug, parts=None):
    shots = [s for s in plan["shots"] if not s.get("drop")]
    total = plan["total_s"]
    dur = lambda s: s["end"] - s["start"]                                  # noqa: E731
    out = []

    def add(name, ok, detail, **data):
        out.append({"name": name, "ok": bool(ok), "detail": detail, **data})

    # the hook: the first 30 s
    hook = [s for s in shots if s["start"] < HOOK_S - TOL]
    still = [round(s["start"], 1) for s in hook if not (moving(s) and animal(s))]
    # a cut is the shot's own length: the last hook shot may run past 30 s, and what is left of
    # it before 30 s is no cut (07.11: a 2.22 s shot from 29.32 s was failed as 0.68 s)
    long_ = [round(dur(s), 2) for s in hook if not (HOOK_CUT[0] - TOL <= dur(s) <= HOOK_CUT[1] + TOL)]
    n_mov = len({clip_key(s) for s in hook if moving(s) and animal(s)})
    hit = bool((sound or {}).get("hook_hit"))
    add("hook", not still and not long_ and n_mov >= HOOK_MOVING_MIN and hit,
        f"{len(hook)} shots in 30 s, {n_mov} moving animal clips (need {HOOK_MOVING_MIN}), "
        f"not moving or no animal at {still or 'none'}, cuts outside 1.5-3 s: {long_ or 'none'}, "
        f"sound hit: {'yes' if hit else 'NO'}")

    n_ltx = len({s["need"] for s in shots if ltx(s)})
    add("ltx", n_ltx >= LTX_MIN, f"{n_ltx} LTX-Video clips (need {LTX_MIN})")

    # the animal on screen where the narration tells of it
    ls = line_species(slug)
    told, shown = defaultdict(float), defaultdict(float)
    for s in shots:
        for sp in ls.get(s["line"], ()):
            told[sp] += dur(s)
            if s.get("species") == sp:
                shown[sp] += dur(s)
    share = {sp: round(shown[sp] / told[sp], 2) for sp in told if told[sp] >= 30}
    low = {sp: v for sp, v in share.items() if v < SUBJECT_MIN}
    land = sum(dur(s) for s in shots if docplan.is_landscape(s)) / max(1e-6, total)
    prehistoric = sum(1 for s in shots if ltx(s))
    add("subject", not low and land <= LANDSCAPE_MAX and prehistoric > 0,
        f"animal on screen where told: {share or 'no animal told'} (need >= {SUBJECT_MIN:.0%}"
        f"{', too low: ' + str(low) if low else ''}); landscapes {land:.0%} (max {LANDSCAPE_MAX:.0%}); "
        f"prehistoric animals in motion: {prehistoric} shots",
        subject_share=share, landscape_share=round(land, 3))

    # pace and repeats
    body = [dur(s) for s in shots if s["start"] >= HOOK_S]
    mean = sum(body) / max(1, len(body))
    hook_mean = sum(dur(s) for s in hook) / max(1, len(hook))
    seen = defaultdict(list)
    for s in shots:
        k = clip_key(s)
        if k:
            seen[k].append(s["start"])
    too_often = {k: len(t) for k, t in seen.items() if len(t) > REPEAT_MAX}
    too_close = {k: [round(x) for x in t] for k, t in seen.items()
                 if any(b - a < REPEAT_GAP_S for a, b in zip(t, t[1:]))}
    add("pace", PACE[0] - TOL <= mean <= PACE[1] + TOL and HOOK_CUT[0] - TOL <= hook_mean <= HOOK_CUT[1] + TOL
        and not too_often and not too_close,
        f"average shot {mean:.1f} s (3-5), hook {hook_mean:.1f} s (1.5-3); shown more than twice: "
        f"{too_often or 'none'}; again within 3 min: {list(too_close)[:8] or 'none'}",
        mean_shot_s=round(mean, 2), repeats=too_close)

    # music
    lv = (sound or {}).get("levels") or {}
    pm = lv.get("per_minute") or []
    bad = [m["minute"] for m in pm if m["under_db"] is not None and not (UNDER[0] - 0.5 <= m["under_db"] <= UNDER[1] + 0.5)]
    louder = [m["minute"] for m in pm if m["voice_lufs"] is not None and m["music_lufs"] is not None
              and m["music_lufs"] >= m["voice_lufs"]]
    silent = lv.get("music_silent_s")
    add("music", bool(pm) and not bad and not louder and silent == 0,
        f"under the voice per minute: {[m['under_db'] for m in pm]} dB (20-24); minutes outside: {bad or 'none'}; "
        f"music as loud as the voice in minute(s): {louder or 'none'}; music silent {silent} s"
        + ("" if pm else " - NO per-minute measurement"))

    # ambience follows the picture
    wrong = [round(s["start"]) for s in shots if s.get("amb") in docplan.SEA_AMB and
             docplan.amb_for(s, s.get("amb")) != s.get("amb")]
    add("ambience", not wrong, f"sea atmosphere under a picture without sea at {wrong[:10] or 'none'} s")

    fm = plan.get("checks", {}).get("first_minute_moving_share", 0)
    add("first_min", fm >= FIRST_MINUTE_MOVING, f"{fm:.0%} moving in the first minute (need 70 %)")

    # prehistoric animals in motion (owner 02.11): LTX clips, real footage of the episode's
    # animals, FLUX animals the render moved with a 2.5D parallax (its report per shot)
    px = parallax_by_shot(parts)
    idx = {id(s): i for i, s in enumerate(plan["shots"])}

    def in_motion(s):
        if ltx(s) or (moving(s) and s.get("species")):
            return True
        if s.get("kind") != "recon" or docplan.is_landscape(s):
            return False
        r = px.get(idx[id(s)]) or {}
        return bool(r.get("layered")) and (r.get("amplitude") or 0) > 0
    mov = sum(dur(s) for s in shots if in_motion(s)) / max(1e-6, total)
    add("motion", mov >= MOTION_MIN - TOL / 10,
        f"prehistoric animals in motion {mov:.0%} of the film (need {MOTION_MIN:.0%}): LTX "
        f"{sum(dur(s) for s in shots if ltx(s)):.0f} s, real footage of the animals "
        f"{sum(dur(s) for s in shots if moving(s) and s.get('species') and not ltx(s)):.0f} s, 2.5D parallax "
        f"{sum(dur(s) for s in shots if s.get('kind') == 'recon' and in_motion(s)):.0f} s" +
        ("" if px else " - NO render report of the parallax moves"), motion_share=round(mov, 3))
    from automation import plan_episode
    prompts = recon_prompts(slug)
    dinos = sorted({s["need"] for s in shots if ltx(s) and plan_episode.is_dino(
        s["need"], prompts.get(str(s.get("id") or s["need"][4:].split("__s")[0]), ""))})
    # owner 06.11: a FLUX dinosaur the render really moved with its 2.5D parallax (the render's
    # report says layered, amplitude > 0) counts as a dinosaur in motion too, while the GPU is
    # not free for an LTX one. A dinosaur already counted by its LTX clip is not counted again.
    animated_ids = {d[4:].split("__s")[0] for d in dinos}
    moved = sorted({str(s.get("id")) for s in shots if not ltx(s) and s.get("kind") == "recon" and s.get("id")
                    and str(s.get("id")) not in animated_ids and in_motion(s)
                    and plan_episode.is_dino(str(s.get("id")), prompts.get(str(s.get("id")), ""))})
    n_dino = len(dinos) + len(moved)
    add("dinosaurs", n_dino >= DINO_LTX_MIN, f"{n_dino} dinosaurs in motion (need {DINO_LTX_MIN}): LTX clips "
        f"{dinos or 'none'}, FLUX dinosaurs moved by the 2.5D parallax {moved or 'none'}"
        + ("" if px else " - NO render report of the parallax moves"))
    wow_ids = set(recon_wow(slug))
    gaps = docplan.wow_gaps(shots, total, wow_ids)
    long_ = [round(sum(gaps[:i])) for i, g in enumerate(gaps) if g > WOW_MAX_S + TOL]
    add("wow", not long_, f"{sum(1 for s in shots if docplan.is_wow(s, wow_ids))} wow moments, the longest gap "
        f"{max(gaps or [0]):.0f} s (max {WOW_MAX_S:.0f}); longer gaps after {long_[:10] or 'none'} s",
        wow_max_gap_s=round(max(gaps or [0]), 1))

    ch = (thumbs or {}).get("choice") or {}
    add("thumbnail", bool(ch.get("best")) and ch.get("judge_pass", False),
        f"chosen: {ch.get('best')}; judge: {ch.get('judge') or ch.get('why')}")
    return out


def contact_sheet(video, dest, every=5, w=240, cols=12):
    tmp = dest.parent / "_frames"
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(video), "-vf",
                    f"fps=1/{every},scale={w}:-2", str(tmp / "f%04d.jpg")], check=True)
    frames = sorted(tmp.glob("*.jpg"))
    if not frames:
        return 0
    fw, fh = Image.open(frames[0]).size
    rows = (len(frames) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * fw, rows * (fh + 14)), (12, 12, 12))
    d = ImageDraw.Draw(sheet)
    for i, f in enumerate(frames):
        x, y = (i % cols) * fw, (i // cols) * (fh + 14)
        sheet.paste(Image.open(f), (x, y + 14))
        t = i * every
        d.text((x + 3, y + 1), f"{t // 60}:{t % 60:02d}", fill=(255, 214, 0))
    sheet.save(dest, quality=80)
    shutil.rmtree(tmp, ignore_errors=True)
    return len(frames)


def sound_graph(sound, dest, w=1600, h=520):
    lv = (sound or {}).get("levels") or {}
    cv = lv.get("curve_dbfs") or {}
    im = Image.new("RGB", (w, h), (250, 250, 250))
    d = ImageDraw.Draw(im)
    if not cv.get("voice"):
        d.text((20, 20), "no sound levels in the render report", fill=(200, 0, 0))
        im.save(dest)
        return
    n = len(cv["voice"])
    top, bot, left = 30, h - 160, 50
    y = lambda db: top + (bot - top) * (0 - max(-80, min(0, db))) / 80              # noqa: E731
    for db in range(0, -81, -20):
        d.line([(left, y(db)), (w - 10, y(db))], fill=(220, 220, 220))
        d.text((5, y(db) - 6), f"{db} dB", fill=(120, 120, 120))
    for key, col in (("amb", (150, 150, 150)), ("music", (220, 120, 0)), ("voice", (30, 90, 200))):
        pts = [(left + (w - left - 10) * i / max(1, n - 1), y(v)) for i, v in enumerate(cv[key])]
        d.line(pts, fill=col, width=1)
    d.text((left, 8), "per second RMS: voice (blue), music (orange), atmosphere (grey)", fill=(0, 0, 0))
    pm = lv.get("per_minute") or []
    by = h - 120
    d.text((left, by - 18), "per minute: music under the voice, dB (green 20-24, red outside)", fill=(0, 0, 0))
    bw = (w - left - 10) / max(1, len(pm))
    for i, m in enumerate(pm):
        u = m.get("under_db")
        ok = u is not None and UNDER[0] - 0.5 <= u <= UNDER[1] + 0.5
        x0 = left + i * bw
        hh = 0 if u is None else min(100, u * 3)
        d.rectangle([x0 + 2, by + 100 - hh, x0 + bw - 2, by + 100], fill=(40, 160, 70) if ok else (210, 40, 40))
        d.text((x0 + 3, by + 100 - hh - 12), "-" if u is None else f"{u:.0f}", fill=(0, 0, 0))
    im.save(dest)


def main(slug, res):
    w = ROOT / "work" / slug
    rel = w / "release"
    plan = json.loads((w / "plan.json").read_text(encoding="utf-8"))
    rep = json.loads((w / f"report_{res}.json").read_text(encoding="utf-8")) if (w / f"report_{res}.json").exists() else {}
    thumbs = json.loads((rel / "thumbnails_report.json").read_text(encoding="utf-8")) \
        if (rel / "thumbnails_report.json").exists() else {}
    checks = checks_of(plan, rep.get("sound"), thumbs, slug, rep.get("parts"))
    ok = all(c["ok"] for c in checks)
    q = {"slug": slug, "pass": ok, "checks": checks, "duration_s": rep.get("duration_s"),
         "first_minute_moving": plan.get("checks", {}).get("first_minute_moving_share")}
    rel.mkdir(parents=True, exist_ok=True)
    (rel / "qa.json").write_text(json.dumps(q, indent=1, ensure_ascii=False), encoding="utf-8")
    out = ROOT / "qa" / slug
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)
    n = contact_sheet(rel / "video.mp4", out / "contact.jpg") if (rel / "video.mp4").exists() else 0
    sound_graph(rep.get("sound"), out / "sound.png")
    # the three the judge passed, the chosen one first; not the 336 px copies (07.11: thumb_1-3
    # were the first files of the folder, small copies and turned-down ones among them)
    judged = [f"thumbnail_{v['id'].split('_', 1)[1]}.jpg" for v in (thumbs.get("variants") or []) if v.get("judge_pass")]
    best = (thumbs.get("choice") or {}).get("best")
    names = ([best] if best else []) + [n for n in judged if n != best]
    picks = [rel / n for n in names if (rel / n).exists()] or \
        [f for f in sorted(rel.glob("thumbnail_*.jpg")) if not f.stem.endswith("_336")]
    for i, f in enumerate(picks[:3], 1):
        Image.open(f).convert("RGB").resize((640, 360)).save(out / f"thumb_{i}.jpg", quality=85)
    (out / "qa.json").write_text(json.dumps({**q, "levels_per_minute": ((rep.get("sound") or {}).get("levels") or {})
                                             .get("per_minute")}, indent=1, ensure_ascii=False), encoding="utf-8")
    md = [f"# QA: {slug}", "", f"**{'PASSED' if ok else 'FAILED'}**", "",
          "| check | ok | what was measured |", "|---|---|---|"]
    md += [f"| {c['name']} | {'yes' if c['ok'] else '**no**'} | {c['detail'].replace('|', '/')} |" for c in checks]
    md += ["", f"- contact sheet: [contact.jpg](contact.jpg) ({n} frames, one every 5 s)",
           "- voice and music: [sound.png](sound.png)",
           "- thumbnails: " + ", ".join(f"[thumb_{i}](thumb_{i}.jpg)" for i in range(1, 4)), ""]
    (out / "README.md").write_text("\n".join(md), encoding="utf-8")
    log.info("QA %s: %s", slug, "PASSED" if ok else "FAILED " + str([c["name"] for c in checks if not c["ok"]]))
    return 0 if ok else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("slug")
    ap.add_argument("--res", default="2160p")
    a = ap.parse_args()
    sys.exit(main(a.slug, a.res))
