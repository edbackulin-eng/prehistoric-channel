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


def moving(s):
    return s.get("kind") == "real" and bool(s.get("clip"))


def ltx(s):
    return str(s.get("need", "")).startswith("ltx-")


def animal(s):
    return bool(s.get("species")) or ltx(s)


def clip_key(s):
    return (s.get("clip") or {}).get("id") or s.get("image")


def line_species(slug):
    """Narration line index -> the animals its beats name (episodes/<slug>/shots.json)."""
    spec = json.loads((ROOT / "episodes" / slug / "shots.json").read_text(encoding="utf-8"))["lines"]
    out = {}
    for i, e in enumerate(spec):
        n = e.get("n", i)
        out[n] = {b["species"] for b in e["beats"] if b.get("species")}
    return out


def checks_of(plan, sound, thumbs, slug):
    shots = [s for s in plan["shots"] if not s.get("drop")]
    total = plan["total_s"]
    dur = lambda s: s["end"] - s["start"]                                  # noqa: E731
    out = []

    def add(name, ok, detail, **data):
        out.append({"name": name, "ok": bool(ok), "detail": detail, **data})

    # the hook: the first 30 s
    hook = [s for s in shots if s["start"] < HOOK_S - TOL]
    still = [round(s["start"], 1) for s in hook if not (moving(s) and animal(s))]
    long_ = [round(dur(s), 2) for s in hook if not (HOOK_CUT[0] - TOL <= min(dur(s), HOOK_S - s["start"])
                                                       and dur(s) <= HOOK_CUT[1] + TOL)]
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
    checks = checks_of(plan, rep.get("sound"), thumbs, slug)
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
    for i, f in enumerate(sorted(rel.glob("thumbnail_*.jpg"))[:3], 1):
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
