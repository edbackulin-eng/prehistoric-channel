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

Owner 07.11, after lebende-fossilien passed with real faults, for every episode from megalodon on:
  background  music + atmosphere together >= 20 dB under the voice in every minute (the
              render's per-minute measurement, docsound.levels); replaces the music-only rule
  ambience    the sea only inside shots whose picture is sea (the render's atmosphere spans)
  repeats     a source - clip or still, any crop or stretch of it, or a frame that looks the
              same (perceptual hash) - at most twice and never within 3 minutes (qa_view)
  match       every frame against its line: the line's animal (the species), no animal the
              text does not name (Gemini, qa_view)
  consistent  one look per animal through the film (Gemini, qa_view)
  diversity   the world around the animal: the main animal <= 40 % of the time; every
              chapter mixes at least 3 kinds (environment, people and history, maps and
              time, the prehistoric world, other animals of the text, the main animal)
  scenes      a dramatic animated scene (a hunt, an attack, the animal beside a human or a
              diver; LTX or real film) in the hook and in every chapter; "wow" counts only
              these
  viewer      Gemini as a viewer on the contact sheet with the text: more than 3 boring,
              repeated or out-of-place marks and the episode does not go up (qa_view)
A chapter is a section of the script when it has more than hook and story, else 3 minutes.

Owner 12.11 - two tiers. Only real faults stop an upload (BLOCKING): an animal on screen other than
the one the voice names (named_animal, match), broken sound or silence (sound), black or broken
frames (frames), wrong text on the thumbnail (text), a source without a confirmed licence (license),
a length outside config length_min_range (duration). Everything else is a score against the
realistic targets of config qa.targets: reported, aimed at, never a reason to stop.
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
BG_UNDER_MIN = 20.0           # music + atmosphere under the voice, every minute (owner 07.11)
MAIN_MAX = 0.40               # the main animal on screen at most this share of the film
KINDS_MIN = 3                 # kinds of picture in every chapter
KIND_MIN_S = 8.0              # ... each at least this long there
CHAPTER_S = 180.0
import re as _re  # noqa: E402
PEOPLE = _re.compile(r"fisher|fischer|scientist|forscher|wissenschaft|museum|archive|archiv|portrait|researcher|"
                     r"diver|taucher|boat|boot|ship|schiff|trawler|expedition|laborator|labor\b|professor|"
                     r"historic photo|old photo|newspaper|zeitung|people|menschen|human|mensch|fossil hunter|"
                     r"excavat|grabung|specimen|präparat|skelet|tooth|zahn|fossil", _re.I)
ACTION = _re.compile(r"attack|hunt|bite|biting|jaws open|open jaws|chase|charg|lunge|strike|ambush|fight|kill|"
                     r"prey|feeding|devour|leap|breach|diver|human for scale|beside a (?:human|diver|boat)|"
                     r"angriff|jagd|jagt|beißt|beute|maul|sprung", _re.I)


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


def main_species(plan, ls):
    """The episode's animal: the one the narration names longest."""
    told = defaultdict(float)
    for s in plan["shots"]:
        for sp in ls.get(s.get("line"), ()):
            told[sp] += s["end"] - s["start"]
    return max(told, key=told.get) if told else None


def kind_of(s, main):
    """What a picture is about (owner 07.11): main, other_animal, environment, people, map,
    prehistoric_world or other."""
    text = " ".join(str(s.get(k) or "") for k in ("topic", "id", "image", "need", "caption", "title")).replace("_", " ")
    c = s.get("clip")
    if isinstance(c, dict):
        text += " " + str(c.get("title") or c.get("tags") or "")
    if s.get("kind") == "anim":
        return "map"
    if s.get("species") and s["species"] == main:
        return "main"
    if PEOPLE.search(text) and s.get("kind") in ("photo", "real") and not s.get("species"):
        return "people"
    if s.get("species"):
        return "other_animal"
    if docplan.is_landscape(s):
        return "environment"
    if s.get("kind") == "recon":
        return "prehistoric_world"
    if PEOPLE.search(text):
        return "people"
    return "other"


def is_scene(s, prompts):
    """A dramatic animated scene: marked so by the plan, or an LTX clip / real film whose
    prompt or topic tells of a hunt, an attack, the animal beside a human or a diver."""
    if s.get("scene"):
        return True
    if not (ltx(s) or moving(s)):
        return False
    pid = str(s.get("id") or str(s.get("need", ""))[4:].split("__s")[0])
    text = " ".join([prompts.get(pid, ""), str(s.get("topic") or ""), str(s.get("need") or "")])
    c = s.get("clip")
    if isinstance(c, dict):
        text += " " + str(c.get("title") or "")
    return bool(ACTION.search(text.replace("_", " ")))


def chapters(plan, shots, total):
    """[(name, start, end)]: the script's sections when it has more than hook and story,
    else the hook and 3-minute chapters after it."""
    secs = [s for s in plan.get("sections") or [] if s.get("name")]
    hook_end = next((s["start"] for s in secs if s["name"] != "hook"), HOOK_S)
    if len([s for s in secs if s["name"] not in ("hook", "story")]) >= 2:
        out = []
        for a, b in zip(secs, secs[1:] + [{"start": total}]):
            out.append((a["name"], a["start"], b["start"]))
        return out
    out = [("hook", 0.0, hook_end)]
    t, k = hook_end, 1
    while t < total - 30:
        out.append((f"chapter {k}", t, min(total, t + CHAPTER_S)))
        t, k = t + CHAPTER_S, k + 1
    return out


def checks_of(plan, sound, thumbs, slug, parts=None, view=None):
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
    # owner 07.11: while a line tells of an animal the picture shows that animal or its world
    # (surroundings, people, maps) - never another animal the line does not name
    wrong_animal = [round(s["start"]) for s in shots if s.get("species") and ls.get(s.get("line"))
                    and s["species"] not in ls.get(s.get("line"), ())]
    land = sum(dur(s) for s in shots if docplan.is_landscape(s)) / max(1e-6, total)
    prehistoric = sum(1 for s in shots if ltx(s))
    add("subject", not wrong_animal and land <= LANDSCAPE_MAX and prehistoric > 0,
        f"animal shown where told: {share or 'no animal told'}; another animal than the line names at "
        f"{wrong_animal[:10] or 'none'} s; landscapes {land:.0%} (max {LANDSCAPE_MAX:.0%}); "
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
    add("music", bool(pm) and not louder and silent == 0,
        f"music under the voice per minute: {[m['under_db'] for m in pm]} dB; "
        f"music as loud as the voice in minute(s): {louder or 'none'}; music silent {silent} s"
        + ("" if pm else " - NO per-minute measurement"))
    # owner 07.11: the whole background, music and atmosphere, >= 20 dB under the voice every minute
    bgu = [m.get("background_under_db") for m in pm]
    bg_bad = [m["minute"] for m in pm if m.get("background_under_db") is None
              or m["background_under_db"] < BG_UNDER_MIN - TOL]
    add("background", bool(pm) and all(v is not None for v in bgu) and not bg_bad,
        f"music + atmosphere under the voice per minute: {bgu} dB (need >= {BG_UNDER_MIN:.0f}); "
        f"minutes too loud: {bg_bad or 'none'}"
        + ("" if pm and all(v is not None for v in bgu) else " - NO measurement of the whole background"))

    # ambience follows the picture
    spans = lv.get("amb_spans")
    if spans is None:
        wrong = [round(s["start"]) for s in shots if s.get("amb") in docplan.SEA_AMB and
                 docplan.amb_for(s, s.get("amb")) != s.get("amb")]
        add("ambience", False, f"NO record of where the render put its atmosphere; by the plan: sea under no "
            f"sea at {wrong[:10] or 'none'} s")
    else:
        allp = plan["shots"]
        wrong = [round(allp[i]["start"]) for sp in spans if sp["amb"] in docplan.SEA_AMB for i in sp["shots"]
                 if i < len(allp) and not docplan.MARINE.search(" ".join(
                     str(allp[i].get(k) or "") for k in ("topic", "id", "image", "need", "species")).replace("_", " "))]
        add("ambience", not wrong, f"{len(spans)} atmosphere stretches, only inside their shots; sea under a "
            f"picture without sea at {wrong[:10] or 'none'} s")

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
    # owner 07.11: "wow" is a dramatic animated scene only, not the number of cuts
    wow_t = sorted(s["start"] for s in shots if is_scene(s, prompts))
    gaps = [b - a for a, b in zip([0.0] + wow_t, wow_t + [total])]
    add("wow", bool(wow_t), f"{len(wow_t)} wow scenes; the longest stretch without one {max(gaps or [total]):.0f} s "
        f"(reported; the rule is a scene in every chapter, see scenes)", wow_max_gap_s=round(max(gaps or [total]), 1))

    # diversity: the world around the animal (owner 07.11)
    main = main_species(plan, ls)
    kinds = defaultdict(float)
    for s in shots:
        kinds[kind_of(s, main)] += dur(s)
    main_share = kinds.get("main", 0.0) / max(1e-6, total)
    poor = {}
    chs = chapters(plan, shots, total)
    for name, a, b in chs:
        if name == "hook":
            continue
        k = defaultdict(float)
        for s in shots:
            o = max(0.0, min(b, s["end"]) - max(a, s["start"]))
            if o > 0:
                k[kind_of(s, main)] += o
        have = [x for x, v in k.items() if v >= KIND_MIN_S and x != "other"]
        if len(have) < KINDS_MIN:
            poor[name] = sorted(have)
    add("diversity", main_share <= MAIN_MAX + TOL / 10 and not poor,
        f"main animal ({main}) {main_share:.0%} of the film (max {MAIN_MAX:.0%}); kinds: "
        f"{ {x: f'{v / total:.0%}' for x, v in sorted(kinds.items(), key=lambda kv: -kv[1])} }; chapters with "
        f"fewer than {KINDS_MIN} kinds: {poor or 'none'}", kind_shares={x: round(v / total, 3) for x, v in kinds.items()})
    # scenes: a dramatic animated scene in the hook and in every chapter; these are the wow
    scenes = [s for s in shots if is_scene(s, prompts)]
    missing = [name for name, a, b in chs if not any(a <= s["start"] < b for s in scenes)]
    add("scenes", not missing, f"{len(scenes)} dramatic animated scenes (LTX or real film); chapters without one: "
        f"{missing or 'none'}", scenes=len(scenes))

    if view is not None:
        REJECTS[:] = view.get("_rejects") or []
        for name in ("repeats", "match", "consistent", "viewer"):
            r = view.get(name)
            if r is None:
                add(name, False, "NOT measured (no video or no frames)")
            else:
                add(name, r[0], r[1])

    ch = (thumbs or {}).get("choice") or {}
    add("thumbnail", bool(ch.get("best")) and ch.get("judge_pass", False),
        f"chosen: {ch.get('best')}; judge: {ch.get('judge') or ch.get('why')}")

    # ---- owner 12.11: the faults that stop an upload
    add("named_animal", not wrong_animal, f"an animal other than the line names at {wrong_animal[:10] or 'none'} s")
    j = ch.get("judge") if isinstance(ch.get("judge"), dict) else {}
    add("text", bool(ch.get("best")) and j.get("german_correct") is not False and j.get("words_readable") is not False,
        f"thumbnail {ch.get('best')}: text {j.get('text')!r}, German correct {j.get('german_correct')}, "
        f"readable {j.get('words_readable')}")
    add("sound", *sound_ok(sound))
    add("duration", *duration_ok(total))
    add("license", *license_ok(plan, slug))
    return out


def sound_ok(sound):
    """Broken sound or silence: clipping, a loudness off the norm, the music over the voice, or a
    stretch of 3 s or more with nothing on any track (the render's own per-second measurement)."""
    sound = sound or {}
    lv = sound.get("levels") or {}
    curve = lv.get("curve_dbfs") or {}
    if not curve:
        return False, "NO measurement of the sound"
    n = max(len(v) for v in curve.values())
    loud = [max((curve[k][t] if t < len(curve[k]) else -200.0) for k in curve) for t in range(n)]
    silent, run = [], 0
    for t, db in enumerate(loud):
        run = run + 1 if db < -55.0 and 2 <= t < n - 3 else 0
        if run == 3:
            silent.append(t - 2)
    pm = lv.get("per_minute") or []
    louder = [m["minute"] for m in pm if m.get("voice_lufs") is not None and m.get("music_lufs") is not None
              and m["music_lufs"] >= m["voice_lufs"]]
    tp, il = sound.get("true_peak_dbtp"), sound.get("integrated_lufs")
    peak_bad = tp is not None and tp > -0.5
    lufs_bad = il is not None and not (-17.0 <= il <= -11.0)
    ok = not silent and not louder and not peak_bad and not lufs_bad
    return ok, (f"silence of 3 s or more at {silent[:10] or 'none'} s; music over the voice in minute(s) "
                f"{louder or 'none'}; true peak {tp} dBTP (max -0.5); loudness {il} LUFS (-17..-11)")


def duration_ok(total):
    lo, hi = json.loads((ROOT / "config" / "channel.json").read_text(encoding="utf-8")).get("length_min_range", [20, 30])
    return lo * 60 <= total <= hi * 60, f"{total / 60:.1f} min ({lo}-{hi})"


LICENSES = _re.compile(r"pixabay content licen|public domain|\bpd\b|cc0|cc[ -]by(?![ -]nc|[ -]nd)|"
                       r"apache|mit\b|ofl|open font", _re.I)


def license_ok(plan, slug):
    """Every clip, track and atmosphere the film uses carries a licence the channel may monetise
    (the same records automation/credits.py writes the description from)."""
    fp = ROOT / "episodes" / slug / "footage.json"
    footage = json.loads(fp.read_text(encoding="utf-8")) if fp.exists() else {}
    lic = {}
    for c in footage.values():
        for x in [c] + (c.get("alternates") or []):
            lic.setdefault(x.get("id"), x.get("license") or ("Pixabay Content License"
                                                             if str(x.get("id", "")).startswith("pixabay") else None))
    bad = []
    for s in plan["shots"]:
        c = s.get("clip") or {}
        cid = c.get("id")
        if s.get("kind") != "real" or not cid or str(cid).startswith("ltx-"):
            continue
        if not LICENSES.search(str(lic.get(cid) or "")):
            bad.append(f"{cid}: {lic.get(cid)}")
            REJECTS.append(str(cid))
    try:
        lib = json.loads((ROOT / "assets" / "music" / "library.json").read_text(encoding="utf-8"))["tracks"]
        used = set(plan.get("music") or [])
        bad += [f"music {t['title']}" for t in lib if t["title"] in used and not LICENSES.search(
            str(t.get("license") or "CC BY 4.0" if "incompetech" in str(t.get("url", "")) else t.get("license") or ""))]
    except (OSError, ValueError, KeyError) as e:
        bad.append(f"music library unreadable ({e})")
    return not bad, f"sources without a confirmed licence: {sorted(set(bad))[:8] or 'none'}"


BLOCKING = ("named_animal", "match", "sound", "frames", "text", "license", "duration")


def score(checks):
    """Owner 12.11: the targets (config qa.targets) as a score; a check that is not BLOCKING is
    measured against its target, never a reason to stop."""
    t = CFG.get("targets") or {}
    for c in checks:
        c["blocking"] = c["name"] in BLOCKING
        if c["blocking"]:
            continue
        if c["name"] == "pace" and "mean_shot_s" in c:
            c["ok"] = c["mean_shot_s"] <= t.get("pace_mean_s", 7.0)
            c["target"] = f"average shot <= {t.get('pace_mean_s', 7.0)} s"
        elif c["name"] == "subject" and "landscape_share" in c:
            c["ok"] = c["landscape_share"] <= t.get("landscape_max", 0.25)
            c["target"] = f"landscapes <= {t.get('landscape_max', 0.25):.0%}"
        elif c["name"] == "motion" and "motion_share" in c:
            c["ok"] = c["motion_share"] >= t.get("motion_min", 0.25)
            c["target"] = f"animals in motion >= {t.get('motion_min', 0.25):.0%}"
        else:
            c["target"] = "the strict rule in its detail"
    scored = [c for c in checks if not c["blocking"]]
    return round(sum(c["ok"] for c in scored) / max(1, len(scored)), 2)


FRAME_CHECK = {}            # the contact sheet's own look at the picture: black frames, decoding errors
REJECTS = []                # sources a rebuild leaves out (the wrong animal, no licence)


def contact_sheet(video, dest, every=5, w=240, cols=12):
    tmp = dest.parent / "_frames"
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    r = subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(video), "-vf",
                        f"fps=1/{every},scale={w}:-2", str(tmp / "f%04d.jpg")], capture_output=True, text=True)
    frames = sorted(tmp.glob("*.jpg"))
    import numpy as _np
    black = [i * every for i, f in enumerate(frames)
             if 0 < i * every and float(_np.asarray(Image.open(f).convert("L")).mean()) < 6.0]
    FRAME_CHECK.update(frames=len(frames), black_s=black,
                       errors=[x for x in (r.stderr or "").splitlines() if x.strip()][:5], rc=r.returncode)
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


def view_checks(plan, video, slug, rel):
    """The checks that look at the finished picture (automation/qa_view.py)."""
    from automation import qa_view
    shots = [s for s in plan["shots"] if not s.get("drop")]
    cache_f = rel / "qa_view_cache.json"
    cache = json.loads(cache_f.read_text(encoding="utf-8")) if cache_f.exists() else {}
    out = {}
    try:
        frames = qa_view.shot_frames(video, shots, rel / "_shot_frames")
        out["repeats"] = qa_view.repeats(shots, frames)[:2]
        meta_p = ROOT / "episodes" / slug / "meta.json"
        topic = json.loads(meta_p.read_text(encoding="utf-8")).get("title", slug) if meta_p.exists() else slug
        m = qa_view.match(shots, frames, line_species(slug), cache, topic=topic)
        out["match"] = m[:2]
        # 12.11: the shots of another animal than the voice names, for the rebuild to leave out
        from pipeline import fragments
        out["_rejects"] = sorted({str(fragments.source_of(shots[b["shot"]]) or "").split(":", 1)[-1].removeprefix("ltx-")
                                  for b in m[2] if b.get("wrong")} - {""})
        out["consistent"] = qa_view.consistent(shots, frames, rel, cache)[:2]

        seen_lines, segs = set(), []
        for s in shots:
            if s.get("line") not in seen_lines:
                seen_lines.add(s.get("line"))
                segs.append((s["start"], s.get("text") or ""))

        def timeline_text(a, b):
            return " ".join(tx for st, tx in segs if a <= st < b)
        out["viewer"] = qa_view.viewer(video, timeline_text, plan["total_s"], rel, cache)[:2]
    except Exception as e:                                               # noqa: BLE001
        log.warning("view checks failed: %s", e)
    finally:
        cache_f.write_text(json.dumps(cache, ensure_ascii=False, indent=1), encoding="utf-8")
    return out


def main(slug, res):
    w = ROOT / "work" / slug
    rel = w / "release"
    plan = json.loads((w / "plan.json").read_text(encoding="utf-8"))
    rep = json.loads((w / f"report_{res}.json").read_text(encoding="utf-8")) if (w / f"report_{res}.json").exists() else {}
    thumbs = json.loads((rel / "thumbnails_report.json").read_text(encoding="utf-8")) \
        if (rel / "thumbnails_report.json").exists() else {}
    view = None
    video = rel / "video.mp4"
    if video.exists():
        view = view_checks(plan, video, slug, rel)
    checks = checks_of(plan, rep.get("sound"), thumbs, slug, rep.get("parts"), view=view)
    out = ROOT / "qa" / slug
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)
    n = contact_sheet(rel / "video.mp4", out / "contact.jpg") if (rel / "video.mp4").exists() else 0
    fc = FRAME_CHECK
    checks.append({"name": "frames", "ok": bool(n) and not fc.get("black_s") and not fc.get("errors") and fc.get("rc") == 0,
                   "detail": f"{n} frames looked at (one every 5 s): black at {fc.get('black_s') or 'none'} s; "
                             f"decoding errors: {fc.get('errors') or 'none'}"})
    sc = score(checks)
    ok = all(c["ok"] for c in checks if c["blocking"])
    q = {"slug": slug, "pass": ok, "score": sc, "checks": checks, "duration_s": rep.get("duration_s"),
         "rejects": sorted(set(REJECTS)),
         "first_minute_moving": plan.get("checks", {}).get("first_minute_moving_share")}
    rel.mkdir(parents=True, exist_ok=True)
    (rel / "qa.json").write_text(json.dumps(q, indent=1, ensure_ascii=False), encoding="utf-8")
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
    md = [f"# QA: {slug}", "", f"**{'PASSED' if ok else 'FAILED'}** - score {sc:.0%} of the targets met", "",
          "| check | blocks upload | ok | target | what was measured |", "|---|---|---|---|---|"]
    md += [f"| {c['name']} | {'yes' if c['blocking'] else 'no'} | {'yes' if c['ok'] else '**no**'} | "
           f"{c.get('target', '')} | {c['detail'].replace('|', '/')} |"
           for c in sorted(checks, key=lambda c: not c["blocking"])]
    md += ["", f"- contact sheet: [contact.jpg](contact.jpg) ({n} frames, one every 5 s)",
           "- voice and music: [sound.png](sound.png)",
           "- thumbnails: " + ", ".join(f"[thumb_{i}](thumb_{i}.jpg)" for i in range(1, 4)), ""]
    (out / "README.md").write_text("\n".join(md), encoding="utf-8")
    log.info("QA %s: %s; score %.0f%%, targets missed %s", slug, "PASSED" if ok else "FAILED " + str(
        [c["name"] for c in checks if c["blocking"] and not c["ok"]]), 100 * sc,
        [c["name"] for c in checks if not c["blocking"] and not c["ok"]])
    return 0 if ok else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("slug")
    ap.add_argument("--res", default="2160p")
    a = ap.parse_args()
    sys.exit(main(a.slug, a.res))
