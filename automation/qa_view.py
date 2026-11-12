"""The QA checks that look at the finished picture (owner 07.11, after lebende-fossilien).

    from automation import qa_view
    frames = qa_view.shot_frames(video, shots)        # one frame from the middle of every shot

  repeats      every shot's frame gets a perceptual hash (dHash, 64 bits). Shots of the same
               source - a clip, a still, a crop or a stretch of either - and shots whose
               frames look the same (Hamming distance <= PHASH_SAME) are one source. A
               source is on screen at most REPEAT_MAX times, and never again within
               REPEAT_GAP_S; a run of shots of one source in a row counts every shot.
  match        Gemini sees every shot's frame with its narration line and the animals the
               line names: the animal on screen must be that animal (the species, not "a
               fish"), and no animal the text does not name may be shown.
  consistent   the frames of each animal of the episode side by side: one look for it all
               through the film (for FLUX: one reference version of the species).
  viewer       Gemini watches as a viewer: the contact sheet (a frame every 5 s) two minutes
               at a time, with the narration of those minutes, and marks what is boring,
               repeated or out of place; more than VIEWER_MAX marks and the episode does not
               go up.
Gemini answers are cached per run in work/<slug>/release/qa_view_cache.json.
"""
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

PHASH_SAME = 8
REPEAT_MAX, REPEAT_GAP_S = 2, 180.0
MATCH_BAD_MAX = 2            # shots Gemini finds showing the wrong or an unnamed animal
VIEWER_MAX = 3
VIEW_MODELS = ("gemini-flash-latest", "gemini-flash-lite-latest", "gemma-4-31b-it", "gemma-4-26b-a4b-it")
BULK_MODELS = ("gemini-flash-lite-latest", "gemma-4-31b-it", "gemma-4-26b-a4b-it", "gemini-flash-latest")


# ------------------------------------------------------------------ frames
def shot_frames(video, shots, dest, w=384):
    """{shot index: path of a frame from the middle of that shot}: one seek and one frame per
    ffmpeg call (one 2160p decoder at a time - many in one call took the runner's memory)."""
    from concurrent.futures import ThreadPoolExecutor
    dest = Path(dest)
    shutil.rmtree(dest, ignore_errors=True)
    dest.mkdir(parents=True)

    def one(item):
        i, t = item
        f = dest / f"s{i:04d}.jpg"
        subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-ss", f"{t:.3f}", "-i", str(video),
                        "-frames:v", "1", "-vf", f"scale={w}:-2", str(f)], check=False)
        return i, f
    mids = [(i, (s["start"] + s["end"]) / 2) for i, s in enumerate(shots)]
    with ThreadPoolExecutor(2) as ex:
        return {i: f for i, f in ex.map(one, mids) if f.exists()}


def dhash(path, n=8):
    im = Image.open(path).convert("L").resize((n + 1, n), Image.LANCZOS)
    a = np.asarray(im, np.int16)
    return int("".join("1" if b else "0" for b in (a[:, 1:] > a[:, :-1]).flatten()), 2)


def hamming(a, b):
    return bin(a ^ b).count("1")


# ------------------------------------------------------------------ repeats
def source_of(s):
    """The source a shot shows, whatever its crop or stretch: the clip, the still or its id."""
    c = s.get("clip") or {}
    if isinstance(c, dict) and c.get("id"):
        return "clip:" + str(c["id"])
    if s.get("image"):
        return "img:" + str(s["image"]).split("#")[0]
    if s.get("anim"):
        return "anim:" + json.dumps(s.get("places") or s.get("anim"), sort_keys=True, default=str)[:120]
    return None


def repeats(shots, frames):
    """(ok, detail, groups): sources by id and by look, their uses and how close they come."""
    parent = list(range(len(shots)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        parent[find(a)] = find(b)
    by_src = {}
    for i, s in enumerate(shots):
        src = source_of(s)
        if src and not src.startswith("anim:"):
            if src in by_src:
                union(i, by_src[src])
            by_src[src] = i
    hashes = {i: dhash(f) for i, f in frames.items()}
    idx = sorted(hashes)
    for x in range(len(idx)):
        for y in range(x + 1, len(idx)):
            a, b = idx[x], idx[y]
            if shots[a].get("kind") == "anim" or shots[b].get("kind") == "anim":
                continue                                # a map is drawn for its places
            if hamming(hashes[a], hashes[b]) <= PHASH_SAME:
                union(a, b)
    groups = {}
    for i in range(len(shots)):
        if shots[i].get("kind") == "anim":
            continue
        groups.setdefault(find(i), []).append(i)
    too_often, too_close = {}, {}
    for g, members in groups.items():
        if len(members) < 2:
            continue
        name = source_of(shots[members[0]]) or f"look#{g}"
        times = sorted(shots[i]["start"] for i in members)
        if len(members) > REPEAT_MAX:
            too_often[name] = len(members)
        if any(b - a < REPEAT_GAP_S for a, b in zip(times, times[1:])):
            too_close[name] = [round(t) for t in times]
    ok = not too_often and not too_close
    return ok, (f"{len(groups)} sources for {len(shots)} shots; more than {REPEAT_MAX} times: "
                f"{dict(list(too_often.items())[:8]) or 'none'}; again within {REPEAT_GAP_S / 60:.0f} min: "
                f"{dict(list(too_close.items())[:6]) or 'none'}"), {"too_often": too_often, "too_close": too_close,
                                                                    "sources": len(groups)}


# ------------------------------------------------------------------ Gemini
def _cache(path):
    p = Path(path)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def _ask(prompt, images, models, cache, key, step):
    if key in cache:
        return cache[key]
    from core import gemini
    try:
        a = gemini.ask(prompt, images=[str(i) for i in images], models=models, temperature=0.0, step=step)
    except Exception as e:                                               # noqa: BLE001
        a = {"error": str(e)[:200]}
    cache[key] = a
    return a


MATCH = """Du prüfst Bilder einer deutschen Urzeit-Doku gegen den gesprochenen Text. Der ganze Film handelt von: {topic}.
Zu jedem nummerierten Bild:
der Satz, der dabei gesprochen wird, und die Tiere, die dieser Satz nennt (Art). Das Bild darf zeigen:
das genannte Tier (genau diese Art - ein Megalodon ist kein beliebiger Hai, ein Quastenflosser kein
beliebiger Fisch), seine Umgebung, Menschen, Fundstücke, Karten oder Grafiken ohne Tier. VERBOTEN: ein
Tier, das der Satz nicht nennt (z. B. ein Frosch, eine Katze, ein moderner Fisch), oder eine falsche Art.
Antworte als JSON: {{"results": [{{"i": <n>, "animal_seen": "<Tier oder none>", "named_animal_shown":
true|false|null, "unnamed_animal": true|false, "ok": true|false, "why": "<kurz>"}}]}}
BILDER:
{rows}"""


def match(shots, frames, line_species, cache, batch=12, topic=""):
    """(ok, detail, bad shots): every frame against its line's text and animals. Owner 12.11: what
    stops an upload is an animal other than the one the voice names (a line that names one and a
    picture of another); an animal under a line that names none is reported, not a fault."""
    rows = [(i, s) for i, s in enumerate(shots) if i in frames and s.get("kind") != "anim"]
    bad = []
    unchecked = 0
    for k in range(0, len(rows), batch):
        part = rows[k:k + batch]
        listing = "\n".join(f"{j}. Satz: \"{(s.get('text') or '')[:220]}\"; genannte Tiere: "
                            f"{', '.join(sorted(line_species.get(s.get('line'), ()) or [])) or 'keine'}"
                            for j, (i, s) in enumerate(part))
        a = _ask(MATCH.format(rows=listing, topic=topic or "Urzeit-Tiere"), [frames[i] for i, _ in part],
                 BULK_MODELS, cache,
                 "match:" + ",".join(str(i) for i, _ in part), "qa-match")
        res = {int(r.get("i", -1)): r for r in (a.get("results") or [])} if isinstance(a, dict) else {}
        for j, (i, s) in enumerate(part):
            r = res.get(j)
            if not r:
                unchecked += 1
                continue
            if not r.get("ok", True) or r.get("unnamed_animal"):
                named = bool(line_species.get(s.get("line")))
                seen = str(r.get("animal_seen") or "none").strip().lower()
                bad.append({"shot": i, "at": round(s["start"], 1), "seen": r.get("animal_seen"),
                            "why": str(r.get("why", ""))[:100],
                            "wrong": named and r.get("named_animal_shown") is False and seen not in ("none", "")})
    wrong = [b for b in bad if b["wrong"]]
    ok = len(wrong) <= MATCH_BAD_MAX and unchecked <= max(3, len(rows) // 20)
    return ok, (f"{len(rows)} shots checked against their line: {len(wrong)} another animal than the voice names "
                f"(blocks above {MATCH_BAD_MAX}){': ' + str([(b['at'], b['seen']) for b in wrong[:8]]) if wrong else ''}; "
                f"{len(bad) - len(wrong)} an animal under a line that names none (reported)"
                f"{': ' + str([(b['at'], b['seen']) for b in bad if not b['wrong']][:6]) if len(bad) > len(wrong) else ''}; "
                f"{unchecked} without an answer"), bad


CONSISTENT = """Diese Bilder stammen aus EINER deutschen Urzeit-Doku und sollen alle dasselbe Tier zeigen: {animal}.
Sieht es überall gleich aus (Körperform, Farbe, Zeichnung, Proportionen), so dass ein Zuschauer es als
dasselbe Tier erkennt? Antworte als JSON: {{"consistent": true|false, "odd_ones": [<Nummern 1..n>],
"why": "<kurz>"}}"""


def montage(paths, dest, w=256, cols=4):
    ims = [Image.open(p).convert("RGB") for p in paths]
    ims = [im.resize((w, int(w * im.height / im.width))) for im in ims]
    h = max(im.height for im in ims)
    rows = (len(ims) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * w, rows * (h + 18)), (10, 10, 10))
    d = ImageDraw.Draw(sheet)
    for k, im in enumerate(ims):
        x, y = (k % cols) * w, (k // cols) * (h + 18)
        sheet.paste(im, (x, y + 18))
        d.text((x + 4, y + 2), str(k + 1), fill=(255, 214, 0))
    sheet.save(dest, quality=85)
    return dest


def consistent(shots, frames, work, cache, per=8):
    """(ok, detail, odd): per animal of the episode, its frames side by side."""
    by = {}
    for i, s in enumerate(shots):
        if i in frames and s.get("species") and s.get("kind") in ("recon", "real"):
            by.setdefault(s["species"], []).append(i)
    odd, asked = {}, 0
    for sp, idx in by.items():
        if len(idx) < 3:
            continue
        pick = [idx[round(k * (len(idx) - 1) / (per - 1))] for k in range(min(per, len(idx)))]
        pick = list(dict.fromkeys(pick))
        sheet = montage([frames[i] for i in pick], Path(work) / f"consistent_{sp}.jpg")
        a = _ask(CONSISTENT.format(animal=sp), [sheet], VIEW_MODELS, cache, f"consistent:{sp}:{pick}",
                 "qa-consistent")
        asked += 1
        if isinstance(a, dict) and a.get("consistent") is False:
            odd[sp] = {"at": [round(shots[pick[n - 1]]["start"]) for n in (a.get("odd_ones") or [])
                              if isinstance(n, int) and 1 <= n <= len(pick)], "why": str(a.get("why", ""))[:120]}
    ok = not odd
    return ok, f"{asked} animals compared across the film; another look for: {odd or 'none'}", odd


VIEWER = """Du bist ein kritischer Zuschauer einer deutschen YouTube-Doku über Urzeittiere. Du siehst zwei Minuten
als Kontaktbogen (ein Bild alle 5 Sekunden, Zeitstempel oben links) und hörst dazu diesen Text:
"{text}"
Markiere jede Stelle, an der ein echter Zuschauer abschalten würde: LANGWEILIG (minutenlang dasselbe Motiv,
nur das Tier ohne Umgebung, Menschen, Karten oder Handlung), WIEDERHOLT (dasselbe Bild oder fast dasselbe noch
einmal), UNPASSEND (das Bild passt nicht zum Text, ein falsches Tier, Schrift im Bild). Sei streng, aber
markiere nur Stellen, die wirklich stören. Antworte als JSON: {{"marks": [{{"t": "<m:ss>", "kind":
"langweilig|wiederholt|unpassend", "why": "<kurz>"}}]}}"""


def viewer(video, timeline_text, total, work, cache, every=5, span=120):
    """(ok, detail, marks): the viewer's marks, two minutes at a time."""
    work = Path(work)
    tmp = work / "_view"
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(video), "-vf",
                    f"fps=1/{every},scale=320:-2", str(tmp / "f%04d.jpg")], check=False)
    frames = sorted(tmp.glob("*.jpg"))
    marks = []
    per = span // every
    for k in range(0, len(frames), per):
        chunk = frames[k:k + per]
        if not chunk:
            continue
        t0 = k * every
        sheet = Image.new("RGB", (6 * 320, ((len(chunk) + 5) // 6) * 196), (10, 10, 10))
        d = ImageDraw.Draw(sheet)
        for j, f in enumerate(chunk):
            im = Image.open(f).convert("RGB").resize((320, 180))
            x, y = (j % 6) * 320, (j // 6) * 196
            sheet.paste(im, (x, y + 16))
            t = t0 + j * every
            d.text((x + 4, y + 2), f"{t // 60}:{t % 60:02d}", fill=(255, 214, 0))
        p = tmp / f"sheet_{k:03d}.jpg"
        sheet.save(p, quality=80)
        text = timeline_text(t0, t0 + span)[:2500]
        a = _ask(VIEWER.format(text=text), [p], VIEW_MODELS, cache, f"viewer:{t0}", "qa-viewer")
        for m in (a.get("marks") or []) if isinstance(a, dict) else []:
            marks.append({"t": m.get("t"), "kind": m.get("kind"), "why": str(m.get("why", ""))[:120],
                          "minute": t0 // 60})
    ok = len(marks) <= VIEWER_MAX
    return ok, (f"{len(marks)} places a viewer would switch off (max {VIEWER_MAX}): "
                f"{[(m['t'], m['kind'], m['why'][:50]) for m in marks[:8]] or 'none'}"), marks
