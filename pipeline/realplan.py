"""The edit plan of a real-footage episode (owner 15.11): every picture once, each one fits its words.

    build(slug, timeline, narration_s, parts) -> plan.json in the shape docplan.build gives,
    so the render (docrender), the sound (docsound) and the QA (automation/qa.py) take it as is.

1. each narration line is tagged (text model, in chunks): which of the episode's animals it names,
   and which kinds of picture suit it best (fossil, skeleton, palaeo-art, the living relative,
   excavation, museum, landscape ...);
2. the line's time is cut into shots of 5-8 s (a still 5-7 s);
3. pictures are given out most-constrained-first: a shot whose line allows few pictures is filled
   before one that allows many, so the rare ones (the extinct animal's fossils and paintings) go
   where only they fit. A picture is used ONCE in the film. Under a line that names an animal the
   picture shows that animal (alive, as a fossil, a skeleton or a painting) or no animal at all -
   never another one;
4. a shot nothing fits runs into its neighbour (a longer shot, never a repeat).
Also written: episodes/<slug>/shots.json (the animals each line names - what the QA checks the
picture against), footage.json (every picture used, with its licence) and thumbs.json.
"""
import json
import math
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LEAD_S = 0.25
SHOT_S = 5.5               # aimed length of a shot
SHOT_MAX_S = 6.0           # owner 15.11: a photograph at most 6 s - every shot is cut to that
STILL_MAX_S = 6.0
PLAIN_RUN_MAX = 3          # owner 15.11: at most 3 photographs in a row without film or an archive picture between
HOLD_MAX_S = 14.0          # a shot that has to run on for want of pictures: up to this
XFADE_S = 0.6
OUTRO_S = 3.0
MOVES = ["push_in", "truck_right", "pull_out", "truck_left", "crane_up", "push_in_left"]
EXTINCT_CATS = ("fossil", "skeleton", "palaeoart", "museum")
WANT_ALL = ["archive_photo", "archive_film", "archive_illustration", "portrait", "excavation", "palaeontologist",
            "preparation", "museum", "skeleton", "fossil", "palaeoart", "living_animal", "landscape"]
ARCHIVE = {"archive_photo", "archive_film", "archive_illustration", "portrait"}
FILL_ONLY = {"living_animal", "landscape"}
FILL_SHARE = 0.25          # owner 15.11: living relatives and landscapes only fill in
SEA = re.compile(r"ocean|sea\b|wave|underwater|reef|coast|beach|shore|water|swim", re.I)

TAG = """Für jede nummerierte Zeile einer deutschen Doku über eine Fossil-Entdeckung: Welche dieser Tiere nennt oder
meint die Zeile (auch über "er", "sie", "diese Jäger", wenn klar ist, wer gemeint ist)? Tiere: {animals}.
Welche Personen, Orte, Museen oder Jahre nennt sie ("names", wie im Text)? Und welche Bildarten passen am besten -
die beste zuerst, 2-4 Stück, aus: archive_photo, archive_film, archive_illustration, portrait, excavation,
palaeontologist, preparation, museum, skeleton, fossil, palaeoart, living_animal, landscape.
(portrait, wenn die Zeile von einer Person erzählt; ein lebendes Tier nur, wenn die Zeile von diesem heute
lebenden Tier spricht.)
Antworte als JSON: {{"lines": [{{"i": <n>, "animals": ["<key>"], "names": ["..."], "want": ["..."]}}]}}
ZEILEN:
{rows}"""


def _names(a):
    out = {a["key"].replace("_", " ").lower()}
    for k in ("name_de", "latin", "name_en"):
        if a.get(k):
            out.add(str(a[k]).lower())
            out.add(str(a[k]).lower().split()[0])
    return {n for n in out if len(n) >= 4}


def tag_lines(timeline, animals):
    """[(animals named, kinds wanted)] per line; the text model, else the names found in the line."""
    from core import gemini, textllm
    keys = {a["key"] for a in animals}
    found = []
    for l in timeline:
        low = l["text"].lower()
        found.append({a["key"] for a in animals if any(n in low for n in _names(a))})
    out = [{"animals": set(f), "want": [], "names": []} for f in found]
    desc = ", ".join(f"{a['key']} ({a.get('name_de') or a.get('latin')}, {'ausgestorben' if a.get('extinct') else 'lebt heute'})"
                     for a in animals)
    for k in range(0, len(timeline), 25):
        rows = "\n".join(f"{i}. {timeline[i]['text'][:300]}" for i in range(k, min(k + 25, len(timeline))))
        try:
            ans = textllm.ask(TAG.format(animals=desc, rows=rows), step="real-tag", temperature=0.1, max_tokens=2500)
        except Exception:                                                # noqa: BLE001
            continue
        for r in gemini.pick(ans, "lines"):
            try:
                i = int(r.get("i"))
            except (TypeError, ValueError):
                continue
            if not k <= i < min(k + 25, len(timeline)):
                continue
            named = {x for x in (r.get("animals") or []) if x in keys}
            out[i]["animals"] = named | found[i]
            out[i]["want"] = [w for w in (r.get("want") or []) if w in WANT_ALL]
            out[i]["names"] = [str(x) for x in (r.get("names") or []) if x][:6]
    return out


def _fits(item, tag, extinct, relatives_of):
    """0 = no; else how well the picture suits the line (higher is better)."""
    d = item["desc"]
    animal = str(d.get("animal") or "none")
    cat = d.get("category")
    named = tag["animals"]
    if animal.startswith("other:"):
        return 0
    if named:
        if animal != "none" and animal not in named:
            return 0
    elif animal != "none" and animal not in extinct:
        return 0                     # a living relative only where the line speaks of it
    score = 1.0
    if animal != "none" and animal in named:
        score += 3.0
    want = tag["want"] or (["archive_photo", "archive_illustration", "fossil", "skeleton"] if named & extinct
                           else ["archive_photo", "archive_illustration", "excavation", "museum"])
    if cat in want:
        score += 2.0 - 0.4 * want.index(cat)
    elif animal == "none" and cat not in ("landscape", "excavation", "preparation", "museum", "palaeontologist")             and cat not in ARCHIVE:
        return 0
    # the history of the find (owner 15.11): its archive pictures first, a person's portrait where the line
    # names that person, a place's old photograph where it names the place
    text = f"{item.get('title') or ''} {item.get('tags') or ''} {item.get('query') or ''}".lower()
    hits = [n for n in tag.get("names") or [] if len(n) >= 4 and _name_in(n, text)]
    if hits:
        score += 3.0
    elif cat == "portrait":
        return 0                     # a face nobody names here is a stranger
    if cat in ARCHIVE:
        score += 1.5
    if cat in FILL_ONLY:
        score -= 1.2
    score += 0.3 * (int(d.get("quality") or 3) - 3)
    if item["kind"] == "video":
        score += 0.6
    return score


def _name_in(name, text):
    """A person's surname, a place or a year of the line in the picture's title or description."""
    parts = [p for p in re.split(r"[\s,.;:()]+", name.lower()) if len(p) >= 4]
    if not parts:
        return False
    surname = parts[-1]
    return surname in text or name.lower() in text


def runs_over(shots, limit):
    """Start times of runs of more than `limit` photographs (not archive pictures) in a row."""
    out, run, first = [], 0, None
    for s in shots:
        if s.get("kind") == "photo" and s.get("category") not in ARCHIVE:
            run += 1
            first = s["start"] if run == 1 else first
            if run == limit + 1:
                out.append(round(first))
        else:
            run = 0
    return out


def build(slug, timeline, narration_s, parts=20):
    ep = ROOT / "episodes" / slug
    pool = json.loads((ep / "pool.json").read_text(encoding="utf-8"))
    animals = pool["animals"]
    extinct = {a["key"] for a in animals if a.get("extinct")}
    relatives_of = {a["key"]: a.get("relative_of") for a in animals if not a.get("extinct")}
    rejects = set(json.loads((ep / "qa_rejects.json").read_text(encoding="utf-8"))) \
        if (ep / "qa_rejects.json").exists() else set()
    from pipeline import realpool
    elsewhere = realpool.used_elsewhere(slug)          # owner 15.11: a picture once in the whole channel
    items = [it for it in pool["items"] if it.get("usable") and it["id"] not in rejects and it["id"] not in elsewhere]
    tags = tag_lines(timeline, animals)

    # ---- slots: each line's time cut into shots
    slots = []
    end_all = narration_s + OUTRO_S
    for i, l in enumerate(timeline):
        a = 0.0 if i == 0 else max(0.0, l["start"] - LEAD_S)
        b = end_all if i == len(timeline) - 1 else max(a + 0.5, timeline[i + 1]["start"] - LEAD_S)
        n = max(1, round((b - a) / SHOT_S))
        while (b - a) / n > SHOT_MAX_S:
            n += 1
        for k in range(n):
            slots.append({"line": i, "start": a + (b - a) * k / n, "end": a + (b - a) * (k + 1) / n})

    # ---- most constrained first: pictures to slots, each picture once
    cand = []
    for s in slots:
        tag = tags[s["line"]]
        cand.append({it["id"]: _fits(it, tag, extinct, relatives_of) for it in items})
        cand[-1] = {k: v for k, v in cand[-1].items() if v > 0}
    by_id = {it["id"]: it for it in items}
    _cand_line = {}

    def cand_of(line):
        if line not in _cand_line:
            _cand_line[line] = {it["id"]: v for it in items
                                for v in [_fits(it, tags[line], extinct, relatives_of)] if v > 0}
        return _cand_line[line]
    used, chosen = set(), [None] * len(slots)
    still_s, video_s, fill_s = 0.0, 0.0, 0.0
    order = list(range(len(slots)))
    while order:
        order.sort(key=lambda j: (len([c for c in cand[j] if c not in used]), slots[j]["start"]))
        j = order.pop(0)
        s = slots[j]
        need = s["end"] - s["start"]
        best, best_sc = None, 0.0
        for cid, sc in cand[j].items():
            if cid in used:
                continue
            it = by_id[cid]
            if it["desc"].get("category") in FILL_ONLY and fill_s + need > FILL_SHARE * end_all:
                sc -= 2.0                     # past a quarter of the film only when nothing else fits
            if it["kind"] == "video":
                avail = float(it.get("stretch_s") or it.get("duration") or 0)
                if avail < need * 0.6:
                    sc -= 1.0                 # it would crawl at under 0.6x
            if sc > best_sc:
                best, best_sc = cid, sc
        if best:
            used.add(best)
            chosen[j] = best
            if by_id[best]["kind"] == "still":
                still_s += need
            else:
                video_s += need
            if by_id[best]["desc"].get("category") in FILL_ONLY:
                fill_s += need

    keys = {a["key"] for a in animals}

    def dress(shot, it):
        """The picture's own fields on a shot (the repair below may give a shot another picture)."""
        for k in ("clip", "image", "still_url"):
            shot.pop(k, None)
        d = it["desc"]
        shot.update({"need": it["id"], "pool_id": it["id"],
                     "species": d.get("animal") if str(d.get("animal")) in keys else None,
                     "category": d.get("category"), "what": d.get("what"), "license": it["license"],
                     "author": it.get("author"), "page": it.get("page"), "source": it["source"]})
        if it["kind"] == "video":
            dur = float(it.get("duration") or 0)
            at = (float(it["at_s"]) / dur) if it.get("at_s") is not None and dur else 0.08
            shot.update({"kind": "real", "clip": {"id": it["id"], "url": it["url"], "width": it.get("width"),
                                                  "height": it.get("height"), "duration": dur,
                                                  "source": it["source"], "at": round(at, 4), "speed": None}})
        else:
            shot.update({"kind": "photo", "image": it["id"], "still_url": it.get("full_url") or it["url"]})

    def make(line, start, end, it):
        shot = {"line": line, "start": round(start, 3), "end": round(end, 3),
                "text": timeline[line]["text"], "section": timeline[line].get("section") or "story"}
        dress(shot, it)
        return shot

    def still(x):
        return x.get("kind") == "photo"

    # ---- shots; an empty slot is taken over by a neighbour - a film may run on, a photograph only to 6 s
    shots = []
    for j, s in enumerate(slots):
        if chosen[j] is None:
            prev = shots[-1] if shots else None
            if prev and prev["line"] == s["line"] and s["end"] - prev["start"] <= (
                    STILL_MAX_S + 0.01 if still(prev) else HOLD_MAX_S):
                prev["end"] = s["end"]
                continue
            nxt = next((k for k in range(j + 1, len(slots)) if slots[k]["line"] == s["line"] and chosen[k]), None)
            if nxt is not None and slots[nxt]["end"] - s["start"] <= (
                    HOLD_MAX_S if by_id[chosen[nxt]]["kind"] == "video" else STILL_MAX_S + 0.01):
                slots[nxt]["start"] = s["start"]
                continue
            if prev:
                prev["end"] = s["end"]
                prev["held"] = True
                continue
            shots.append({"line": s["line"], "start": s["start"], "end": s["end"], "empty": True})
            continue
        shots.append(make(s["line"], s["start"], slots[j]["end"], by_id[chosen[j]]))

    # ---- owner 15.11: no more than 3 photographs in a row; the 4th gets a film or an archive picture
    # from what is left, else it trades pictures with a film shot elsewhere that fits both lines
    def plain(x):
        return still(x) and x.get("category") not in ARCHIVE

    def breaker(it):
        return it["kind"] == "video" or it["desc"].get("category") in ARCHIVE

    run = 0
    for k, sh in enumerate(shots):
        if sh.get("empty") or not plain(sh):
            run = 0
            continue
        run += 1
        if run <= PLAIN_RUN_MAX:
            continue
        tag = tags[sh["line"]]
        spare = sorted(((c, v) for c, v in cand_of(sh["line"]).items() if c not in used and breaker(by_id[c])),
                       key=lambda kv: -kv[1])
        if spare:
            used.discard(sh["pool_id"])
            used.add(spare[0][0])
            dress(sh, by_id[spare[0][0]])
            run = 0
            continue
        for m, other in enumerate(shots):
            if m == k or other.get("empty") or not breaker(by_id[other["pool_id"]]):
                continue
            mine, theirs = by_id[sh["pool_id"]], by_id[other["pool_id"]]
            if _fits(theirs, tag, extinct, relatives_of) <= 0 or \
                    _fits(mine, tags[other["line"]], extinct, relatives_of) <= 0:
                continue
            if other["end"] - other["start"] > STILL_MAX_S + 0.01:
                continue
            near = [shots[x] for x in range(max(0, m - 3), min(len(shots), m + 4)) if x != m]
            if sum(1 for x in near if plain(x)) >= PLAIN_RUN_MAX:
                continue            # the photograph would make a run of 4 where it lands
            dress(sh, theirs)
            dress(other, mine)
            run = 0
            break
    if shots and shots[0].get("empty"):
        raise RuntimeError("the film's first line has no picture")
    shots = [s for s in shots if not s.get("empty")]
    shots[0]["start"] = 0.0
    for a_, b_ in zip(shots, shots[1:]):            # no gaps
        a_["end"] = b_["start"]
    shots[-1]["end"] = round(end_all, 3)

    # ---- transitions, moves, atmosphere
    for n, s in enumerate(shots):
        first_of_section = n == 0 or shots[n - 1]["section"] != s["section"]
        s["transition"] = "cut" if first_of_section else "fade"
        s["move"] = MOVES[n % len(MOVES)]
        s["duration"] = round(s["end"] - s["start"], 3)
        what = f"{s.get('what') or ''} {s.get('category') or ''}"
        if s["category"] in ("museum", "preparation", "fossil", "skeleton", "palaeoart", "portrait",
                             "archive_illustration"):
            s["amb"] = "room"
        elif SEA.search(what):
            s["amb"] = "underwater" if re.search(r"underwater|reef|swim", what, re.I) else "ocean"
        elif s["category"] in ("excavation", "palaeontologist", "landscape", "archive_photo", "archive_film"):
            s["amb"] = "wind"
        else:
            s["amb"] = None
        s["amb_line"] = s["amb"]
    sections = []
    for s in shots:
        if not sections or sections[-1]["name"] != s["section"]:
            sections.append({"name": s["section"], "start": s["start"], "end": s["end"]})
        sections[-1]["end"] = s["end"]
    ids = [s["pool_id"] for s in shots]
    held = [round(s["start"]) for s in shots if s.get("held")]
    from pipeline import docplan
    checks = {"format": "real", "shots": len(shots), "unique_pictures": len(set(ids)), "repeats": len(ids) - len(set(ids)),
              "video_share": round(sum(s["duration"] for s in shots if s["kind"] == "real") / end_all, 3),
              "still_share": round(sum(s["duration"] for s in shots if s["kind"] == "photo") / end_all, 3),
              "mean_shot_s": round(end_all / max(1, len(shots)), 2), "held_on_at": held[:20],
              "archive_share": round(sum(s["duration"] for s in shots if s.get("category") in ARCHIVE) / end_all, 3),
              "fill_share": round(sum(s["duration"] for s in shots if s.get("category") in FILL_ONLY) / end_all, 3),
              "longest_shot_s": round(max(s["duration"] for s in shots), 1),
              "stills_over_6s": [round(s["start"]) for s in shots if s["kind"] == "photo"
                                 and s["duration"] > STILL_MAX_S + 0.05][:20],
              "photo_runs_over_3": runs_over(shots, PLAIN_RUN_MAX)}
    assert checks["repeats"] == 0, "a picture used twice"
    plan = {"slug": slug, "format": "real", "narration_s": round(narration_s, 3), "total_s": round(end_all, 3),
            "xfade_s": XFADE_S, "shots": shots, "sections": sections, "checks": checks,
            "parts": docplan.split(shots, parts)}

    # ---- what the QA and the credits read
    spec = {"lines": [{"n": i, "text": l["text"], "beats": [{"species": k} for k in sorted(tags[i]["animals"])],
                       "want": tags[i]["want"]} for i, l in enumerate(timeline)], "format": "real"}
    (ep / "shots.json").write_text(json.dumps(spec, indent=1, ensure_ascii=False), encoding="utf-8")
    used_items = {s["pool_id"]: by_id[s["pool_id"]] for s in shots}
    (ep / "footage.json").write_text(json.dumps(used_items, indent=1, ensure_ascii=False), encoding="utf-8")
    return plan


def thumbs_plan(slug, title):
    """Three thumbnail bases from the pool: the extinct animal's paintings and fossils, big and sharp."""
    ep = ROOT / "episodes" / slug
    pool = json.loads((ep / "pool.json").read_text(encoding="utf-8"))
    extinct = {a["key"] for a in pool["animals"] if a.get("extinct")}
    stills = [it for it in pool["items"] if it.get("usable") and it["kind"] == "still"
              and str(it["desc"].get("animal")) in extinct]
    rank = {"palaeoart": 0, "skeleton": 1, "fossil": 2, "museum": 3}
    stills.sort(key=lambda it: (rank.get(it["desc"]["category"], 9), -int(it["desc"].get("quality") or 0),
                                -(it.get("width") or 0)))
    return stills[:6]
