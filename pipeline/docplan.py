"""The edit plan: which picture is on screen when, cut to the narration.

Inputs: the narration timeline (core/narrate_de.py), shots.json (what each
line shows), footage.json (the real clips and their alternates) and the
species check's picks (pipeline/speciescheck.py: for every beat, the
candidates that are the right animal and fit the line, in order). Output:
plan.json, a flat list of shots with times, transitions, motion and overlays,
plus the parts the render is split into.

Rules
  - a shot lasts 4-7 s (3.5-7.5 s at the edges); a line too long for one shot
    gets a second one from the beat's next passing candidate - another clip,
    photo or seed - so no picture is ever shown twice;
  - a beat none of whose candidates passed the species check is dropped, and
    the shot before it runs on over its words;
  - pictures lead the words by a quarter of a second, as an editor cuts;
  - transitions are 0.6 s dissolves, except hard cuts where the story turns
    (and the render parts begin only on those cuts, so the parts join with
    -c copy and no dissolve is ever split between two machines).
"""
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LEAD_S = 0.25
MIN_S, MAX_S = 3.5, 7.5
XFADE_S = 0.6
OUTRO_S = 3.0
MOVES = ["push_in", "truck_right", "pull_out", "truck_left", "crane_up", "push_in_left"]


HOOK_S = (1.5, 3.0)        # the hook's shots: short, and only moving pictures
FIRST_MINUTE_MOVING = 0.70
MUSEUM_MAX = 3


def _match(lines, at, n=None):
    if n is not None and n < len(lines):
        return n
    for i, l in enumerate(lines):
        if l["text"].startswith(at):
            return i
    raise KeyError(f"no narration line starts with {at!r}")


def _clips(footage):
    """Every clip by id: the chosen ones and their alternates."""
    out = {}
    for need, c in footage.items():
        out.setdefault(c["id"], c)
        for alt in c.get("alternates") or []:
            out.setdefault(alt["id"], {**c, **alt})
    return out


def build(slug, timeline, narration_s, parts=6, picks=None, animated=None):
    from pipeline import speciescheck
    ep = ROOT / "episodes" / slug
    spec = json.loads((ep / "shots.json").read_text(encoding="utf-8"))["lines"]
    footage = json.loads((ep / "footage.json").read_text(encoding="utf-8"))
    clips = _clips(footage)
    picks = picks or {}
    museum = set(picks.get("_museum", []))
    animated = animated or {}                      # FLUX frame id -> LTX clip {path, w, h, duration}
    used, shown_museum = set(), []

    def take(beat):
        """The beat's next passing candidate not shown yet, or None. Museum
        specimens and display cases: MUSEUM_MAX an episode."""
        bk = speciescheck.beat_key(beat)
        order = picks.get(bk)
        if order is None:                          # no check ran: the candidates in their own order
            order = [c[0] for c in speciescheck.candidates(beat, footage, Path("."))]
        for cid in order:
            if cid in used or (cid in museum and len(shown_museum) >= MUSEUM_MAX):
                continue
            used.add(cid)
            if cid in museum:
                shown_museum.append(cid)
            return cid
        return None

    starts = [max(0.0, l["start"] - LEAD_S) for l in timeline]
    starts[0] = 0.0
    end_all = narration_s + OUTRO_S
    segs = []
    for k, entry in enumerate(spec):
        i = _match(timeline, entry["at"], entry.get("n"))
        j = _match(timeline, spec[k + 1]["at"], spec[k + 1].get("n")) if k + 1 < len(spec) else None
        a, b = starts[i], (starts[j] if j is not None else end_all)
        segs.append((entry, i, a, b))

    shots = []
    for entry, li, a, b in segs:
        hook = (entry.get("section") or timeline[li].get("section")) == "hook"
        span = b - a
        beats = list(entry["beats"])
        base = {"line": li, "text": timeline[li]["text"], "amb": entry.get("amb"),
                "cut_here": bool(entry.get("cut")), "section": timeline[li].get("section", "story")}
        if not beats:
            shots.append({**base, "kind": "anim", "start": round(a, 3), "end": round(b, 3), "drop": True})
            continue
        lo, hi = HOOK_S if hook else (MIN_S, MAX_S)
        target = 2.4 if hook else 5.6
        # as many shots as the line needs, each from the beat's next passing
        # candidate; a beat that has run out of candidates but has shown a clip
        # gives another moment of the same clip (no frame is shown twice)
        n = max(1, min(max(len(beats), math.ceil(span / target)), max(1, int(span // lo))))
        last_clip = {}
        for k in range(n):
            beat = beats[k % len(beats)]
            s0, e0 = a + span * k / n, a + span * (k + 1) / n
            shot = {**base, "start": round(s0, 3), "end": round(e0, 3),
                    **{kk: v for kk, v in beat.items() if v is not None and kk not in ("alts", "allow")}}
            if k:                                  # a caption or title belongs to the line's first shot
                shot.pop("label", None)
                shot.pop("title", None)
            if beat["kind"] != "anim":
                cid = take(beat)
                bk = speciescheck.beat_key(beat)
                if cid is None and beat["kind"] == "real" and bk in last_clip:
                    prev = last_clip[bk]
                    cid, shot["at_override"] = prev["id"], round((0.2 + 0.31 * (k + 1)) % 0.85, 2)
                if cid is None:
                    shot["drop"] = True
                elif beat["kind"] == "real":
                    clip = dict(clips[cid])
                    if "at_override" in shot:
                        clip["at"] = shot.pop("at_override")
                    shot["clip"] = {kk: clip.get(kk) for kk in ("id", "url", "width", "height", "duration",
                                                                "source", "at", "speed")}
                    last_clip[bk] = clip
                elif cid in animated:              # an LTX-Video clip of this FLUX frame: a moving shot
                    c = animated[cid]
                    shot.update({"kind": "real", "need": f"ltx-{cid}", "image": cid,
                                 "clip": {"id": f"ltx-{cid}", "url": "file:" + c["path"], "width": c["w"],
                                          "height": c["h"], "duration": c["duration"], "source": "LTX-Video",
                                          "at": 0.0, "speed": None}})
                else:
                    shot["image"] = cid
            shots.append(shot)

    # dropped beats and shots that came out too short go into their neighbour
    merged = []
    for s in shots:
        if s.get("drop") and not merged:
            continue
        short = HOOK_S[0] if s.get("section") == "hook" else MIN_S
        if merged and (s.get("drop") or (s["end"] - s["start"] < short and not s.get("label")
                                         and not s.get("title"))):
            merged[-1]["end"] = s["end"]
            continue
        merged.append(s)
    if shots and shots[0].get("drop") and merged:          # a dropped opening: the next shot starts at 0
        merged[0]["start"] = 0.0
    shots = merged

    for n, s in enumerate(shots):
        first_of_line = n == 0 or shots[n - 1]["line"] != s["line"]
        turn = s.get("cut_here", False)
        fast = timeline[s["line"]]["kind"] == "fast"
        s["transition"] = "cut" if n == 0 or (first_of_line and (turn or fast)) else "fade"
        s["move"] = MOVES[n % len(MOVES)]
        s["duration"] = round(s["end"] - s["start"], 3)
    sections = []
    for s in shots:
        if not sections or sections[-1]["name"] != s.get("section", "story"):
            sections.append({"name": s.get("section", "story"), "start": s["start"], "end": s["end"]})
        sections[-1]["end"] = s["end"]
    # the first minute: share of moving pictures (real footage, LTX clips, maps)
    first = [(min(x["end"], 60) - x["start"], x["kind"] in ("real", "anim")) for x in shots if x["start"] < 60]
    moving = sum(d for d, m in first if m) / max(1e-6, sum(d for d, _ in first))
    hook = [x for x in shots if x.get("section") == "hook"]
    checks = {"first_minute_moving_share": round(moving, 2),
              "first_minute_ok": moving >= FIRST_MINUTE_MOVING,
              "hook_shots": len(hook), "hook_still_shots": sum(1 for x in hook if x["kind"] not in ("real", "anim")),
              "museum_shots": len(shown_museum), "animated_clips_used": sum(1 for x in shots
                                                                           if str(x.get("need", "")).startswith("ltx-"))}
    return {"slug": slug, "narration_s": round(narration_s, 3), "total_s": round(end_all, 3),
            "xfade_s": XFADE_S, "shots": shots, "sections": sections, "checks": checks,
            "parts": split(shots, parts)}


def split(shots, parts):
    """[(first, end)] shot ranges of similar length, each starting on a cut."""
    total = shots[-1]["end"]
    bounds = [0]
    for k in range(1, parts):
        target = total * k / parts
        cands = [i for i, s in enumerate(shots) if i > bounds[-1] and s["transition"] == "cut"]
        if not cands:
            break
        best = min(cands, key=lambda i: abs(shots[i]["start"] - target))
        if best not in bounds:
            bounds.append(best)
    bounds.append(len(shots))
    return [[a, b] for a, b in zip(bounds, bounds[1:]) if b > a]
