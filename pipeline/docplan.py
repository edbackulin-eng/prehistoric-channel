"""The edit plan: which picture is on screen when, cut to the narration.

Inputs: the narration timeline (core/narrate.py), shots.json (what each line
shows), footage.json (the chosen real clips) and recon_picks.json (which seed
of each reconstruction). Output: plan.json, a flat list of shots with times,
transitions, motion and overlays, plus the parts the render is split into.

Rules
  - a shot lasts 4-7 s (3.5-7.5 s at the edges); a line too long for one shot
    gets a second one from a different source - another clip, the other seed -
    so no picture is ever shown twice;
  - pictures lead the words by a quarter of a second, as an editor cuts;
  - transitions are 0.6 s dissolves, except hard cuts where the story turns
    (and the render parts begin only on those cuts, so the parts join with
    -c copy and no dissolve is ever split between two machines).
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LEAD_S = 0.25
MIN_S, MAX_S = 3.5, 7.5
XFADE_S = 0.6
OUTRO_S = 3.0
# Lines that open a new movement: a hard cut, and a place a render part may start.
CUT_BEFORE = ("But some animals", "They crawl up", "East London", "It is a coelacanth",
              "Then, after the Cretaceous", "Yet fishermen", "And slowly, the living",
              "So how does")
MOVES = ["push_in", "truck_right", "pull_out", "truck_left", "crane_up", "push_in_left"]


def _match(lines, at):
    for i, l in enumerate(lines):
        if l["text"].startswith(at):
            return i
    raise KeyError(f"no narration line starts with {at!r}")


def build(slug, timeline, narration_s, parts=6):
    ep = ROOT / "episodes" / slug
    spec = json.loads((ep / "shots.json").read_text(encoding="utf-8"))["lines"]
    footage = json.loads((ep / "footage.json").read_text(encoding="utf-8"))
    picks_path = ep / "recon_picks.json"
    picks = json.loads(picks_path.read_text(encoding="utf-8")) if picks_path.exists() else {}

    starts = [max(0.0, l["start"] - LEAD_S) for l in timeline]
    starts[0] = 0.0
    end_all = narration_s + OUTRO_S
    segs = []
    for k, entry in enumerate(spec):
        i = _match(timeline, entry["at"])
        j = _match(timeline, spec[k + 1]["at"]) if k + 1 < len(spec) else None
        a, b = starts[i], (starts[j] if j is not None else end_all)
        segs.append((entry, i, a, b))

    shots = []
    for entry, li, a, b in segs:
        beats = entry["beats"]
        span = b - a
        # one beat on a long line: a second shot from another source
        if len(beats) == 1 and span > MAX_S and beats[0]["kind"] != "anim":
            beats = [beats[0], {**beats[0], "second": True, "label": None, "title": None}]
        n = len(beats)
        for bi, beat in enumerate(beats):
            s = a + span * bi / n
            e = a + span * (bi + 1) / n
            shot = {"line": li, "text": timeline[li]["text"], "start": round(s, 3), "end": round(e, 3),
                    "amb": entry.get("amb"), **{k: v for k, v in beat.items() if v is not None}}
            if beat["kind"] == "real":
                clip = footage[beat["need"]]
                if beat.get("second"):
                    alt = (clip.get("alternates") or [None])[0]
                    if alt is None:
                        raise RuntimeError(f"{beat['need']}: a second shot is needed and there is no alternate")
                    clip = {**clip, **alt}
                shot["clip"] = {k: clip.get(k) for k in ("id", "url", "width", "height", "duration",
                                                         "source", "at", "speed")}
            elif beat["kind"] in ("recon", "photo"):
                rid = beat.get("id") or beat["need"]
                seed = picks.get(rid, 0)
                if beat.get("second"):
                    seed = 1 - seed if beat["kind"] == "recon" else seed
                shot["image"] = rid if beat["kind"] == "photo" else f"{rid}__s{seed}"
            shots.append(shot)

    # merge shots that came out too short into their neighbour
    merged = []
    for s in shots:
        if merged and s["end"] - s["start"] < MIN_S and not s.get("label") and not s.get("title"):
            merged[-1]["end"] = s["end"]
            continue
        merged.append(s)
    shots = merged

    for n, s in enumerate(shots):
        first_of_line = n == 0 or shots[n - 1]["line"] != s["line"]
        turn = any(s["text"].startswith(c) for c in CUT_BEFORE)
        fast = timeline[s["line"]]["kind"] == "fast"
        s["transition"] = "cut" if n == 0 or (first_of_line and (turn or fast)) else "fade"
        s["move"] = MOVES[n % len(MOVES)]
        s["duration"] = round(s["end"] - s["start"], 3)
    return {"slug": slug, "narration_s": round(narration_s, 3), "total_s": round(end_all, 3),
            "xfade_s": XFADE_S, "shots": shots, "parts": split(shots, parts)}


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
