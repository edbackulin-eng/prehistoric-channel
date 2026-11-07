"""Fragments: more cuts from the same material, never the same piece twice (owner 04.11).

A plan built line by line runs short of candidates: a beat with nothing left was
dropped and the shot before it ran on, so lebende-fossilien came out at 11 s a
shot with 28 % landscapes and 8 minutes without a wow moment. This pass works on
the finished shot list:

  1. split   every shot longer than the cut allows (5 s; 3 s in the first 30 s)
             is cut into pieces of about 4 s (2.4 s). Each later piece shows
             another fragment of the same source - a later stretch of the clip
             (real footage, an LTX clip), another framing of the still (CROPS) -
             or, when the source has none left, the line's animal from the pool;
  2. landscape  past 15 % of the film, landscape shots give way to the episode's
             animals (the line's species first);
  3. first minute  still pictures in the first 60 s give way to moving animals
             until 72 % of it moves (the rule is 70 %);
  4. wow     where 36 s have gone by without a wow moment, the shot there becomes
             one: an LTX fragment, real footage of an animal, or a wow still.

A fragment is a piece of a source: the clip from a point on ("<id>@0.40"), or a
still in one framing ("<image>#2"). The repeat rule (twice at most, never
within 3 minutes) applies to fragments - a piece of film is never shown again
inside 3 minutes and never three times; another stretch of the same clip is
another piece. automation/qa.py reads the same keys.

Owner 07.11: a fragment or a crop of a clip is that clip again. The repeat rule now counts
the SOURCE (source_of: the clip, or the still whatever its framing): at most twice in the
film, never twice within 3 minutes, whichever stretch or crop is shown. A piece the rule
leaves nothing for takes another source of the pool; with none left, its shot runs on.
"""
import math

CROPS = [None, (0.0, 0.125, 0.75, 0.875), (0.25, 0.125, 1.0, 0.875), (0.125, 0.0, 0.875, 0.75),
         (0.125, 0.25, 0.875, 1.0)]
SPLIT_MAX, SPLIT_TARGET, SPLIT_MIN = 5.5, 3.4, 3.0   # an LTX clip (6.7 s) gives two pieces
HOOK_MAX, HOOK_TARGET = 3.0, 2.4
FIRST_MINUTE_AIM = 0.72
LANDSCAPE_AIM = 0.14
CLIP_KEYS = ("clip", "need", "image", "crop", "topic", "dark", "landscape", "anim")
LTX_SHOT_S = 6.6            # docplan.LTX_SHOT_S, and the 0.1 s a dissolve takes
ANIM_FIRST_S = 6.0          # a map or timeline keeps its first 6 s; the rest of a long one shows the animals


def key(s):
    """The fragment a shot shows (automation/qa.py clip_key)."""
    c = s.get("clip") or {}
    if c.get("id"):
        at = c.get("at")
        return f"{c['id']}@{float(at):.2f}" if at not in (None, 0, 0.0) else c["id"]
    if s.get("image"):
        return f"{s['image']}#{s['crop']}" if s.get("crop") else s["image"]
    return None


def source_of(s):
    """The source a shot shows, whatever its crop or stretch (automation/qa_view.source_of)."""
    c = s.get("clip") or {}
    if isinstance(c, dict) and c.get("id"):
        return "clip:" + str(c["id"])
    if s.get("image"):
        return "img:" + str(s["image"]).split("#")[0]
    return None


def interval(s):
    """(clip id, source from, source to) a shot shows, or None (for the repeat rule)."""
    c = s.get("clip") or {}
    d = float(c.get("duration") or 0)
    if not c.get("id") or d <= 0:
        return None
    a = float(c.get("at") or 0.0) * d
    return c["id"], a, min(d, a + (s["end"] - s["start"]))


class Pool:
    """Every fragment the episode has, and when each went on screen.

    Clips are tracked as the stretches of source shown and when (a new piece is
    a stretch overlapping nothing shown within 3 minutes, and nothing shown
    twice); stills by framing."""

    def __init__(self, shots, animated, clips, subjects, picks, beat_key, repeat_max, repeat_gap):
        self.shown, self.spans = {}, {}
        self.uses = {}                     # source -> times it went on screen (owner 07.11)
        self.repeat_max, self.repeat_gap = repeat_max, repeat_gap
        for s in shots:
            self.note(s)
        self.animated, self.clips = animated, clips
        # the episode's animals: (beat, candidate id), LTX-animated ones first
        self.subjects = []
        for bt in subjects:
            for cid in picks.get(beat_key(bt)) or []:
                self.subjects.append((bt, cid))
        self.subjects.sort(key=lambda r: 0 if r[1] in animated else 1 if r[0]["kind"] == "real" else 2)

    def src_free(self, src, at):
        t = self.uses.get(src, [])
        return src is None or (len(t) < self.repeat_max and all(abs(at - x) >= self.repeat_gap for x in t))

    def use(self, src, at):
        if src:
            self.uses.setdefault(src, []).append(at)

    def note(self, s):
        self.use(source_of(s), s["start"])
        iv = interval(s)
        if iv:
            self.spans.setdefault(iv[0], []).append((iv[1], iv[2], s["start"]))
        elif key(s):
            self.shown.setdefault(key(s), []).append(s["start"])

    def forget(self, s):
        src = source_of(s)
        if src in self.uses:
            self.uses[src] = [x for x in self.uses[src] if x != s["start"]]
        iv = interval(s)
        if iv:
            self.spans[iv[0]] = [x for x in self.spans.get(iv[0], []) if x[2] != s["start"]]
        elif key(s) in self.shown:
            self.shown[key(s)] = [x for x in self.shown[key(s)] if x != s["start"]]

    def free(self, k, at):
        t = self.shown.get(k, [])
        return len(t) < self.repeat_max and all(abs(at - x) >= self.repeat_gap for x in t)

    def span_free(self, cid, a, b, at):
        over = [x for x in self.spans.get(cid, []) if x[0] < b - 0.05 and a < x[1] - 0.05]
        return len(over) < self.repeat_max and all(abs(at - x[2]) >= self.repeat_gap for x in over)

    def piece(self, clip, length, at, after=0.0):
        """The `at` fraction of a stretch of `length` s of the clip free to show at `at`, or None:
        the first free one from `after` (a fraction) on, then from the start."""
        d = float(clip.get("duration") or 0)
        if d <= 0 or length <= 0 or length > d or not self.src_free("clip:" + str(clip["id"]), at):
            return None
        starts = [round(x * 0.25, 2) for x in range(int((d - length) / 0.25) + 1)]
        starts = [x for x in starts if x >= after * d] + [x for x in starts if x < after * d]
        for x in starts:
            if self.span_free(clip["id"], x, x + length, at):
                self.spans.setdefault(clip["id"], []).append((x, x + length, at))
                self.use("clip:" + str(clip["id"]), at)
                return round(x / d, 3)
        return None

    def same_source(self, s, at, length):
        """Another fragment of the shot's own source, as shot fields, or None."""
        c = s.get("clip") or {}
        if c.get("id"):
            a = self.piece(c, length, at, float(c.get("at") or 0.0))
            return {"clip": {**c, "at": a}} if a is not None else None
        if s.get("image") and s.get("kind") == "recon":
            # another framing of the same still is the same source (owner 07.11)
            src = "img:" + str(s["image"]).split("#")[0]
            if not self.src_free(src, at):
                return None
            order = list(range(len(CROPS)))
            k0 = int(s.get("crop") or 0)
            for i in order[k0 + 1:] + order[:k0 + 1]:
                k = f"{s['image']}#{i}" if i else s["image"]
                if self.free(k, at):
                    self.shown.setdefault(k, []).append(at)
                    self.use(src, at)
                    return {"image": s["image"], "crop": i or None}
        return None

    def subject(self, at, length, species=None, moving=False, wow_ids=(), wow=False, still_first=False,
                any_animal=False):
        """A fragment of one of the episode's animals, as shot fields, or None: the line's species
        while it has any piece left; another animal only with any_animal (owner 07.11: an animal
        the line does not name is never shown). `moving`: an LTX clip or real footage only;
        `wow`: a wow moment only (an LTX clip, real footage of an animal, a wow still)."""
        if not species and not any_animal:
            return None
        for pass_species in ((species,) if species else (None,)):
            got = self._subject(at, length, pass_species, moving, wow_ids, wow, still_first)
            if got:
                return got
        return None

    def _subject(self, at, length, species, moving, wow_ids, wow, still_first=False):
        rows = self.subjects
        if still_first:
            # an ordinary piece: another framing of a still first, so the moving pictures (few:
            # LTX clips, real footage of the animals) stay for the first minute and the wow moments
            rows = sorted(rows, key=lambda r: 0 if r[0]["kind"] == "recon" and r[1] not in self.animated else 1)
        for bt, cid in rows:
            if species and bt.get("species") != species:
                continue
            base = {kk: v for kk, v in bt.items() if v is not None and kk not in ("alts", "allow", "label", "title")}
            if cid in self.animated:
                c = self.animated[cid]
                clip = {"id": f"ltx-{cid}", "url": "file:" + c["path"], "width": c["w"], "height": c["h"],
                        "duration": c["duration"], "source": "LTX-Video", "at": 0.0, "speed": None}
                a = self.piece(clip, length, at)
                if a is not None:
                    return {**base, "kind": "real", "need": f"ltx-{cid}", "image": cid,
                            "clip": {**clip, "at": a}, "wow": True}
                continue
            if bt["kind"] == "real" and cid in self.clips:
                if wow and not bt.get("species"):
                    continue
                c = self.clips[cid]
                clip = {kk: c.get(kk) for kk in ("id", "url", "width", "height", "duration", "source", "at", "speed")}
                a = self.piece(clip, length, at)
                if a is not None:
                    return {**base, "clip": {**clip, "at": a}}
                continue
            if moving or bt["kind"] != "recon" or (wow and bt.get("id") not in wow_ids):
                continue
            src = "img:" + str(cid).split("#")[0]
            if not self.src_free(src, at):
                continue
            for i in range(len(CROPS)):
                k = f"{cid}#{i}" if i else cid
                if self.free(k, at):
                    self.shown.setdefault(k, []).append(at)
                    self.use(src, at)
                    return {**base, "kind": "recon", "image": cid, "crop": i or None}
        return None


LINE_KEYS = ("line", "text", "amb", "cut_here", "section", "start", "end")


def _set(s, fields):
    """The shot shows another picture: its line, section and times stay its own."""
    keep = {kk: s[kk] for kk in LINE_KEYS if kk in s}
    for kk in CLIP_KEYS:
        s.pop(kk, None)
    s.update(fields)
    s.update(keep)


def refine(shots, end_all, pool, is_landscape, is_wow, wow_ids, hook_window, wow_aim, landscape_max,
           first_minute_moving, named=None):
    # 1. split the long shots into fragments
    out = []
    for s in shots:
        d = s["end"] - s["start"]
        hook = s["start"] < hook_window or s.get("section") == "hook"
        cap, target = (HOOK_MAX, HOOK_TARGET) if hook else (SPLIT_MAX, SPLIT_TARGET)
        if s.get("kind") == "anim":
            # a map that ran on over lines with no picture of their own (04.11: 44, 106 and 34 s)
            cap = max(cap, ANIM_FIRST_S)
        if str(s.get("need", "")).startswith("ltx-") and not hook:
            cap = max(cap, LTX_SHOT_S)              # an LTX clip runs its length (owner 02.11)
        if d <= cap + 0.05:
            out.append(s)
            continue
        if s.get("kind") == "anim":
            first = ANIM_FIRST_S
            n = 1 + max(1, min(int((d - first) // SPLIT_MIN), round((d - first) / target)))
            bounds = [s["start"]] + [s["start"] + first + (d - first) * k / (n - 1) for k in range(n)]
        else:
            n = max(2, math.ceil(d / target)) if hook else max(2, min(int(d // SPLIT_MIN), round(d / target)))
            bounds = [s["start"] + d * k / n for k in range(n + 1)]
        pool.forget(s)                                  # it shows only its first piece now
        for k in range(n):
            a, b = bounds[k], bounds[k + 1]
            step = b - a
            p = {**s, "start": round(a, 3), "end": round(b, 3)}
            if not k:
                pool.note(p)
            if k:
                p.pop("label", None)
                p.pop("title", None)
                same = pool.same_source(s, a, step)
                if same is not None:
                    # another fragment of the same source: everything else of the shot stays
                    # (04.11: a real clip's piece lost its "need", and the render failed on it)
                    same = {**{kk: s[kk] for kk in CLIP_KEYS if kk in s and kk not in ("clip", "crop")}, **same}
                got = same or pool.subject(a, step, species=s.get("species"), moving=hook, still_first=not hook)
                if got is None:                       # nothing left anywhere: the piece stays with its shot
                    out[-1]["end"] = p["end"]
                    continue
                _set(p, got)
            out.append(p)
    shots = out

    def share(test, upto=None):
        return sum(min(s["end"], upto or s["end"]) - s["start"] for s in shots
                   if test(s) and (upto is None or s["start"] < upto)) / max(1e-6, upto or end_all)

    # 3. the first minute moves
    moving = lambda x: x.get("kind") in ("real", "anim")                    # noqa: E731
    for s in [x for x in shots if x["start"] < 60 and not moving(x)]:
        if share(moving, 60) >= FIRST_MINUTE_AIM:
            break
        got = pool.subject(s["start"], s["end"] - s["start"], species=s.get("species"), moving=True)
        if got:
            pool.forget(s)
            _set(s, got)

    # 4. a wow moment at least every 45 s, aimed at 36 s
    last = 0.0
    for s in shots:
        if is_wow(s, wow_ids):
            s["wow"], last = True, s["start"]
            continue
        if s["start"] - last < wow_aim or s["start"] < hook_window or s.get("label") or s.get("title"):
            continue
        got = pool.subject(s["start"], s["end"] - s["start"], species=s.get("species"), wow_ids=wow_ids, wow=True)
        if got:
            pool.forget(s)
            _set(s, got)
            s["wow"], last = True, s["start"]
    # 2. landscapes down to 15 %
    for s in sorted((x for x in shots if is_landscape(x) and x["start"] >= hook_window),
                    key=lambda x: -(x["end"] - x["start"])):
        if share(is_landscape) <= min(landscape_max, LANDSCAPE_AIM):
            break
        got = None
        for sp in sorted((named or {}).get(s.get("line"), ())):      # only an animal the line names
            got = pool.subject(s["start"], s["end"] - s["start"], species=sp)
            if got:
                break
        if got:
            pool.forget(s)
            _set(s, got)

    for s in shots:
        s["wow"] = is_wow(s, wow_ids)
    return shots
