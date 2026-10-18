"""The edit: which picture is on screen when, and what moves on it.

The rhythm is decided here, by code, so it holds whatever a model answers:
  - a cut every 3-5 seconds, snapped to a gap between spoken words;
  - a new family of visual every 20-40 seconds: photo (generated stills, moving or
    annotated), screen (browser, chat, notification, cursor) and data (counter,
    timeline, redacted document) take turns, so the eye never settles;
  - a punch-in on every slow line, a riser into it and an impact on it, and a
    small shake on the last reveal.

The model is only asked to fill each beat with content of the family already
chosen for it, from the narration being spoken over it. Anything it returns that
does not fit the menu is replaced by a safe default for that family.
"""
import json
import random
import re

from core import llm
from core.logger import get_logger

log = get_logger("editplan")

CUT_MIN, CUT_MAX = 3.0, 5.0
BLOCK_MIN, BLOCK_MAX = 20.0, 40.0
FAMILIES = ("photo", "screen", "data")
TYPES = {
    "photo": ("flux", "annotate", "stock"),
    "screen": ("browser", "chat", "notification", "cursor"),
    "data": ("quote", "counter", "timeline", "censor", "map"),
}
# At most this share of all shots may be a generated monitor or screen glow. The
# first hook was mostly empty monitors; the rest of the picture is real footage,
# story objects and places, and drawn data.
MONITOR_SHARE_MAX = 0.20
SCREEN_WORDS = re.compile(
    r"\b(screen|screens|monitor|monitors|display|computer|computers|laptop|laptops|"
    r"crt|tv|television|phone|smartphone)\b", re.I)
TRANSITIONS = ("vhs", "rgb", "fade")
PLAN_BATCH = 25

PLANNER_SYSTEM = """You plan the pictures for a documentary-style YouTube video about an
internet mystery. You are given beats: a few seconds of narration each, and the visual
TYPE already chosen for that beat. Fill in that type's fields. Return JSON only.

RULES FOR GENERATED STILLS (any "prompt" field)
- Describe a photograph in one sentence: a physical object or a place FROM THIS
  STORY, in a colourful lit night (neon teal and magenta, amber street light).
  Think of what the narration is about: old books, papers taped to a telephone
  pole, a folded map, a sealed letter, a lamp post on a wet street, a library at
  night, a payphone, a stack of floppy disks. Vary the angle: close-up, overhead,
  low angle, wide establishing shot.
- Do NOT default to a computer monitor or a laptop. Screens may appear in at most
  one prompt in five.
- Paper is fine - books, letters, maps, posters - but any writing on it must be
  tiny, faded or out of focus. No calendars, newspapers, headlines, street signs,
  labels, or screens showing content. Screens are only glow.
- No real logos, no real websites or apps, no brand names, no readable text.
- No identifiable people. At most one person, seen from behind or in silhouette,
  never looking at the camera. No children.
- Nothing violent, no blood, no weapons.

RULES FOR DRAWN SCREENS AND DATA
- Text must come from the narration of that beat or be a neutral label. Do not invent
  facts, numbers or quotes the narration does not state.
- Use generic labels instead of real site names ("image board", "forum", "inbox").
- Keep it short: an address under 40 characters, a heading under 8 words, a body
  under 30 words, at most 3 chat messages of under 12 words each.

TYPES AND FIELDS
  stock:        {"prompt": "..."}    (a fallback still in case no real footage is found)
  map:          {}                   (drawn by code from the places the narration names)
  flux:         {"prompt": "..."}
  annotate:     {"prompt": "...", "circle": [x, y, r]}    (x, y, r between 0 and 1)
  browser:      {"address": "...", "heading": "...", "body": "..."}
  chat:         {"header": "...", "messages": [["who", "text"], ...]}
  notification: {"prompt": "...", "app": "...", "text": "..."}
  cursor:       {"prompt": "..."}
  counter:      {"label": "...", "end": 123, "prefix": "", "suffix": ""}   (only for a number the narration says)
  timeline:     {"events": [["when", "what"], ...]}                      (2-4 events from the narration)
  censor:       {"heading": "...", "lines": ["...", "..."], "redact": ["word", ...]}
  quote:        {"text": "the 6-14 most striking words of this beat, exactly as spoken"}

Return: {"beats": [{"i": 0, "type": "...", ...fields}, ...]}
"""


# --- rhythm -----------------------------------------------------------------

def _gaps(words):
    """Moments between spoken words, where a cut does not land mid-syllable."""
    return [(a["end"] + b["start"]) / 2 for a, b in zip(words, words[1:])]


def cut_points(words, total, seed=0):
    rng = random.Random(seed)
    gaps = _gaps(words) or [total]
    cuts, t = [0.0], 0.0
    while total - t > CUT_MAX:
        want = t + rng.uniform(3.4, 4.6)
        near = [g for g in gaps if t + CUT_MIN <= g <= t + CUT_MAX]
        t = min(near, key=lambda g: abs(g - want)) if near else t + 4.0
        cuts.append(round(t, 3))
    if total - cuts[-1] < CUT_MIN and len(cuts) > 1:
        cuts.pop()
    cuts.append(round(total, 3))
    return cuts


def assign_families(cuts, seed=0):
    """Blocks of 20-40 s, families in turn. Every video opens on photographs: the
    first fifteen seconds sell the story, and a moving still does that better
    than a window of text."""
    rng = random.Random(seed + 1)
    # Photo every other block, screen and data taking turns between: about half
    # the running time is moving stills, a quarter each screens and data. Equal
    # thirds made a third of a 14-minute video text cards.
    order = ["photo", "screen", "photo", "data"]
    span = {"photo": (26, 36), "screen": (22, 32), "data": (20, 26)}
    fams, f, block_end = [], len(order) - 1, 0.0      # first increment lands on photo
    for a, b in zip(cuts, cuts[1:]):
        if a >= block_end:
            f = (f + 1) % len(order)
            block_end = a + rng.uniform(*span[order[f]])
        fams.append(order[f])
    return fams


def _spoken(words, a, b):
    return " ".join(w["word"] for w in words if a <= (w["start"] + w["end"]) / 2 < b).strip()


# --- content ------------------------------------------------------------------

def _whole_sentences(text):
    """Complete sentences only. A beat starts and ends wherever a cut fell, so its
    first and last pieces are usually halves of sentences ("...decryption. And the
    website was removed. Just") and are dropped."""
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    out = []
    for k, p in enumerate(parts):
        p = p.strip()
        if not p:
            continue
        first = p.split()[0]
        # "4chan." opens with a digit but is the tail of a sentence; a real start is
        # a capital or a number standing alone ("3301 was the signature.")
        starts = p[:1].isupper() or first.strip(".,").isdigit()
        ends = p[-1:] in ".!?"
        if (k == 0 and not starts) or (k == len(parts) - 1 and not ends):
            continue
        out.append(p)
    return out


def notification_line(text, most=16):
    """The one whole sentence a notification shows, or None.

    The first hook's notification read "and the website was removed. Just" - the
    end of one sentence and the start of the next, because a beat is cut where a
    word gap falls. A card now shows only a sentence that starts and ends inside
    the beat, the longest one that still fits."""
    sents = [x for x in _whole_sentences(text) if 3 <= len(x.split()) <= most]
    return max(sents, key=lambda x: len(x.split())) if sents else None


def _redactable(sentence):
    """Words worth a censor bar: names and numbers, never the first word."""
    words = sentence.split()
    return [w.strip(".,;:!?\"'") for k, w in enumerate(words)
            if k > 0 and (w[:1].isupper() or any(c.isdigit() for c in w))
            and len(w.strip(".,;:!?\"'")) > 1]


def _default(t, text, i, still=None):
    """A safe beat of type `t` when the model's answer cannot be used.

    Per type, not per family, so a bad answer does not undo the rotation that keeps
    two identical shots from following each other.
    """
    still = still or SAFE_STILLS[i % len(SAFE_STILLS)]
    words = text.split()
    if t == "stock":
        return {"type": t, "prompt": still}
    if t == "map":
        from pipeline import geo
        places = geo.places_in(text)
        if places:
            return {"type": t, "places": [list(x) for x in places]}
        return _default("quote", text, i, still)
    if t == "flux":
        return {"type": t, "prompt": still}
    if t == "annotate":
        return {"type": t, "prompt": still, "circle": [0.5, 0.48, 0.2]}
    if t in ("browser", "chat"):
        sents = [x for x in _whole_sentences(text) if 3 <= len(x.split()) <= 16]
        if not sents:
            return {"type": "flux", "prompt": still}
        if t == "browser":
            head = min(sents, key=lambda x: len(x.split()))
            rest = " ".join(x for x in sents if x != head)
            return {"type": t, "address": "archive / thread", "heading": head, "body": rest[:220] or " "}
        return {"type": t, "header": "Thread",
                "messages": [["anon" if k % 2 == 0 else "user", x] for k, x in enumerate(sents[:3])]}
    if t == "notification":
        line = notification_line(text)
        if line:
            return {"type": t, "prompt": still, "app": "Inbox", "text": line}
        return {"type": "flux", "prompt": still}
    if t == "cursor":
        return {"type": t, "prompt": still}
    if t in ("quote", "censor"):
        sents = _whole_sentences(text)
        if t == "censor":
            lines = sents[:3]
            redact = [w for s in lines for w in _redactable(s)][:3]
            if lines and redact:
                return {"type": t, "heading": "file", "lines": lines, "redact": redact}
            # nothing worth blacking out: a quote says more than random bars
        best = max((s for s in sents if len(s.split()) <= 14), key=lambda s: len(s.split()),
                   default=None)
        if not best and sents:                  # a long whole sentence beats a fragment
            best = min(sents, key=lambda s: len(s.split()))
        if not best:
            best = " ".join(words[:12]).rstrip(",;:") + "..." if words else "..."
        return {"type": "quote", "text": best}
    if t == "counter":
        n = re.search(r"\d[\d,]*", text)
        return {"type": t, "label": " ".join(words[:4]) or "count",
                "end": int(n.group(0).replace(",", "")) if n else 1}
    if t == "timeline":
        years = re.findall(r"\b(?:19|20)\d\d\b", text)
        if len(set(years)) >= 2:
            return {"type": t, "events": [[y, ""] for y in dict.fromkeys(years)][:4]}
        return _default("quote", text, i, still)
    return _default("quote", text, i, still)


_ALIASES = {"url": "address", "link": "address", "title": "heading", "headline": "heading",
            "text_body": "body", "content": "body", "description": "prompt",
            "image": "prompt", "image_prompt": "prompt", "msgs": "messages", "chat": "messages",
            "value": "end", "number": "end", "items": "events", "redacted": "redact"}


def _normalise(b):
    """Map the field names models reach for onto the menu's names."""
    out = dict(b)
    for k, v in b.items():
        if k in _ALIASES and _ALIASES[k] not in out:
            out[_ALIASES[k]] = v
    fields = out.pop("fields", None)
    if isinstance(fields, dict):
        out.update({k: v for k, v in fields.items() if k not in out})
    if out.get("type") == "browser" and "body" not in out and "text" in out:
        out["body"] = out["text"]
    if isinstance(out.get("messages"), list):
        msgs = []
        for m in out["messages"]:
            if isinstance(m, dict):
                m = [m.get("who") or m.get("user") or m.get("from") or "anon",
                     m.get("text") or m.get("message") or ""]
            if isinstance(m, (list, tuple)) and len(m) >= 2 and str(m[1]).strip():
                msgs.append([str(m[0]), str(m[1])])
        out["messages"] = msgs
    if isinstance(out.get("events"), list):
        ev = []
        for e in out["events"]:
            if isinstance(e, dict):
                e = [e.get("when") or e.get("date") or e.get("year") or "",
                     e.get("what") or e.get("text") or e.get("event") or ""]
            if isinstance(e, (list, tuple)) and len(e) >= 2:
                ev.append([str(e[0]), str(e[1])])
        out["events"] = ev
    return out


def _why_invalid(b, family):
    t = b.get("type")
    if t not in TYPES[family]:
        return f"type {t!r} not in {family}"
    need = {"flux": ["prompt"], "annotate": ["prompt", "circle"], "browser": ["address", "heading", "body"],
            "chat": ["messages"], "notification": ["prompt", "app", "text"], "cursor": ["prompt"],
            "counter": ["label", "end"], "timeline": ["events"], "censor": ["lines", "redact"],
            "quote": ["text"], "stock": ["prompt"], "map": ["places"]}[t]
    missing = [k for k in need if k not in b or b[k] in ("", [], None)]
    return f"{t} missing {missing}" if missing else None


def _facts_hold(b, narration):
    """Dates and numbers drawn on screen must be ones the narration actually says.

    The planner, asked for a timeline, wrote "2014 puzzle begins", "2015 first phone
    number" - plausible and wrong; the puzzle began in 2012. So every digit-bearing
    label is checked against the words being spoken over that beat.
    """
    said = set(re.findall(r"\d[\d,.]*", narration.replace(",", "")))
    t = b.get("type")
    if t == "timeline":
        for when, _ in b.get("events", []):
            for num in re.findall(r"\d[\d,.]*", str(when).replace(",", "")):
                if num not in said:
                    return False
        return bool(b.get("events"))
    if t == "counter":
        return str(b.get("end")).replace(",", "") in said
    if t == "quote":
        spoken = set(re.findall(r"[a-z0-9']+", narration.lower()))
        q = re.findall(r"[a-z0-9']+", str(b.get("text", "")).lower())
        return bool(q) and sum(w in spoken for w in q) / len(q) >= 0.8
    for field in ("heading", "body", "text", "label"):
        for num in re.findall(r"\d[\d,.]*", str(b.get(field, "")).replace(",", "")):
            if num not in said:
                return False
    return True


def _valid(b, family):
    t = b.get("type")
    if t not in TYPES[family]:
        return False
    need = {"flux": ["prompt"], "annotate": ["prompt", "circle"], "browser": ["address", "heading", "body"],
            "chat": ["messages"], "notification": ["prompt", "app", "text"], "cursor": ["prompt"],
            "counter": ["label", "end"], "timeline": ["events"], "censor": ["lines", "redact"],
            "quote": ["text"], "stock": ["prompt"], "map": ["places"]}[t]
    if t == "browser" and "body" in b and b["body"] in ("", " "):
        need = [k for k in need if k != "body"]
    if any(k not in b or b[k] in ("", [], None) for k in need):
        return False
    if t == "counter":
        try:
            b["end"] = int(re.sub(r"[^0-9-]", "", str(b["end"])))
        except ValueError:
            return False
    return True


# Objects FLUX cannot draw without lettering on them. A prompt that asks for one
# is refused before it reaches the GPU: the visual check would only reject the
# frame afterwards, and the planner kept reaching for them (a torn calendar page
# with a date, newspaper clippings, a pinned map).
_TEXT_OBJECTS = re.compile(
    r"\b(calendar|newspaper|clipping|headline|post-?it|sticky note|note ?pad|"
    r"sign|signs|signage|billboard|label|book cover|magazine|"
    r"printout|receipt|ticket|license plate|keyboard keys|"
    r"screen showing|on the screen|on a screen|website|web page|logo|brand|"
    r"displaying|showing|notice|thread|post|posts|message|messages|notification|"
    r"text|words|writing|caption|subtitle|interface|keypad|dial pad|"
    r"forum|homepage|home page|browser|cursor|app|icon|icons)\b", re.I)

# Substitutes when a prompt cannot be used. Mostly places and objects, lit
# colourfully at night; a few screens, no more than the share the edit allows.
SAFE_STILLS = [
    "rain running down a window at night, blurred teal and magenta city lights beyond",
    "a stack of old leather-bound books on a wooden table under a warm desk lamp, close-up",
    "a wooden telephone pole on a wet street at night, layers of torn paper flyers on it, neon reflections",
    "a folded paper map on a table lit by amber lamp light, overhead view, details out of focus",
    "a sealed envelope with a wax seal on a dark wooden desk, magenta neon light from a window",
    "an empty library aisle at night, tall shelves of books, warm lamps and teal shadows",
    "a payphone on an empty street corner at night under an amber streetlight, wet asphalt",
    "a city street at night seen from a rooftop, neon signs glowing teal and magenta, low angle",
    "a lamp post in fog at night, amber light, a bicycle leaning against it",
    "a stack of floppy disks and cassette tapes on a shelf, lit by a teal neon tube",
    "an old rotary telephone on a side table, warm lamp light, a magenta glow on the wall",
    "a shortwave radio on a shelf, its dial glowing amber, a teal window behind",
    "a satellite dish on a rooftop at night against a deep violet sky",
    "a tall radio mast at dusk with red warning lights over a city glow",
    "a dark stairwell lit by a green exit light and a magenta neon reflection",
    "a quiet suburban street at night, one window lit upstairs, wet road reflecting light",
    "a rain-soaked parking lot at night, puddles reflecting neon signs",
    "an old typewriter on a desk under a lamp, close-up of its keys catching amber light",
    "a cork board with pinned photographs and string, blurred and out of focus, lamp light",
    "a vintage key on an open old book, shallow depth of field, warm light",
    "fog over an empty road at night, distant headlights and a teal sky",
    "a subway platform at night, empty, teal and magenta lights reflecting on tiles",
    "an attic with a single bare bulb, cardboard boxes and old papers in warm light",
    "a hand holding a magnifying glass over an old photograph, out of focus, lamp light",
    "a chess board mid-game on a cafe table at night, neon light through the window",
    "a row of server racks with blue and green indicator lights, wide shot",
    "an empty internet cafe at night, chairs and desks, neon sign light from the street",
    "a dark desk with one glowing CRT monitor, teal light across the wall",
]

REPAIR_SYSTEM = """You rewrite image prompts for a photo generator that cannot draw text.
Each prompt below was refused because it names something that carries writing (a
calendar, a page, a sign, a screen showing content, a website...). Rewrite each as a
moody photograph of a place or object with NO writing anywhere: screens are only
glow, nothing is printed, nothing is labelled. Keep the mood and the idea. One
sentence each. Return JSON only: {"prompts": [{"i": <index>, "prompt": "..."}]}"""


def assign_types(beats):
    """Rotate the types inside each family so no shot repeats the one before it.

    Photo beats alternate real footage and generated stills, with an annotated
    still now and then. A beat that names two or more places becomes a map."""
    from pipeline import geo
    turn = {f: 0 for f in FAMILIES}
    prev = None
    for b in beats:
        if len(geo.places_in(b["text"])) >= 2:
            b["family"], b["type"], prev = "data", "map", "map"
            continue
        options = [t for t in TYPES[b["family"]] if t != "map"]
        if b["family"] == "data":
            if not re.search(r"\d", b["text"]):
                options = [t for t in options if t != "counter"]
            if len(re.findall(r"(?:19|20)\d\d", b["text"])) < 2:
                options = [t for t in options if t != "timeline"]
        if b["family"] == "photo":
            options = ["stock", "flux", "stock", "annotate", "flux"]
        if b["family"] == "screen":
            options = ["browser", "chat", "notification"]    # cursor shots were empty monitors
        k = turn[b["family"]]
        t = options[k % len(options)]
        if t == prev and len(set(options)) > 1:
            k += 1
            t = options[k % len(options)]
        turn[b["family"]] = k + 1
        b["type"], prev = t, t
    return beats


def is_monitor(shot):
    """A generated still of a screen: what the edit keeps under MONITOR_SHARE_MAX."""
    return shot["type"] in ("flux", "annotate", "notification", "cursor") and \
        bool(SCREEN_WORDS.search(shot.get("prompt") or ""))


def cap_monitors(shots, pool):
    """Swap monitor prompts for places and objects until at most
    MONITOR_SHARE_MAX of the shots are screens. Returns how many were swapped."""
    limit = int(MONITOR_SHARE_MAX * len(shots))
    mons = [s for s in shots if is_monitor(s)]
    swapped = 0
    # keep the ones spread furthest apart; replace every other one first
    for s in mons[1::2] + mons[0::2]:
        if sum(is_monitor(x) for x in shots) <= limit:
            break
        s["prompt"] = pool()
        swapped += 1
    return swapped


def plan(lines, words, total, seed=0, providers=llm.PROVIDERS):
    cuts = cut_points(words, total, seed)
    fams = assign_families(cuts, seed)
    beats = [{"i": i, "start": a, "end": b, "family": fams[i], "text": _spoken(words, a, b)}
             for i, (a, b) in enumerate(zip(cuts, cuts[1:]))]
    assign_types(beats)
    # In batches: two hundred beats in one request came back truncated and every
    # one of them fell to a default. A batch that fails costs only its own beats.
    filled = {}
    for k in range(0, len(beats), PLAN_BATCH):
        part = beats[k:k + PLAN_BATCH]
        ask = json.dumps([{"i": b["i"], "type": b["type"], "narration": b["text"]} for b in part],
                         ensure_ascii=False)
        try:
            answer = llm.complete_json(f"BEATS:\n{ask}", system=PLANNER_SYSTEM, max_tokens=6000,
                                       temperature=0.6, providers=providers)
            rows = answer.get("beats", []) if isinstance(answer, dict) else answer
            filled.update({int(x.get("i", -1)): x for x in rows if isinstance(x, dict)})
        except Exception as e:                                   # noqa: BLE001
            log.warning("planner batch %d failed (%s); its beats get defaults", k // PLAN_BATCH, e)
    # Prompts that ask for lettering go back to the model once, to be rewritten
    # without it; only what is still refused after that falls to the safe pool.
    bad = {i: x["prompt"] for i, x in ((i, _normalise(x)) for i, x in filled.items())
           if x.get("prompt") and _TEXT_OBJECTS.search(x["prompt"])}
    for k in range(0, len(bad), PLAN_BATCH):
        part = dict(list(bad.items())[k:k + PLAN_BATCH])
        try:
            a = llm.complete_json("PROMPTS:\n" + json.dumps([{"i": i, "prompt": p} for i, p in part.items()]),
                                  system=REPAIR_SYSTEM, max_tokens=3000, temperature=0.5,
                                  providers=providers)
            for r in (a.get("prompts", []) if isinstance(a, dict) else a):
                i, p = int(r.get("i", -1)), str(r.get("prompt", ""))
                if i in filled and p and not _TEXT_OBJECTS.search(p):
                    filled[i]["prompt"] = p
        except Exception as e:                                   # noqa: BLE001
            log.warning("prompt repair failed (%s)", e)

    shots, used_defaults, refused = [], 0, 0
    # One pool for every substitute still, so no two shots draw the same one
    # until all of them are used; then a still nobody has used for a while.
    spare = list(SAFE_STILLS)
    used_at = {}

    def next_safe(i, screens=True):
        cands = [q for q in spare if screens or not SCREEN_WORDS.search(q)]
        if cands:
            p = cands[0]
            spare.remove(p)
        else:
            pool = [q for q in SAFE_STILLS if screens or not SCREEN_WORDS.search(q)]
            p = min(pool, key=lambda q: used_at.get(q, -999))
        used_at[p] = i
        return p
    for b in beats:
        got = _normalise(filled.get(b["i"], {}))
        got["type"] = b["type"]
        if got.get("prompt") and _TEXT_OBJECTS.search(got["prompt"]):
            log.info("beat %d: prompt asks for lettering (%s), replaced", b["i"],
                     _TEXT_OBJECTS.search(got["prompt"]).group(0))
            got["prompt"] = next_safe(b["i"])
            refused += 1
        if got.get("type") == "map":
            from pipeline import geo
            got["places"] = [list(x) for x in geo.places_in(b["text"])]
        if got.get("type") == "notification" and got.get("text") is not None:
            # whole sentences only, from what is being said
            line = notification_line(b["text"])
            if line:
                got["text"] = line
            else:
                got = {"type": "flux", "prompt": got.get("prompt") or next_safe(b["i"], False)}
        if got.get("type") in ("quote", "censor"):
            # The first full episode showed "THEN, JULY 2015," and "Call us on a"
            # as cards: model answers cut mid-sentence. Cards now carry only
            # whole sentences of the beat, through the same default the planner
            # falls back to; with none to show, the shot is a still.
            whole = _whole_sentences(b["text"])
            ok = False
            if got["type"] == "quote":
                q = str(got.get("text", "")).strip()
                ok = any(q.strip(' "') == x.strip(' "') for x in whole)
            else:
                card = [str(x).strip() for x in got.get("lines", [])]
                ok = bool(card) and all(any(c == x for x in whole) for c in card)
            if not ok:
                got = _default(got["type"], b["text"], b["i"], next_safe(b["i"], False))
                if got["type"] == "quote" and got["text"].endswith("..."):
                    got = {"type": "flux", "prompt": next_safe(b["i"], False)}
        if got.get("type") in ("browser", "chat"):
            # Words in a window are whole sentences from what is being said, like
            # the notification: the second hook showed "appears on 4chan. If you".
            sents = [x for x in _whole_sentences(b["text"]) if 3 <= len(x.split()) <= 16]
            if not sents:
                got = {"type": "flux", "prompt": next_safe(b["i"], False)}
            elif got["type"] == "browser":
                head = min(sents, key=lambda x: len(x.split()))
                rest = [x for x in sents if x != head]
                got["heading"] = head
                got["body"] = " ".join(rest)[:220] if rest else ""
                got.setdefault("address", "archive / thread")
            else:
                got["messages"] = [["", x] for x in sents[:3]]
        if got.get("type") == "chat":
            # Nobody real speaks in these threads: handles are always faceless.
            got["messages"] = [["anon" if k % 2 == 0 else "user", m[1]]
                               for k, m in enumerate(got.get("messages") or [])][:3]
        fam = "photo" if got["type"] == "flux" and b["family"] == "screen" else b["family"]
        if not _valid(got, fam):
            log.info("beat %d: %s; default used", b["i"], _why_invalid(got, b["family"]))
            got = _default(b["type"], b["text"], b["i"], next_safe(b["i"], False))
            used_defaults += 1
        elif not _facts_hold(got, b["text"]):
            log.info("beat %d: %s shows a number the narration does not say; default used",
                     b["i"], got["type"])
            got = _default(b["type"], b["text"], b["i"], next_safe(b["i"]))
            used_defaults += 1
        got.update({"start": b["start"], "end": b["end"], "family": b["family"],
                    "narration": b["text"]})
        shots.append(got)
    swapped = cap_monitors(shots, lambda: next_safe(-1, screens=False))
    for k, s in enumerate(shots):
        s["transition"] = TRANSITIONS[k % 3] if k else None
        s["move"] = k
    share = sum(is_monitor(s) for s in shots) / max(1, len(shots))
    log.info("monitor shots: %.0f%% (%d swapped for places and objects)", share * 100, swapped)
    log.info("%d shots, %d defaults, %d prompts refused for lettering; families: %s",
             len(shots), used_defaults, refused, " ".join(s["family"][0] for s in shots))
    return {"shots": shots, "accents": accents(lines, words)}


def accents(lines, words):
    """Punch-ins, risers, impacts, kinetic words and the final shake, from the delivery.

    Every slow line is a reveal: a riser runs into it, an impact lands on its first
    word, the frame punches in, and its strongest word or two appear on screen. The
    last slow line also gets a small shake.
    """
    out = {"punch": [], "riser": [], "impact": [], "kinetic": [], "shake": []}
    slow = [l for l in lines if l["kind"] == "slow"]
    for n, l in enumerate(slow):
        first = next((w for w in words if w["start"] >= l["start"] - 0.05), None)
        if not first:
            continue
        hit = first["start"]
        out["riser"].append(round(max(0.0, hit - 1.8), 3))
        out["impact"].append(round(hit, 3))
        out["punch"].append([round(hit, 3), round(min(l["end"], hit + 2.2), 3)])
        line_words = [w for w in words if l["start"] - 0.05 <= w["start"] < l["end"]]
        pick = _strong_words(_script_words(l, line_words))
        if pick:
            out["kinetic"].append({"words": [w["word"] for w in pick],
                                   "start": pick[0]["start"],
                                   "end": round(max(pick[-1]["end"] + 0.9, pick[0]["start"] + 1.4), 3)})
        if n == len(slow) - 1:
            out["shake"].append([round(hit, 3), round(hit + 0.6, 3)])
    return out


def _script_words(line, heard):
    """The line's own words from the script, each timed from what Whisper heard.

    Whisper's spelling is not the script's ("hit" for "hid", digits for numbers),
    and words put on screen must be the script's. So the text comes from the line
    and only the timing from Whisper: each script word takes the time of the best
    matching heard word, or a position proportional to where it sits in the line.
    """
    import difflib
    script = [w for w in re.findall(r"[\w'.-]+", line["text"]) if w.strip(".-")]
    out, j = [], 0
    for k, word in enumerate(script):
        best, best_r = None, 0.0
        for m in range(j, min(len(heard), j + 4)):
            r = difflib.SequenceMatcher(None, word.lower(), heard[m]["word"].lower().strip(".,!?")).ratio()
            if r > best_r:
                best, best_r = m, r
        if best is not None and best_r >= 0.5:
            out.append({"word": word, "start": heard[best]["start"], "end": heard[best]["end"]})
            j = best + 1
        else:
            span = line["end"] - line["start"]
            t = line["start"] + span * k / max(1, len(script))
            out.append({"word": word, "start": round(t, 3), "end": round(t + 0.3, 3)})
    return out


_STOP = set("the a an and or but of to in on at for with that this it is was were be been "
            "you your we our they their who what why how when where there here so then".split())


_WEAK = set("call start look stay find follow said says made make take came went "
             "told tell ever never still just only really very".split())


def _strong_words(line_words, most=2):
    """One or two words worth putting on screen: numbers, names, long words.

    A capital counts as a name only away from the start of the line - "Call" and
    "Start" opening a sentence were being picked as if they mattered.
    """
    def score(k, w):
        word = re.sub(r"[^\w]", "", w["word"])
        if not word or word.lower() in _STOP or word.lower() in _WEAK:
            return -1
        name = word[:1].isupper() and k > 0
        return (3 if word[:1].isdigit() else 0) + (2 if name else 0) + len(word) / 4
    scored = sorted(((score(k, w), k, w) for k, w in enumerate(line_words)),
                    key=lambda x: x[0], reverse=True)
    pick = [w for s, k, w in scored[:most] if s > 1.9]
    return sorted(pick, key=lambda w: w["start"])
