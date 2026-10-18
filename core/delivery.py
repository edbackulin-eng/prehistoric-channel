"""Delivery markup: how a line is spoken, written into the script itself.

The voice of this channel is not one speed. The hook and the transitions go
fast and sure; a reveal slows down, and is preceded by a beat of silence. So
the script carries that, line by line:

    > fast line        quick and confident: hooks, transitions, lists
    < slow line        slower, with a pause before it: reveals, key facts
    plain line         the base pace
    [pause 0.8]        an explicit silence, in seconds

Every TTS engine reads the same parse, so swapping the voice does not change
the timing the script was written for.
"""
import re
from dataclasses import dataclass

FAST, BASE, SLOW = 1.10, 1.02, 0.95       # Kokoro speed factors, kept in 0.95-1.12
SLOW_LEAD_S = 0.75                        # silence before every slow line
LINE_GAP_S = 0.12                         # ordinary breath between lines
_PAUSE = re.compile(r"^\[pause\s+([0-9.]+)\]$", re.I)


@dataclass
class Line:
    text: str
    kind: str          # "fast" | "base" | "slow"
    speed: float
    pause_before: float


def parse(script):
    lines, pending = [], 0.0
    for raw in script.splitlines():
        raw = raw.strip()
        if not raw:
            continue
        m = _PAUSE.match(raw)
        if m:
            pending += float(m.group(1))
            continue
        if raw.startswith(">"):
            kind, speed, text = "fast", FAST, raw[1:].strip()
        elif raw.startswith("<"):
            kind, speed, text = "slow", SLOW, raw[1:].strip()
        else:
            kind, speed, text = "base", BASE, raw
        gap = pending if pending else LINE_GAP_S
        if kind == "slow":
            gap = max(gap, SLOW_LEAD_S)
        lines.append(Line(text, kind, speed, 0.0 if not lines and not pending else gap))
        pending = 0.0
    return lines


def plain_text(script):
    """The words alone, for subtitles and word counts."""
    return " ".join(l.text for l in parse(script))
