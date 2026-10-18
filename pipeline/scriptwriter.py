"""Write one Closed Tabs script from its gathered sources, and nothing else.

The facts come only from the sources passed in; the model is told so and the
sources are listed in the description. The shape is written for a voice and for
retention: the most impossible fact in the first fifteen seconds with a promise
of an answer, open loops, a small cliffhanger roughly every two minutes, the
answer or the best theory at the end, and a subscribe line only in the last
twenty seconds.

Each script gets its own structure and its own form of title. A run of
identically shaped videos ("10 Stoic Lessons...") is what YouTube reads as
mass-produced, which is a monetisation risk, so the recent titles are passed in
and their form must not be repeated.
"""
import re

from core import llm
from core.logger import get_logger

log = get_logger("script")

TARGET_WORDS = 2000          # ~11.5-12.5 min at 170 wpm plus the pauses
SOURCE_CHARS = 14000         # per source, enough facts without drowning the prompt

SYSTEM = """You write narration for Closed Tabs, an English-language YouTube channel about
real internet mysteries. You are a careful researcher and a gripping storyteller.

HARD RULES
- Use ONLY facts stated in the SOURCES. If the sources do not say it, do not say it.
  Where sources disagree or are unsure, say so in the narration. Never invent dates,
  numbers, quotes, names or events.
- Never talk about "the sources", "the encyclopedia", "our magazine account" or "the
  report we are using". Attribute the way a journalist would, inside the story: "one
  account from 2014 says...", "investigators at the time...", "a technology magazine
  reported...". The sources are listed in the description; the narration is a story.
- Never name private individuals, and never describe one by identifying detail.
  Solvers, witnesses, users and anyone a journalist interviewed are "one early solver",
  "a user", "a researcher": no age, no home city or neighbourhood, no job, school,
  thesis or family detail. Tell the story through what happened, not who it happened to. Never discuss real deaths, suicides, crime victims,
  minors, active criminal cases, or graphic violence.
- No greeting, no channel intro, no "in this video", no request to subscribe until the
  final twenty seconds.

SHAPE
- 0-15 seconds: open on the single most impossible true fact, then promise that the
  answer (or the best explanation) is coming.
- Keep open loops: "but that was not the strangest part", a question left hanging.
  A small cliffhanger roughly every two minutes.
- The answer, or the best-supported theory with its weak points, comes near the end.
- The last twenty seconds may invite the viewer to subscribe, once, briefly.
- Choose a structure that suits THIS story (chronological, countdown of clues,
  investigation, two competing theories...). Do not reuse the shape of earlier videos.

WRITTEN FOR THE VOICE
- Short sentences. Fragments are fine. Questions to the viewer.
- One line per spoken phrase, with delivery marks:
    > line     fast and confident: hooks, transitions, lists
    < line     slower, preceded by a pause: reveals and key facts
    plain      normal pace
    [pause 0.7]  an explicit silence, between 0.4 and 1.0 seconds, used sparingly
"""

PROMPT = """TOPIC: {name}
ONE-LINE PITCH: {pitch}

RECENT TITLES ON THE CHANNEL (do not imitate their form):
{recent}

SOURCES (the only facts you may use):
{sources}

Write the full narration, about {words} words of spoken text.

Return exactly this layout:
TITLE: <a YouTube title under 70 characters, its own form - not a list, not "The Mystery of">
THUMBNAIL: <two or three words for the thumbnail that add to the title rather than repeat it>
===SCRIPT===
<the marked-up narration>
===CLAIMS===
<one line per factual claim: the claim | the source number it came from>
"""


def _sources_block(sources):
    return "\n\n".join(f"[{i}] {s['title']} ({s['url']})\n{s['text'][:SOURCE_CHARS]}"
                       for i, s in enumerate(sources, 1))


def write(topic, sources, recent_titles=(), words=TARGET_WORDS, providers=llm.PROVIDERS):
    prompt = PROMPT.format(name=topic["name"], pitch=topic.get("pitch", ""),
                           recent="\n".join(f"- {t}" for t in recent_titles) or "- (none yet)",
                           sources=_sources_block(sources), words=words)
    raw = llm.complete(prompt, system=SYSTEM, max_tokens=8000, temperature=0.8,
                       providers=providers)
    return parse(raw)


def parse(raw):
    title = re.search(r"^TITLE:\s*(.+)$", raw, re.M)
    thumb = re.search(r"^THUMBNAIL:\s*(.+)$", raw, re.M)
    body = raw.split("===SCRIPT===", 1)[-1]
    script, _, claims = body.partition("===CLAIMS===")
    script = script.strip()
    if len(script.split()) < 300:
        raise ValueError(f"the reply has no usable script ({len(script.split())} words)")
    return {"title": title.group(1).strip().strip('"') if title else "",
            "thumbnail_words": thumb.group(1).strip().strip('"') if thumb else "",
            "script": script,
            "claims": [c.strip() for c in claims.strip().splitlines() if c.strip()]}


def extend(topic, sources, script, missing_words, providers=llm.PROVIDERS):
    """One round of lengthening a script that reads short, from the same sources."""
    prompt = (f"This narration about {topic['name']} reads about {missing_words} words short.\n"
              f"Expand it from the SOURCES only: add detail where the story moves fastest, "
              f"keep every delivery mark, keep the ending as it is.\n\n"
              f"SOURCES:\n{_sources_block(sources)}\n\nNARRATION:\n{script}\n\n"
              f"Return only the full expanded narration, same markup.")
    out = llm.complete(prompt, system=SYSTEM, max_tokens=8000, temperature=0.7,
                       providers=providers).strip()
    return out if len(out.split()) > len(script.split()) else script
