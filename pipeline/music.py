"""Choosing and building the music bed: which mood, where, and in what order.

Episode one used one track from end to end. A twenty-minute piece has an arc -
a threat posed, a mechanism explained, a consequence that lands - and holding
one loop under all of it is what makes a long video feel flat even when the
pictures change every five seconds.

So the bed is assembled from the library by mood: the arc is read off the
script once per episode, each stretch gets a track tagged for that mood, and
the joins are crossfaded so nothing switches abruptly under the narration.
"""
import hashlib
import json
import subprocess
from pathlib import Path

from core.logger import get_logger

log = get_logger("music")

ROOT = Path(__file__).resolve().parent.parent
LIBRARY = ROOT / "assets" / "music" / "library.json"

# Long enough that a change of track reads as a transition rather than an edit.
XFADE_S = 6.0
# Below this a stretch is not worth its own track; it gets merged into the next.
MIN_SEGMENT_S = 75.0

MOODS = ("tense", "curious", "wonder", "driving")

# Used when there is no LLM to read the arc: the shape nearly every "what if"
# script follows anyway - a threat, the physics behind it, what it leaves.
DEFAULT_ARC = (("tense", 0.00), ("curious", 0.22), ("driving", 0.55),
               ("wonder", 0.80))

ARC_PROMPT = """You are scoring music for a science documentary.

Below is the narration script. Divide it into 3 to 5 consecutive stretches and
give each one a mood from exactly this list: tense, curious, wonder, driving.

- tense: a threat is being set up, something is wrong
- curious: a mechanism is being explained, numbers and evidence
- driving: consequences arriving, events in sequence, momentum
- wonder: scale, aftermath, what it all means

Return JSON only: {"segments": [{"start_fraction": 0.0, "mood": "tense"}, ...]}
start_fraction is where the stretch begins as a fraction of the whole script,
first one always 0.0, strictly increasing, all below 1.0.

SCRIPT:
"""


def load_library(path=LIBRARY):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    tracks = data["tracks"]
    for t in tracks:
        t["path"] = str((Path(path).parent / t["file"]).resolve())
        missing = [m for m in t["moods"] if m not in MOODS]
        if missing:
            raise ValueError(f"{t['file']} has unknown moods {missing}")
    return tracks


def _duration(path):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "csv=p=0", str(path)], capture_output=True, text=True)
    return float(out.stdout.strip())


def read_arc(script_text, use_llm=True):
    """The mood arc of this script, as (mood, start_fraction) pairs.

    One LLM call per episode. A keyword list was the obvious cheaper option and
    the wrong one: "collapse" is tense in a hook and matter-of-fact in a
    paragraph about orbital mechanics, and the planner already showed how badly
    hand-written word lists read context.
    """
    if not use_llm:
        return list(DEFAULT_ARC)
    try:
        from core import llm
        # The lite model, with room to spare: the thinking model spent 383 of a
        # 400-token budget reasoning about a four-line answer and returned
        # nothing at all.
        segments = llm.complete_json(
            ARC_PROMPT + script_text[:12000], max_tokens=800, temperature=0.2,
            model=llm.GEMINI_BULK_MODEL)["segments"]
        arc = [(s["mood"], float(s["start_fraction"])) for s in segments
               if s.get("mood") in MOODS]
        if not arc or arc[0][1] > 0.001:
            raise ValueError(f"arc does not start at zero: {arc}")
        fractions = [f for _, f in arc]
        if fractions != sorted(fractions) or fractions[-1] >= 1.0:
            raise ValueError(f"arc fractions are not increasing below 1: {arc}")
        log.info("mood arc: %s", ", ".join(f"{m}@{f:.2f}" for m, f in arc))
        return arc
    except Exception as e:
        log.warning("could not read the mood arc (%s); using the default shape", e)
        return list(DEFAULT_ARC)


def plan(script_text, total_s, tracks=None, use_llm=True):
    """Turn the arc into concrete (track, seconds) stretches covering total_s.

    A track is not reused back to back, so a repeated mood still changes colour.
    """
    tracks = tracks or load_library()
    arc = read_arc(script_text, use_llm=use_llm)

    bounds = [f * total_s for _, f in arc] + [total_s]
    stretches = []
    for (mood, _), start, end in zip(arc, bounds, bounds[1:]):
        if stretches and end - start < MIN_SEGMENT_S:
            stretches[-1] = (stretches[-1][0], stretches[-1][1] + (end - start))
            continue
        stretches.append((mood, end - start))

    # The choice is varied but deterministic: seeded from the script, so two
    # episodes do not open on the same track while a re-render of one episode
    # is identical to the first. Taking options[0] every time meant a library
    # of eight tracks only ever played two.
    seed = int(hashlib.sha256(script_text.encode("utf-8")).hexdigest()[:8], 16)
    plan_out, previous, used = [], None, set()
    for position, (mood, length) in enumerate(stretches):
        options = [t for t in tracks if mood in t["moods"]]
        if not options:
            options = tracks
            log.warning("no track tagged '%s'; using the whole library", mood)
        fresh = [t for t in options if t["file"] not in used] or options
        fresh = [t for t in fresh if t["path"] != previous] or fresh
        pick = fresh[(seed + position) % len(fresh)]
        previous, _ = pick["path"], used.add(pick["file"])
        plan_out.append({"mood": mood, "seconds": round(length, 2),
                         "track": pick})
    for p in plan_out:
        log.info("  %-8s %5.0fs  %s", p["mood"], p["seconds"], p["track"]["title"])
    return plan_out


def build_bed(plan_out, out_path, work_dir=None):
    """Render the planned stretches into one continuous, crossfaded bed.

    Each stretch is cut XFADE_S longer than asked so the crossfades do not eat
    into the running time; the bed therefore ends slightly long, which is what
    the mixer wants - it is truncated to the narration.
    """
    out_path = Path(out_path)
    work = Path(work_dir or out_path.parent) / "_bed"
    work.mkdir(parents=True, exist_ok=True)

    parts = []
    for i, p in enumerate(plan_out):
        src = p["track"]["path"]
        length = p["seconds"] + XFADE_S
        part = work / f"{i:02d}.wav"
        # aloop on the decoded stream so a track shorter than its stretch
        # simply repeats rather than running out into silence.
        subprocess.run(
            ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", src,
             "-af", f"aresample=48000,aformat=channel_layouts=stereo,"
                    f"aloop=loop=-1:size=2e9,atrim=0:{length:.3f},"
                    f"afade=t=in:st=0:d=2",
             "-ar", "48000", "-ac", "2", str(part)], check=True)
        parts.append(part)

    if len(parts) == 1:
        parts[0].replace(out_path)
        return out_path

    inputs, graph, last = [], [], "0:a"
    for i, part in enumerate(parts):
        inputs += ["-i", str(part)]
    for i in range(1, len(parts)):
        label = f"x{i}"
        graph.append(f"[{last}][{i}:a]acrossfade=d={XFADE_S}:c1=tri:c2=tri[{label}]")
        last = label
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", *inputs,
         "-filter_complex", ";".join(graph), "-map", f"[{last}]",
         "-ar", "48000", "-ac", "2", str(out_path)], check=True)
    for part in parts:
        part.unlink(missing_ok=True)
    log.info("bed built: %.0fs from %d stretches", _duration(out_path), len(parts))
    return out_path


def attribution(plan_out):
    """The credit lines for every track actually used, in order, deduplicated."""
    seen, lines = set(), []
    for p in plan_out:
        t = p["track"]
        if t["file"] in seen:
            continue
        seen.add(t["file"])
        lines.append(f'{t["title"]} by {t["artist"]} ({t["source"]}) '
                     f'- {t["license"]}')
    return lines
