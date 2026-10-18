"""Build a full episode from its script: narration to finished, upload-ready files.

    python automation/build_episode.py cicada-3301 [--force-window] [--stills-only]

Stages, each skipped when its output already exists (so a failed run resumes):
  narrate   Kokoro am_michael over the marked-up script; Whisper times every word.
            Length is judged by the narration's duration, 11-14 minutes. A short
            script gets one round of lengthening from the same sources; after that
            the video goes ahead whatever the length, with the miss recorded.
  plan      the edit (pipeline/editplan.py)
  stock     real footage and photographs for the stock shots, checked and credited
            (automation/stock_assets.py)
  stills    FLUX on Kaggle plus thumbnail bases, all checked (automation/stills.py)
  render    picture, transitions, sound, accents (pipeline/edit.py)
  package   subtitles from the script's own words, chapters, description with
            sources and music credits, tags, thumbnail
Output: queue/<slug>/out/ with video.mp4, thumbnail.jpg, subs.srt, metadata.json.
"""
import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

import soundfile as sf

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from core import llm, narrate, style  # noqa: E402
from core.logger import get_logger  # noqa: E402
from core.voice import SAMPLE_RATE  # noqa: E402
from pipeline import edit, editplan, scriptwriter, sources  # noqa: E402

log = get_logger("episode")

MIN_S, MAX_S = 11 * 60, 14 * 60
NEEDS_STILL = {"flux", "annotate", "notification", "cursor"}


def _timed_words(wav):
    from faster_whisper import WhisperModel
    model = WhisperModel("small.en", device="cpu", compute_type="int8")
    segments, _ = model.transcribe(str(wav), word_timestamps=True, vad_filter=False)
    return [{"word": w.word.strip(), "start": round(w.start, 3), "end": round(w.end, 3)}
            for s in segments for w in s.words]


def stage_narrate(q, b, report):
    wav = b / "narration.wav"
    if wav.exists() and (b / "words.json").exists():
        return
    kokoro = narrate.load_kokoro()
    script = (q / "script.txt").read_text(encoding="utf-8")
    audio, lines = narrate.synthesize(kokoro, script)
    seconds = len(audio) / SAMPLE_RATE
    if seconds < MIN_S:
        topic = json.loads((q / "meta.json").read_text(encoding="utf-8"))
        queue = json.loads((ROOT / "topics" / "queue.json").read_text(encoding="utf-8"))
        t = next(x for x in queue["topics"] if x["slug"] == q.name)
        missing = int((MIN_S + 45 - seconds) * 170 / 60)
        log.info("narration %.1f min is short; one round of lengthening (+%d words)",
                 seconds / 60, missing)
        longer = scriptwriter.extend(t, sources.gather(t), script, missing)
        (q / "script.txt").write_text(longer + "\n", encoding="utf-8")
        audio, lines = narrate.synthesize(kokoro, longer)
        seconds = len(audio) / SAMPLE_RATE
        report["lengthened"] = True
    if not MIN_S <= seconds <= MAX_S:
        report.setdefault("warnings", []).append(
            f"narration is {seconds / 60:.1f} min, outside 11-14; the video goes ahead")
    sf.write(wav, audio, SAMPLE_RATE)
    (b / "lines.json").write_text(json.dumps(lines, indent=1, ensure_ascii=False), encoding="utf-8")
    (b / "words.json").write_text(json.dumps(_timed_words(wav), indent=1), encoding="utf-8")
    report["narration_min"] = round(seconds / 60, 2)


def stage_plan(b, slug):
    if (b / "plan.json").exists():
        return
    lines = json.loads((b / "lines.json").read_text(encoding="utf-8"))
    words = json.loads((b / "words.json").read_text(encoding="utf-8"))
    total = sf.info(str(b / "narration.wav")).duration
    plan = editplan.plan(lines, words, total, seed=sum(map(ord, slug)))
    plan["prompts"] = {s["move"]: style.scene_prompt(s["prompt"], "person" in s["prompt"])
                       for s in plan["shots"] if s["type"] in NEEDS_STILL}
    (b / "plan.json").write_text(json.dumps(plan, indent=1, ensure_ascii=False), encoding="utf-8")


def stage_stock(b):
    if (b / "credits.json").exists():
        return
    subprocess.run([sys.executable, str(ROOT / "automation" / "stock_assets.py"),
                    str(b / "plan.json")], check=True)


def stage_stills(b, force_window):
    cmd = [sys.executable, str(ROOT / "automation" / "stills.py"), str(b / "plan.json"),
           "--thumbs", "8"]
    if force_window:
        cmd.append("--force-window")
    subprocess.run(cmd, check=True)


# --- packaging ---------------------------------------------------------------------

def _ts(t, srt=True):
    h, rem = divmod(t, 3600)
    m, s = divmod(rem, 60)
    if srt:
        return f"{int(h):02d}:{int(m):02d}:{int(s):02d},{int(round((s % 1) * 1000)) % 1000:03d}"
    return f"{int(m)}:{int(s):02d}" if h < 1 else f"{int(h)}:{int(m):02d}:{int(s):02d}"


def subtitles(lines, path, max_chars=42):
    """One cue per line of the script, in its own spelling; long lines are split
    into two-row cues with the time shared out by length."""
    cues = []
    for l in lines:
        words = l["text"].split()
        chunks, cur = [], ""
        for w in words:
            if len(cur) + len(w) + 1 > max_chars * 2 and cur:
                chunks.append(cur)
                cur = w
            else:
                cur = f"{cur} {w}".strip()
        if cur:
            chunks.append(cur)
        total = sum(len(c) for c in chunks) or 1
        t = l["start"]
        for c in chunks:
            d = (l["end"] - l["start"]) * len(c) / total
            text = c
            if len(c) > max_chars:                       # wrap into two rows
                cut = c.rfind(" ", 0, max_chars + 1)
                text = c[:cut] + "\n" + c[cut + 1:] if cut > 0 else c
            cues.append((t, t + d, text))
            t += d
    with open(path, "w", encoding="utf-8") as f:
        for i, (a, z, text) in enumerate(cues, 1):
            f.write(f"{i}\n{_ts(a)} --> {_ts(z)}\n{text}\n\n")
    return len(cues)


def chapters_and_tags(lines, title):
    """5-7 chapter titles at line starts, and 12-18 tags, from one model call."""
    listing = "\n".join(f"{i}|{l['start']:.0f}|{l['text'][:90]}" for i, l in enumerate(lines))
    prompt = (f"Video title: {title}\nNarration lines as index|seconds|text:\n{listing}\n\n"
              "Return JSON: {\"chapters\": [{\"line\": <index>, \"title\": \"<2-5 words>\"}], "
              "\"tags\": [\"...\"]}. 5 to 7 chapters, the first at line 0, each at least "
              "60 seconds after the previous, titles that intrigue without spoiling the "
              "answer. 12 to 18 tags, lower case, no hashtags.")
    try:
        a = llm.complete_json(prompt, max_tokens=1500, temperature=0.4)
        chs = [c for c in a.get("chapters", []) if isinstance(c, dict)]
        tags = [str(t).strip()[:30] for t in a.get("tags", []) if str(t).strip()][:18]
    except Exception as e:                                      # noqa: BLE001
        log.warning("chapters/tags failed (%s); using a plain set", e)
        chs, tags = [], []
    out, last = [], -999
    for c in sorted(chs, key=lambda c: int(c.get("line", 0))):
        i = max(0, min(len(lines) - 1, int(c.get("line", 0))))
        t = 0.0 if not out else lines[i]["start"]
        if t - last >= 60 or not out:
            out.append((t, str(c.get("title", "")).strip()[:40] or "Part"))
            last = t
    if len(out) < 3:                                    # YouTube needs at least three
        n = len(lines)
        out = [(0.0, "The message")] + [(lines[n * k // 4]["start"], f"Part {k + 1}")
                                        for k in (1, 2, 3)]
    if not tags:
        tags = ["internet mysteries", "unsolved", "internet history", "arg", "documentary"]
    return out, tags


def description(meta, chapters, music_lines, hook_line, footage=()):
    src = "\n".join(f"- {s['title']}: {s['url']}" for s in meta["sources"])
    chs = "\n".join(f"{_ts(t, srt=False)} {name}" for t, name in chapters)
    music = "\n".join(music_lines)
    foot = ""
    if footage:
        from pipeline import stock
        foot = ("FOOTAGE AND PHOTOGRAPHS\n" + "\n".join(f"- {stock.credit(a)}" for a in footage)
                + "\nWorld map outline: Natural Earth (public domain).\n\n")
    return (f"{hook_line}\n\n"
            f"CHAPTERS\n{chs}\n\n"
            f"SOURCES\nThis episode is written only from these open sources:\n{src}\n\n"
            f"{foot}"
            f"MUSIC\n{music}\nLicensed under Creative Commons: By Attribution 4.0\n"
            f"https://creativecommons.org/licenses/by/4.0/\n\n"
            f"HOW THIS IS MADE\nNarration is a synthetic voice. Some imagery is generated or "
            f"drawn in code and is illustrative, not archival; real footage and photographs are "
            f"licensed stock and archive material, credited above, and show no one from the "
            f"story. No real people from the story are depicted.\n\n"
            f"#internetmysteries #unsolved #closedtabs")


def stage_package(q, b, out, report):
    meta = json.loads((q / "meta.json").read_text(encoding="utf-8"))
    lines = json.loads((b / "lines.json").read_text(encoding="utf-8"))
    report["subtitle_cues"] = subtitles(lines, out / "subs.srt")
    chapters, tags = chapters_and_tags(lines, meta["title"])
    hook_line = next((l["text"] for l in lines), meta["name"])
    credits = json.loads((b / "credits.json").read_text(encoding="utf-8")) \
        if (b / "credits.json").exists() else []
    desc = description(meta, chapters, report.get("music", []), hook_line, credits)
    import shutil
    if (b / "credits.json").exists():
        shutil.copy(b / "credits.json", out / "credits.json")
    (out / "metadata.json").write_text(json.dumps({
        "title": meta["title"], "description": desc, "tags": tags,
        "chapters": [[round(t, 1), n] for t, n in chapters],
    }, indent=1, ensure_ascii=False), encoding="utf-8")
    subprocess.run([sys.executable, str(ROOT / "automation" / "make_thumbnail.py"), q.name,
                    b.name], check=True)
    import shutil
    shutil.copy(b / "thumbs" / "thumbnail.jpg", out / "thumbnail.jpg")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("slug")
    ap.add_argument("--force-window", action="store_true")
    ap.add_argument("--skip-stills", action="store_true",
                    help="the stills are already in place (automation/resume_build.py)")
    ap.add_argument("--stage", choices=("voice", "stills", "part", "merge"),
                    help="run one stage of the split build (daily.yml) instead of all")
    ap.add_argument("--part", type=int, default=0)
    args = ap.parse_args()
    q = ROOT / "queue" / args.slug
    b = q / "build_full"
    out = q / "out"
    b.mkdir(parents=True, exist_ok=True)
    out.mkdir(exist_ok=True)
    if args.stage:
        return run_stage(args, q, b, out)
    report = {}
    stage_narrate(q, b, report)
    stage_plan(b, args.slug)
    stage_stock(b)
    # the stills stage reads build/ next to its plan, so it gets this folder
    if not args.skip_stills:
        stage_stills(b, args.force_window)
    script_text = (q / "script.txt").read_text(encoding="utf-8")
    plan = json.loads((b / "plan.json").read_text(encoding="utf-8"))
    r = edit.render(plan, b / "stills", b / "narration.wav", out / "video.mp4", b / "work",
                    script_text=script_text)
    report.update(r)
    stage_package(q, b, out, report)
    edit.write(report, out / "report.json")
    print(json.dumps({k: report[k] for k in ("checks", "problems") if k in report}, indent=1))
    return 0


def run_stage(args, q, b, out):
    """One job of the split build. Each stage reads what the stage before left in
    build_full/ (restored from that job's artifact) and leaves its own result."""
    from pipeline import parts
    if args.stage == "voice":
        report = {}
        stage_narrate(q, b, report)
        stage_plan(b, args.slug)
        stage_stock(b)
        (b / "voice_report.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
        return 0
    if args.stage == "stills":
        stage_stills(b, args.force_window)
        return 0
    plan = json.loads((b / "plan.json").read_text(encoding="utf-8"))
    if args.stage == "part":
        parts.render_part(plan, b / "stills", args.part, b / f"work_part{args.part}", b / "parts")
        return 0
    # merge: join, sound, package
    report = json.loads((b / "voice_report.json").read_text(encoding="utf-8")) \
        if (b / "voice_report.json").exists() else {}
    script_text = (q / "script.txt").read_text(encoding="utf-8")
    report.update(parts.merge(plan, b / "parts", b / "narration.wav", out / "video.mp4",
                              b / "work_merge", script_text=script_text))
    stage_package(q, b, out, report)
    edit.write(report, out / "report.json")
    print(json.dumps({k: report[k] for k in ("checks", "problems") if k in report}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
