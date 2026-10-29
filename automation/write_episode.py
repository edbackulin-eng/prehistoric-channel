"""A German 25-minute script for a topic of the queue, from Wikipedia, fact-checked.

    python automation/write_episode.py <slug>

1. sources: the topic's Wikipedia articles (de and en), as plain text;
2. writing (Gemini, this channel's key): German documentary narration in the
   delivery markup (core/delivery.py), in four sections - [section hook] about
   30 s: the episode's strongest moment or riddle, a question that holds, a
   promise of what the viewer will see; then story, climax, finale. Only facts
   the sources carry;
3. fact check: every line is checked against the sources; a line that is not
   supported is corrected from them or removed; twice;
4. length: about 25 minutes at the channel's pace (core/narrate_de.py), the
   story is extended from the sources if it came out short;
5. episodes/<slug>/script.txt, sources.json and meta.json (German title,
   description with the sources, tags).
"""
import json
import re
import sys
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from core import delivery, gemini  # noqa: E402
from core.logger import get_logger  # noqa: E402

log = get_logger("write")
CFG = json.loads((ROOT / "config" / "channel.json").read_text(encoding="utf-8"))
WPM = CFG["voice"]["target_wpm"]
# The narration is calibrated to WPM over the whole programme, pauses included
# (core/narrate_de.py), so minutes are words / WPM: 25 min is 3200 words, and the
# script must land within 22-27 min - longer costs GPU frames and render time.
WORDS = int(CFG["length_min"] * WPM)
MIN_WORDS, MAX_WORDS = int(22 * WPM), int(27 * WPM)
UA = {"User-Agent": "prehistoric-channel/1.0 (documentary research; vladgula557@gmail.com)"}

WRITE = """Du schreibst den Sprechertext für eine deutsche Natur- und Urzeit-Dokumentation auf YouTube
(Stil: ruhige, präzise, packende Doku-Sprache wie Terra X; kein Clickbait im Text, keine Floskeln,
kein "KI", keine Anrede "Leute"). Thema: "{title}" - {focus}.

Länge: etwa {words} Wörter (das sind rund {minutes} Minuten). Verwende NUR Fakten aus den Quellen unten.
Zahlen dürfen als Ziffern stehen. Keine Überschriften, keine Regieanweisungen außer dem Markup.

Markup, eine Zeile pro Satzgruppe von 1-3 Sätzen:
  [section hook] | [section story] | [section climax] | [section finale]  - je einmal, in dieser Reihenfolge
  > schnelle, sichere Zeile (Aufzählungen, Übergänge)
  < langsame Zeile mit Pause davor (Enthüllungen, Schlüsselfakten)
  normale Zeile ohne Zeichen
  [pause 0.8]  eine Pause in Sekunden, sparsam

Aufbau:
- [section hook]: höchstens 30 Sekunden (ca. 60 Wörter): der stärkste Moment oder das größte Rätsel der
  Folge, eine Frage, die hängen bleibt, und ein Versprechen, was der Zuschauer erfahren wird.
- [section story]: die Geschichte, chronologisch oder als Spurensuche, mit Orten, Jahreszahlen, Namen.
- [section climax]: der Höhepunkt - die überraschendste Erkenntnis, das Drama, die Auflösung des Rätsels.
- [section finale]: ein ruhiger Ausklang mit dem Bogen zurück zur Frage aus dem Hook.

Antworte als JSON:
{{"title_de": "<YouTube-Titel, max 70 Zeichen, neugierig machend, wahr>",
  "description_de": "<2-4 Sätze Beschreibung ohne Quellenliste>",
  "tags": ["<10-15 deutsche Tags>"],
  "script": "<der ganze Text im Markup, Zeilen mit \\n getrennt>"}}

QUELLEN:
{sources}"""

OUTLINE = """Plane eine deutsche Urzeit-Doku für YouTube (Stil Terra X: ruhig, präzise, packend; kein Clickbait im
Text, kein "KI"). Thema: "{title}" - {focus}. Gesamtlänge {words} Wörter (= {minutes} Minuten).
Gliederung: ein Hook (65 Wörter: der stärkste Moment oder das größte Rätsel der Folge, eine Frage, die hängen
bleibt, ein Versprechen), dann 4-5 Kapitel "story", ein Kapitel "climax" (die überraschendste Erkenntnis),
ein Kapitel "finale" (ruhiger Ausklang, zurück zur Frage aus dem Hook). Die Wortzahlen der Kapitel ergeben
zusammen {words}. Nur Inhalte, die die Quellen tragen.
Antworte als JSON:
{{"title_de": "<YouTube-Titel, max 70 Zeichen, neugierig, wahr>", "description_de": "<2-4 Sätze>",
  "tags": ["<10-15 deutsche Tags>"],
  "chapters": [{{"section": "hook|story|climax|finale", "heading": "<worum es geht>",
                 "points": ["<Fakten und Momente aus den Quellen, die dieses Kapitel erzählt>"], "words": <Zahl>}}]}}

QUELLEN:
{sources}"""

PART = """Schreibe ein Kapitel des Sprechertexts einer deutschen Urzeit-Doku (Stil Terra X: ruhig, präzise,
bildhaft, packend; keine Floskeln, kein "KI", keine Anrede). Thema der Folge: "{title}".
Dieses Kapitel ({section}): {heading}. Es erzählt: {points}.
Länge: {words} Wörter - nicht weniger. Nur Fakten aus den Quellen. Zahlen dürfen als Ziffern stehen.
Markup, eine Zeile pro Satzgruppe von 1-3 Sätzen: "> " schnelle Zeile, "< " langsame Zeile mit Pause davor
(Enthüllungen), sonst normale Zeile; [pause 0.8] sparsam. Keine Überschriften.
So endet das Kapitel davor (nahtlos anschließen, nichts wiederholen):
{before}
Antworte als JSON {{"lines": ["<Zeile>", "..."]}}

QUELLEN:
{sources}"""

MORE = """Setze dieses Kapitel einer deutschen Urzeit-Doku fort ({section}: {heading}). Es soll noch etwa
{words} Wörter mehr werden, mit weiteren Fakten und Momenten AUS DEN QUELLEN zu: {points}. Nichts wiederholen,
was schon gesagt ist; selbes Markup ("> ", "< ", normale Zeilen), selber Ton.
Bisher:
{so_far}
Antworte als JSON {{"lines": ["<neue Zeile>", "..."]}}

QUELLEN:
{sources}"""

CHECK = """Prüfe jede nummerierte Zeile dieses deutschen Doku-Texts gegen die Quellen.
Eine Zeile ist "ok", wenn jede Tatsachenbehauptung darin von den Quellen gedeckt ist (stilistische Sätze
ohne Behauptung sind ok). Sonst "fix": gib eine korrigierte Fassung, die nur Gedecktes sagt, im selben
Ton und Markup-Präfix (>, < oder nichts) - oder "drop", wenn sich nichts retten lässt.
Antworte als JSON: {{"lines": [{{"n": <Nummer>, "verdict": "ok|fix|drop", "text": "<neue Zeile bei fix>",
"why": "<kurz>"}}]}} - nur Zeilen, die nicht ok sind.

TEXT:
{lines}

QUELLEN:
{sources}"""

EXTEND = """Der folgende deutsche Doku-Text ist {have} Wörter lang, er soll etwa {want} haben.
Erweitere den Teil [section story] (und wo es passt [section climax]) mit weiteren Fakten AUS DEN QUELLEN,
im selben Stil und Markup. Hook und Finale bleiben, wie sie sind. Antworte als JSON {{"script": "..."}}.

TEXT:
{script}

QUELLEN:
{sources}"""


SHORTEN = """Der folgende deutsche Doku-Text ist {have} Wörter lang (etwa {minutes:.0f} Minuten), er darf höchstens
{want} Wörter haben (25 Minuten). Kürze ihn auf etwa {want} Wörter: verdichte [section story] und
[section climax], streiche Wiederholungen und Nebensächliches, behalte alle Schlüsselfakten, den roten Faden,
Hook und Finale. Nur Fakten aus den Quellen, selbes Markup. Antworte als JSON {{"script": "..."}}.

TEXT:
{script}

QUELLEN:
{sources}"""

HOOK = """Schreibe den Hook (die ersten 30 Sekunden, 55-70 Wörter) für diese deutsche Urzeit-Doku. Regel:
1. der stärkste Moment oder das größte Rätsel der ganzen Folge, konkret und bildhaft,
2. eine Frage, die den Zuschauer nicht loslässt,
3. ein Versprechen, was er in dieser Folge erfahren wird.
Nur Fakten aus den Quellen. 2-4 Zeilen im Markup (> schnell, < langsam mit Pause davor, sonst normal).
Antworte als JSON {{"hook": ["<Zeile>", "..."]}}

DER TEXT DER FOLGE:
{script}

QUELLEN:
{sources}"""

HOOK_CHECK = """Prüfe diesen Hook einer deutschen Urzeit-Doku (die ersten 30 Sekunden):
{hook}
Antworte als JSON {{"strongest_moment": true|false, "question": true|false, "promise": true|false,
"facts_supported": true|false, "words": <Anzahl>, "note": "<kurz>"}}
"facts_supported": jede Behauptung ist von den Quellen gedeckt.

QUELLEN:
{sources}"""


def wiki(lang, title, limit=14000):
    q = urllib.parse.urlencode({"action": "query", "prop": "extracts|info", "explaintext": 1, "redirects": 1,
                                "inprop": "url", "titles": title, "format": "json"})
    d = json.loads(urllib.request.urlopen(urllib.request.Request(
        f"https://{lang}.wikipedia.org/w/api.php?{q}", headers=UA), timeout=60).read())
    page = next(iter(d["query"]["pages"].values()))
    text = page.get("extract") or ""
    text = re.sub(r"\n==+ (Literatur|Weblinks|Einzelnachweise|References|External links|See also|Further reading)"
                  r" ==+[\s\S]*", "", text)
    return {"lang": lang, "title": page.get("title", title), "url": page.get("fullurl"), "text": text[:limit]}


def sources(topic):
    out = []
    for lang in ("de", "en"):
        for t in topic["articles"].get(lang, []):
            try:
                s = wiki(lang, t)
                if len(s["text"]) > 500:
                    out.append(s)
            except Exception as e:                                       # noqa: BLE001
                log.warning("wikipedia %s:%s: %s", lang, t, e)
    if not out:
        raise RuntimeError("no sources could be read")
    return out


def _src_block(src, total=90000):
    per = max(4000, total // len(src))
    return "\n\n".join(f"### {s['title']} ({s['lang']}.wikipedia.org)\n{s['text'][:per]}" for s in src)


def words(script):
    return len(delivery.plain_text(script).split())


def fact_check(script, src_block, rounds=2):
    log_rows = []
    for r in range(rounds):
        raw = [l for l in script.splitlines() if l.strip()]
        numbered = "\n".join(f"{i}: {l}" for i, l in enumerate(raw))
        ans = gemini.ask(CHECK.format(lines=numbered, sources=src_block), temperature=0.1)
        bad = {int(x["n"]): x for x in ans.get("lines", []) if str(x.get("n", "")).isdigit()}
        out = []
        for i, l in enumerate(raw):
            x = bad.get(i)
            if x and not l.startswith("["):
                log_rows.append({"round": r, "n": i, "was": l, **x})
                if x["verdict"] == "drop":
                    continue
                if x["verdict"] == "fix" and x.get("text"):
                    l = x["text"].strip()
            out.append(l)
        script = "\n".join(out)
        log.info("fact check round %d: %d lines changed or dropped", r + 1, len(bad))
        if not bad:
            break
    return script, log_rows


NAMES = {"hook": "hook", "einstieg": "hook", "story": "story", "geschichte": "story", "climax": "climax",
         "höhepunkt": "climax", "hoehepunkt": "climax", "finale": "finale", "schluss": "finale", "ausklang": "finale"}


def normalise(script):
    """Section markers as the delivery markup reads them ([section hook] ...)."""
    out = []
    for raw in script.splitlines():
        m = re.match(r"^\s*\[\s*(?:section|abschnitt)?\s*[:=]?\s*([A-Za-zÄÖÜäöü]+)\s*\]\s*$", raw, re.I)
        if m and m.group(1).lower() in NAMES:
            out.append(f"[section {NAMES[m.group(1).lower()]}]")
        else:
            out.append(raw)
    return "\n".join(out)


def has_hook(script):
    lines = delivery.parse(script)
    return bool(lines) and lines[0].section == "hook"


def write_hook(script, block, tries=3):
    """A hook by the rule, written on its own and checked before it opens the episode."""
    body = "\n".join(l for l in script.splitlines() if l.strip() != "[section hook]")
    last = None
    for _ in range(tries):
        hook = [l.strip() for l in gemini.ask(HOOK.format(script=body[:40000], sources=block),
                                              temperature=0.7)["hook"] if l.strip()]
        v = gemini.ask(HOOK_CHECK.format(hook="\n".join(hook), sources=block), temperature=0.1)
        n = len(" ".join(hook).split())
        last = {"hook": hook, "check": v, "words": n}
        if all(v.get(k) for k in ("strongest_moment", "question", "promise", "facts_supported")) and 40 <= n <= 85:
            first = next((l for l in body.splitlines() if l.strip()), "")
            rest = body if first.startswith("[section") else "[section story]\n" + body
            log.info("hook written and checked: %s", v)
            return "[section hook]\n" + "\n".join(hook) + "\n" + rest, last
        log.info("hook rejected: %s", v)
    raise RuntimeError(f"no hook passed its check: {last}")


def _section_at(rows, i):
    sec = "story"
    for l in rows[:i + 1]:
        m = re.match(r"^\[section (\w+)\]$", l.strip())
        if m:
            sec = m.group(1)
    return sec


def fit_length(script, block):
    """22-27 minutes: a long script is condensed, a short one extended from the sources."""
    for _ in range(6):
        n = words(script)
        if MIN_WORDS <= n <= MAX_WORDS:
            return script
        if n > MAX_WORDS:
            script = gemini.ask(SHORTEN.format(have=n, minutes=n / WPM, want=WORDS, script=script, sources=block),
                                temperature=0.4)["script"].replace("\\n", "\n")
        else:
            # added to the story where it ends, not rewritten whole: a whole
            # rewrite comes back cut at the model's answer length
            rows = script.splitlines()
            end = max((i for i, l in enumerate(rows) if not l.startswith("[")
                       and _section_at(rows, i) == "story"), default=len(rows) - 1)
            more = gemini.ask(MORE.format(section="story", heading="weitere Fakten und Momente", words=WORDS - n,
                                          points="was die Quellen noch erzählen", so_far="\n".join(rows[max(0, end - 25):end + 1]),
                                          sources=block), temperature=0.6)
            add = [l.strip() for l in more.get("lines", []) if l.strip() and l.strip() not in rows]
            script = "\n".join(rows[:end + 1] + add + rows[end + 1:])
        log.info("length: %d -> %d words", n, words(script))
    n = words(script)
    if not MIN_WORDS <= n <= MAX_WORDS:
        raise RuntimeError(f"script is {n} words ({n / WPM:.1f} min); it must be {MIN_WORDS}-{MAX_WORDS}")
    return script


def write_by_chapters(topic, block):
    """The script chapter by chapter, each to its word budget: one long answer
    came out 463 words long when the model fell back (29.10)."""
    plan = gemini.ask(OUTLINE.format(title=topic["title"], focus=topic["focus"], words=WORDS,
                                     minutes=CFG["length_min"], sources=block), temperature=0.5)
    chapters = plan["chapters"]
    total = sum(int(c.get("words") or 0) for c in chapters) or WORDS
    out, before, section = [], "(Anfang der Folge)", None
    for c in chapters:
        want = max(50, int(int(c.get("words") or 0) * WORDS / total))
        ans = gemini.ask(PART.format(title=topic["title"], section=c["section"], heading=c["heading"],
                                     points="; ".join(c.get("points", [])), words=want,
                                     before=before, sources=block), temperature=0.6)
        lines = [l.strip() for l in ans.get("lines", []) if l.strip()]
        # a lighter model writes 150-300 words an answer: the chapter is
        # continued until it reaches its budget
        for _ in range(5):
            have = len(" ".join(lines).split())
            if have >= 0.9 * want:
                break
            more = gemini.ask(MORE.format(section=c["section"], heading=c["heading"], words=want - have,
                                          points="; ".join(c.get("points", [])), so_far="\n".join(lines),
                                          sources=block), temperature=0.6)
            add = [l.strip() for l in more.get("lines", []) if l.strip() and l.strip() not in lines]
            if not add:
                break
            lines += add
        if c["section"] != section:
            out.append(f"[section {c['section']}]")
            section = c["section"]
        out += lines
        before = "\n".join(lines[-3:])
        log.info("chapter %s/%s: %d words (wanted %d)", c["section"], c["heading"][:40],
                 len(" ".join(lines).split()), want)
    return plan, "\n".join(out)


def main(slug):
    q = json.loads((ROOT / "topics" / "queue.json").read_text(encoding="utf-8"))
    topic = next(t for t in q["topics"] if t["slug"] == slug)
    ep = ROOT / "episodes" / slug
    ep.mkdir(parents=True, exist_ok=True)
    src = sources(topic)
    block = _src_block(src)
    ans, script = write_by_chapters(topic, block)
    script = normalise(script)
    (ep / "script_draft.txt").write_text(script, encoding="utf-8")
    script = normalise(fit_length(script, block))
    script, checks = fact_check(script, block)
    script = normalise(script)
    hook_check = None
    if not has_hook(script):
        script, hook_check = write_hook(script, block)
    script = fit_length(script, block)                # the fact check may have shortened it
    lines = delivery.parse(script)
    if not lines or lines[0].section != "hook":
        raise RuntimeError("the script has no hook section at its start")
    (ep / "script.txt").write_text(script.strip() + "\n", encoding="utf-8")
    (ep / "sources.json").write_text(json.dumps([{k: s[k] for k in ("lang", "title", "url")} for s in src],
                                                indent=1, ensure_ascii=False), encoding="utf-8")
    quellen = "\n".join(f"- {s['title']} (Wikipedia, {s['lang']}): {s['url']}" for s in src)
    meta = {"slug": slug, "title": ans["title_de"][:100], "tags": ans.get("tags", [])[:15],
            "description": f"{ans['description_de'].strip()}\n\nQuellen:\n{quellen}",
            "words": words(script), "minutes_est": round(words(script) / WPM, 1), "fact_check": checks,
            "hook_written_separately": hook_check}
    (ep / "meta.json").write_text(json.dumps(meta, indent=1, ensure_ascii=False), encoding="utf-8")
    log.info("%s: %d words, %d lines, %d fact-check changes", slug, meta["words"], len(lines), len(checks))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
