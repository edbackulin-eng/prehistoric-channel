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
WORDS = int(CFG["length_min"] * WPM * 0.93)            # pauses take the rest of the 25 minutes
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
    """Section markers as the delivery markup reads them; a script that does not
    open with its hook gets one: its first lines, up to about 30 seconds."""
    out = []
    for raw in script.splitlines():
        m = re.match(r"^\s*\[\s*(?:section|abschnitt)?\s*[:=]?\s*([A-Za-zÄÖÜäöü]+)\s*\]\s*$", raw, re.I)
        if m and m.group(1).lower() in NAMES:
            out.append(f"[section {NAMES[m.group(1).lower()]}]")
        else:
            out.append(raw)
    text = "\n".join(out)
    first = next((l for l in text.splitlines() if l.strip()), "")
    if first != "[section hook]":
        body = [l for l in text.splitlines() if l.strip() != "[section hook]"]
        n, cut = 0, 0
        for i, l in enumerate(body):
            if l.startswith("["):
                if i:
                    break
                continue
            n += len(l.split())
            cut = i + 1
            if n >= 55:
                break
        rest = body[cut:]
        if not rest or not rest[0].startswith("[section"):
            rest = ["[section story]"] + rest
        text = "\n".join(["[section hook]"] + body[:cut] + rest)
    return text


def main(slug):
    q = json.loads((ROOT / "topics" / "queue.json").read_text(encoding="utf-8"))
    topic = next(t for t in q["topics"] if t["slug"] == slug)
    ep = ROOT / "episodes" / slug
    ep.mkdir(parents=True, exist_ok=True)
    src = sources(topic)
    block = _src_block(src)
    ans = gemini.ask(WRITE.format(title=topic["title"], focus=topic["focus"], words=WORDS,
                                  minutes=CFG["length_min"], sources=block), temperature=0.6)
    script = ans["script"].replace("\\n", "\n")
    for _ in range(2):
        if words(script) >= WORDS * 0.9:
            break
        script = gemini.ask(EXTEND.format(have=words(script), want=WORDS, script=script, sources=block),
                            temperature=0.5)["script"].replace("\\n", "\n")
    (ep / "script_draft.txt").write_text(script, encoding="utf-8")
    script, checks = fact_check(script, block)
    script = normalise(script)
    lines = delivery.parse(script)
    if not lines or lines[0].section != "hook":
        raise RuntimeError("the script has no hook section at its start")
    (ep / "script.txt").write_text(script.strip() + "\n", encoding="utf-8")
    (ep / "sources.json").write_text(json.dumps([{k: s[k] for k in ("lang", "title", "url")} for s in src],
                                                indent=1, ensure_ascii=False), encoding="utf-8")
    quellen = "\n".join(f"- {s['title']} (Wikipedia, {s['lang']}): {s['url']}" for s in src)
    meta = {"slug": slug, "title": ans["title_de"][:100], "tags": ans.get("tags", [])[:15],
            "description": f"{ans['description_de'].strip()}\n\nQuellen:\n{quellen}",
            "words": words(script), "fact_check": checks}
    (ep / "meta.json").write_text(json.dumps(meta, indent=1, ensure_ascii=False), encoding="utf-8")
    log.info("%s: %d words, %d lines, %d fact-check changes", slug, meta["words"], len(lines), len(checks))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
