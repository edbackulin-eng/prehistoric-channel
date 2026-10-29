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
from pipeline import youtube
from core import delivery, gemini, textllm  # noqa: E402
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
  "tags": ["<10-15 deutsche Tags zum Thema; KEINE Sender-, Sendungs-, Film- oder Kanalnamen>"],
  "script": "<der ganze Text im Markup, Zeilen mit \\n getrennt>"}}

QUELLEN:
{sources}"""

OUTLINE = """Plane eine deutsche Urzeit-Doku für YouTube (Stil Terra X: ruhig, präzise, packend; kein Clickbait im
Text, kein "KI"). Thema: "{title}" - {focus}. Gesamtlänge {words} Wörter (= {minutes} Minuten).
Gliederung: ein Hook (65 Wörter: der stärkste Moment oder das größte Rätsel der Folge, eine Frage, die hängen
bleibt, ein Versprechen), dann 4-5 Kapitel "story", ein Kapitel "climax" (die überraschendste Erkenntnis),
ein Kapitel "finale" (ruhiger Ausklang, zurück zur Frage aus dem Hook). Die Wortzahlen der Kapitel ergeben
zusammen {words}. Nur Inhalte, die die Quellen tragen. Jede Begebenheit wird GENAU EINMAL erzählt, im Kapitel,
in das sie gehört - kein Kapitel wiederholt, was ein anderes schon erzählt hat. Plane mindestens zwei offene
Fragen, die erst in einem späteren Kapitel beantwortet werden.
Antworte als JSON:
{{"title_de": "<YouTube-Titel, max 70 Zeichen, neugierig, wahr>", "description_de": "<2-4 Sätze>",
  "tags": ["<10-15 deutsche Tags zum Thema; KEINE Sender-, Sendungs-, Film- oder Kanalnamen>"],
  "chapters": [{{"section": "hook|story|climax|finale", "heading": "<worum es geht>",
                 "points": ["<Fakten und Momente aus den Quellen, die dieses Kapitel erzählt>"], "words": <Zahl>}}]}}

QUELLEN:
{sources}"""

PART = """Schreibe ein Kapitel des Sprechertexts einer deutschen Urzeit-Doku (Stil Terra X: ruhig, präzise,
bildhaft, packend; keine Floskeln, kein "KI", keine Anrede). Thema der Folge: "{title}".
ERZÄHLEN, NICHT AUFZÄHLEN: eine Dokumentation erzählt eine Geschichte, sie reiht keine Fakten.
- Szenen: ein Ort, eine Zeit, ein Mensch, der etwas tut oder sieht - der Zuschauer ist dabei.
- Figuren: Forscher, Fischer, Entdecker mit dem, was sie wollten, fürchteten, nicht wussten.
- Spannung: was auf dem Spiel steht; was noch niemand ahnt.
- Übergänge: jede Zeile führt in die nächste ("Doch...", "Was er nicht wusste...", "Erst Jahre später...").
- Offene Schleifen: eine Frage stellen, deren Antwort erst später kommt ("Wie konnte dieser Fisch überleben? Die
  Antwort lag in 200 Metern Tiefe - aber dazu später.").
- Rhetorische Fragen, sparsam, an den Wendepunkten.
- Rhythmus: kurze Sätze für Wucht. Längere, fließende Sätze für Bilder und Zusammenhänge. Abwechseln.
- Sauberes, natürliches Deutsch.

Dieses Kapitel ({section}): {heading}. Es erzählt: {points}.
Länge: {words} Wörter - nicht weniger. Nur Fakten aus den Quellen. Zahlen dürfen als Ziffern stehen.
Markup, eine Zeile pro Satzgruppe von 1-3 Sätzen: "> " schnelle Zeile, "< " langsame Zeile mit Pause davor
(Enthüllungen), sonst normale Zeile; [pause 0.8] sparsam. Keine Überschriften.
So endet das Kapitel davor (nahtlos anschließen, nichts wiederholen):
{before}
Antworte als JSON {{"lines": ["<Zeile>", "..."]}}

QUELLEN:
{sources}"""

EDIT = """Du bist Redakteur einer deutschen Natur-Doku (Terra X). Überarbeite die nummerierten Zeilen des
Sprechertexts so, dass sie erzählen statt aufzählen. Regeln:
ERZÄHLEN, NICHT AUFZÄHLEN: eine Dokumentation erzählt eine Geschichte, sie reiht keine Fakten.
- Szenen: ein Ort, eine Zeit, ein Mensch, der etwas tut oder sieht - der Zuschauer ist dabei.
- Figuren: Forscher, Fischer, Entdecker mit dem, was sie wollten, fürchteten, nicht wussten.
- Spannung: was auf dem Spiel steht; was noch niemand ahnt.
- Übergänge: jede Zeile führt in die nächste ("Doch...", "Was er nicht wusste...", "Erst Jahre später...").
- Offene Schleifen: eine Frage stellen, deren Antwort erst später kommt ("Wie konnte dieser Fisch überleben? Die
  Antwort lag in 200 Metern Tiefe - aber dazu später.").
- Rhetorische Fragen, sparsam, an den Wendepunkten.
- Rhythmus: kurze Sätze für Wucht. Längere, fließende Sätze für Bilder und Zusammenhänge. Abwechseln.
- Sauberes, natürliches Deutsch.
- Gleich viele Zeilen, in derselben Reihenfolge: Zeile n bleibt Zeile n und behält ihr Thema (zu jeder Zeile
  sind schon Bilder geplant). Sätze innerhalb einer Zeile dürfen neu gebaut werden.
- KEINE Tatsache ändern, keine neue hinzufügen, keine weglassen: Namen, Zahlen, Orte, Daten bleiben exakt.
- Das Markup-Präfix jeder Zeile ("> ", "< " oder keines) bleibt; Zeilen wie [pause 0.8] bleiben unverändert.
- Schon gute Zeilen dürfen bleiben, wie sie sind.
Antworte als JSON {{"lines": ["<Zeile 1>", "..."]}} mit genau {n} Zeilen.

So geht es davor weiter: {before}

ZEILEN:
{lines}"""

MORE = """Setze dieses Kapitel einer deutschen Urzeit-Doku fort ({section}: {heading}). Es soll noch etwa
{words} Wörter mehr werden, mit weiteren Fakten und Momenten AUS DEN QUELLEN zu: {points}. Nichts wiederholen,
was schon gesagt ist; selbes Markup ("> ", "< ", normale Zeilen), selber Ton.
Bisher:
{so_far}
Antworte als JSON {{"lines": ["<neue Zeile>", "..."]}}

QUELLEN:
{sources}"""

CHECK = """Prüfe jede nummerierte Zeile dieses deutschen Doku-Texts gegen die Quellen.
Eine Zeile ist "ok", wenn jede Tatsachenbehauptung darin von den Quellen gedeckt ist. Erzählerisches ist ok:
Szenen, Übergänge, rhetorische Fragen, offene Fragen, Spannung, Bilder - solange sie keine neue Tatsache
behaupten. Ändere NIE den Stil oder die Grammatik einer gedeckten Zeile; nur eine falsche oder ungedeckte
Tatsache wird korrigiert, und dann im selben erzählenden Ton. Sonst "fix": gib eine korrigierte Fassung, die nur Gedecktes sagt, im selben
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


def fact_check(script, src, rounds=2, chunk=18, keep_count=False):
    """Every line against the sources, in chunks of lines, each chunk with the
    source paragraphs that concern it."""
    log_rows = []
    for r in range(rounds):
        raw = [l for l in script.splitlines() if l.strip()]
        bad = {}
        for c0 in range(0, len(raw), chunk):
            part = raw[c0:c0 + chunk]
            numbered = "\n".join(f"{c0 + j}: {l}" for j, l in enumerate(part))
            ans = textllm.ask(CHECK.format(lines=numbered, sources=textllm.relevant(src, " ".join(part), 9000)),
                              step="fact_check", temperature=0.1, max_tokens=2000)
            for x in gemini.pick(ans, "lines"):
                if isinstance(x, dict) and str(x.get("n", "")).isdigit():
                    bad[int(x["n"])] = x
        out = []
        for i, l in enumerate(raw):
            x = bad.get(i)
            if x and not l.startswith("["):
                log_rows.append({"round": r, "n": i, "was": l, **x})
                if x.get("verdict") == "drop" and not keep_count:
                    continue
                if x.get("verdict") == "fix" and x.get("text"):
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


def hook_text(script):
    return [l.text for l in delivery.parse(script) if l.section == "hook"]


def _gist(body, limit=5000):
    """What a hook is written from: the story's opening and its climax, short
    enough for one request."""
    rows = [l for l in body.splitlines() if l.strip() and not l.startswith("[")]
    climax = []
    sec = None
    for l in body.splitlines():
        if l.startswith("[section"):
            sec = l
        elif sec == "[section climax]":
            climax.append(l)
    return ("\n".join(rows[:25]))[: limit // 2] + "\n...\n" + ("\n".join(climax))[: limit // 2]


def hook_sha(script):
    import hashlib
    return hashlib.sha1(" ".join(t.strip() for t in hook_text(script)).encode("utf-8")).hexdigest()[:12]


def hook_ok(script, block):
    """The hook the script has, judged by the rule: strongest moment, a question,
    a promise, facts the sources carry, 40-85 words. (A marker alone proves
    nothing: 29.10 a hook with no question and a copy of itself in the story
    passed.)"""
    h = hook_text(script)
    if not h:
        return False, None
    v = textllm.ask(HOOK_CHECK.format(hook="\n".join(h), sources=textllm.relevant(block, " ".join(h), 8000)),
                    step="hook", temperature=0.1, max_tokens=600)
    n = len(" ".join(h).split())
    ok = all(v.get(k) for k in ("strongest_moment", "question", "promise", "facts_supported")) and 40 <= n <= 85
    return ok, {"hook": h, "check": v, "words": n}


def strip_hook(script):
    """The script without its hook section - and without the hook's lines wherever
    else the model repeated them."""
    rows, out, section = script.splitlines(), [], None
    gone = {t.strip() for t in hook_text(script)}
    for l in rows:
        m = re.match(r"^\[section (\w+)\]$", l.strip())
        if m:
            section = m.group(1)
            if section != "hook":
                out.append(l)
            continue
        if section == "hook":
            continue
        t = re.sub(r"^[<>]\s*", "", l.strip())
        if t in gone:
            continue
        out.append(l)
    return "\n".join(out)


def write_hook(script, block, tries=4):
    """A hook by the rule, written on its own and checked before it opens the episode."""
    body = strip_hook(script)
    last = None
    for _ in range(tries):
        gist = _gist(body)
        hook = [l.strip() for l in gemini.pick(textllm.ask(HOOK.format(script=gist, sources=textllm.relevant(
            block, gist, 6000)), step="hook", temperature=0.7, max_tokens=800), "hook") if l.strip()]
        v = textllm.ask(HOOK_CHECK.format(hook="\n".join(hook), sources=textllm.relevant(block, " ".join(hook), 8000)),
                        step="hook", temperature=0.1, max_tokens=600)
        n = len(" ".join(hook).split())
        last = {"hook": hook, "check": v, "words": n}
        if all(v.get(k) for k in ("strongest_moment", "question", "promise", "facts_supported")) and 40 <= n <= 85:
            first = next((l for l in body.splitlines() if l.strip()), "")
            rest = body if first.startswith("[section") else "[section story]\n" + body
            log.info("hook written and checked: %s", v)
            return "[section hook]\n" + "\n".join(hook) + "\n" + rest, last
        log.info("hook rejected: %s", v)
    raise RuntimeError(f"no hook passed its check: {last}")


def ensure_hook(slug):
    """For a script already in the repository: judge its hook, replace it if it fails."""
    q = json.loads((ROOT / "topics" / "queue.json").read_text(encoding="utf-8"))
    topic = next(t for t in q["topics"] if t["slug"] == slug)
    ep = ROOT / "episodes" / slug
    script = (ep / "script.txt").read_text(encoding="utf-8")
    meta = json.loads((ep / "meta.json").read_text(encoding="utf-8"))
    if meta.get("hook_sha") == hook_sha(script):          # this very hook has passed its check
        log.info("%s: hook already checked", slug)
        return 0
    block = sources(topic)
    ok, rep = hook_ok(script, block)
    if ok:
        log.info("%s: hook passes (%s)", slug, rep["check"])
        meta["hook_sha"] = hook_sha(script)
        (ep / "meta.json").write_text(json.dumps(meta, indent=1, ensure_ascii=False), encoding="utf-8")
        return 0
    script, rep = write_hook(script, block)
    (ep / "script.txt").write_text(script.strip() + "\n", encoding="utf-8")
    meta["hook_written_separately"] = rep
    meta["hook_sha"] = hook_sha(script)
    meta["words"] = words(script)
    meta["minutes_est"] = round(words(script) / WPM, 1)
    (ep / "meta.json").write_text(json.dumps(meta, indent=1, ensure_ascii=False), encoding="utf-8")
    # the plan was made for the old script: it is made again
    (ep / "plan_report.json").unlink(missing_ok=True)
    log.info("%s: hook replaced; %d words", slug, meta["words"])
    return 0


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
            script = textllm.ask(SHORTEN.format(have=n, minutes=n / WPM, want=WORDS, script=script,
                                                sources=textllm.relevant(block, script[:3000], 12000)),
                                 step="script", temperature=0.4, big=True)["script"].replace("\\n", "\n")
        else:
            # added to the story where it ends, not rewritten whole: a whole
            # rewrite comes back cut at the model's answer length
            rows = script.splitlines()
            end = max((i for i, l in enumerate(rows) if not l.startswith("[")
                       and _section_at(rows, i) == "story"), default=len(rows) - 1)
            so_far = "\n".join(rows[max(0, end - 12):end + 1])
            more = textllm.ask(MORE.format(section="story", heading="weitere Fakten und Momente",
                                           words=min(500, WORDS - n), points="was die Quellen noch erzählen",
                                           so_far=so_far, sources=textllm.relevant(block, so_far, 7000)),
                               step="script", temperature=0.6, max_tokens=1600)
            add = [l.strip() for l in gemini.pick(more, "lines") if l.strip() and l.strip() not in rows]
            script = "\n".join(rows[:end + 1] + add + rows[end + 1:])
        log.info("length: %d -> %d words", n, words(script))
    n = words(script)
    if not MIN_WORDS <= n <= MAX_WORDS:
        raise RuntimeError(f"script is {n} words ({n / WPM:.1f} min); it must be {MIN_WORDS}-{MAX_WORDS}")
    return script


def edit(lines, before="(Anfang)", step="editor"):
    """The editor's pass over lines: told, not listed; same number of lines, same
    facts. An answer of another length keeps the lines as they were."""
    if not lines:
        return lines
    numbered = "\n".join(f"{i + 1}: {l}" for i, l in enumerate(lines))
    try:
        ans = textllm.ask(EDIT.format(n=len(lines), before=before[-400:], lines=numbered), step=step,
                          temperature=0.6, max_tokens=min(4500, 300 + 150 * len(lines)))
        out = [re.sub(r"^\d+:\s*", "", str(l)).strip() for l in gemini.pick(ans, "lines")]
    except Exception as e:                                               # noqa: BLE001
        log.warning("editor pass failed (%s); lines kept", str(e)[:120])
        return lines
    if len(out) != len(lines) or any(not l for l in out):
        log.warning("editor gave %d lines for %d; kept as they were", len(out), len(lines))
        return lines
    # a [pause] or [section] line stays exactly what it was
    return [a if a.startswith("[") else b for a, b in zip(lines, out)]


def restyle(slug, chunk=10):
    """An existing script, told instead of listed: every line after the hook goes
    through the editor, line for line, so the plan made for it stays right;
    then the fact check, which may correct a line but not remove one."""
    q = json.loads((ROOT / "topics" / "queue.json").read_text(encoding="utf-8"))
    topic = next(t for t in q["topics"] if t["slug"] == slug)
    ep = ROOT / "episodes" / slug
    rows = (ep / "script.txt").read_text(encoding="utf-8").splitlines()
    hook_end = next((i for i, l in enumerate(rows) if l.strip().startswith("[section") and "hook" not in l and i), 0)
    head, body = rows[:hook_end], rows[hook_end:]
    out, before = [], "\n".join(head[-2:])
    for c0 in range(0, len(body), chunk):
        part = body[c0:c0 + chunk]
        idx = [i for i, l in enumerate(part) if l.strip() and not l.strip().startswith("[")]
        new = edit([part[i] for i in idx], before)
        for i, l in zip(idx, new):
            part[i] = l
        out += part
        before = "\n".join([l for l in part if not l.startswith("[")][-2:])
        log.info("restyled lines %d-%d", c0 + hook_end, c0 + hook_end + len(part) - 1)
    script = "\n".join(head + out)
    n_before = len(delivery.parse("\n".join(rows)))
    script, checks = fact_check(script, sources(topic), keep_count=True)
    lines = delivery.parse(script)
    if len(lines) != n_before:
        raise RuntimeError(f"restyle changed the number of lines ({n_before} -> {len(lines)}); not kept")
    (ep / "script.txt").write_text(script.strip() + "\n", encoding="utf-8")
    meta = json.loads((ep / "meta.json").read_text(encoding="utf-8"))
    meta.update({"words": words(script), "minutes_est": round(words(script) / WPM, 1), "restyled": True,
                 "restyle_fact_check": checks})
    (ep / "meta.json").write_text(json.dumps(meta, indent=1, ensure_ascii=False), encoding="utf-8")
    log.info("%s restyled: %d lines, %d words (%.1f min), %d fact-check changes", slug, len(lines), meta["words"],
             meta["minutes_est"], len(checks))
    return 0


def rewrite_body(slug):
    """The script after its hook written anew, chapter by chapter in the told
    style, each chapter through the editor; then length and fact check. The hook
    stays word for word (its check is remembered by hook_sha)."""
    q = json.loads((ROOT / "topics" / "queue.json").read_text(encoding="utf-8"))
    topic = next(t for t in q["topics"] if t["slug"] == slug)
    ep = ROOT / "episodes" / slug
    old = (ep / "script.txt").read_text(encoding="utf-8")
    rows = old.splitlines()
    hook_end = next((i for i, l in enumerate(rows) if l.strip().startswith("[section") and "hook" not in l and i),
                    len(rows))
    head = rows[:hook_end]
    src = sources(topic)
    hook_words = len(" ".join(hook_text(old)).split())
    plan, body = write_by_chapters({**topic, "focus": topic["focus"] + ". Der Hook ist schon geschrieben: "
                                    + " ".join(hook_text(old))}, src, words=WORDS - hook_words, skip_hook=True)
    script = normalise("\n".join(head) + "\n" + body)
    script = normalise(fit_length(script, src))
    script, checks = fact_check(script, src)
    script = fit_length(normalise(script), src)
    if hook_text(script) != hook_text(old):
        raise RuntimeError("the hook changed; not kept")
    (ep / "script.txt").write_text(script.strip() + "\n", encoding="utf-8")
    meta = json.loads((ep / "meta.json").read_text(encoding="utf-8"))
    meta.update({"words": words(script), "minutes_est": round(words(script) / WPM, 1), "restyled": "rewritten",
                 "rewrite_fact_check": checks, "chapters": [c.get("heading") for c in gemini.pick(plan, "chapters")]})
    (ep / "meta.json").write_text(json.dumps(meta, indent=1, ensure_ascii=False), encoding="utf-8")
    (ep / "plan_report.json").unlink(missing_ok=True)      # the lines are new: the plan is made again
    log.info("%s rewritten: %d lines, %d words (%.1f min), %d fact-check changes", slug,
             len(delivery.parse(script)), meta["words"], meta["minutes_est"], len(checks))
    return 0


def write_by_chapters(topic, block, words=None, skip_hook=False):
    """The script chapter by chapter, each to its word budget: one long answer
    came out 463 words long when the model fell back (29.10)."""
    words = words or WORDS
    plan = textllm.ask(OUTLINE.format(title=topic["title"], focus=topic["focus"], words=words,
                                      minutes=CFG["length_min"],
                                      sources=textllm.relevant(block, f"{topic['title']} {topic['focus']}", 10000)),
                       step="script", temperature=0.5, max_tokens=2000)
    chapters = [c for c in gemini.pick(plan, "chapters") if isinstance(c, dict)
                and not (skip_hook and c.get("section") == "hook")]
    total = sum(int(c.get("words") or 0) for c in chapters) or words
    out, before, section = [], "(Anfang der Folge)", None
    for c in chapters:
        want = max(50, int(int(c.get("words") or 0) * words / total))
        topic_q = f"{c['heading']} {' '.join(c.get('points', []))}"
        ans = textllm.ask(PART.format(title=topic["title"], section=c["section"], heading=c["heading"],
                                      points="; ".join(c.get("points", [])), words=min(want, 450),
                                      before=before, sources=textllm.relevant(block, topic_q, 7500)),
                          step="script", temperature=0.6, max_tokens=1700)
        lines = [l.strip() for l in gemini.pick(ans, "lines") if l.strip()]
        # a lighter model writes 150-300 words an answer: the chapter is
        # continued until it reaches its budget
        for _ in range(5):
            have = len(" ".join(lines).split())
            if have >= 0.9 * want:
                break
            more = textllm.ask(MORE.format(section=c["section"], heading=c["heading"], words=min(450, want - have),
                                           points="; ".join(c.get("points", [])), so_far="\n".join(lines[-12:]),
                                           sources=textllm.relevant(block, topic_q, 7000)),
                               step="script", temperature=0.6, max_tokens=1600)
            add = [l.strip() for l in gemini.pick(more, "lines") if l.strip() and l.strip() not in lines]
            if not add:
                break
            lines += add
        lines = edit(lines, before)                       # the editor: told, not listed
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
    block = src
    ans, script = write_by_chapters(topic, block)
    script = normalise(script)
    (ep / "script_draft.txt").write_text(script, encoding="utf-8")
    script = normalise(fit_length(script, block))
    script, checks = fact_check(script, block)
    script = normalise(script)
    ok, hook_check = hook_ok(script, block)
    if not ok:
        script, hook_check = write_hook(script, block)
    script = fit_length(script, block)                # the fact check may have shortened it
    lines = delivery.parse(script)
    if not lines or lines[0].section != "hook":
        raise RuntimeError("the script has no hook section at its start")
    (ep / "script.txt").write_text(script.strip() + "\n", encoding="utf-8")
    (ep / "sources.json").write_text(json.dumps([{k: s[k] for k in ("lang", "title", "url")} for s in src],
                                                indent=1, ensure_ascii=False), encoding="utf-8")
    quellen = "\n".join(f"- {s['title']} (Wikipedia, {s['lang']}): {s['url']}" for s in src)
    meta = {"slug": slug, "title": ans["title_de"][:100], "tags": youtube.own_tags(ans.get("tags", []))[:15],
            "description": f"{ans['description_de'].strip()}\n\nQuellen:\n{quellen}",
            "words": words(script), "minutes_est": round(words(script) / WPM, 1), "fact_check": checks,
            "hook_written_separately": hook_check, "hook_sha": hook_sha(script)}
    (ep / "meta.json").write_text(json.dumps(meta, indent=1, ensure_ascii=False), encoding="utf-8")
    log.info("%s: %d words, %d lines, %d fact-check changes", slug, meta["words"], len(lines), len(checks))
    return 0


if __name__ == "__main__":
    if sys.argv[1] == "--restyle":
        sys.exit(restyle(sys.argv[2]))
    if sys.argv[1] == "--rewrite":
        sys.exit(rewrite_body(sys.argv[2]))
    sys.exit(ensure_hook(sys.argv[2]) if sys.argv[1] == "--hook-only" else main(sys.argv[1]))
