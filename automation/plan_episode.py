"""From the German script to what each line shows: shots, FLUX prompts, footage candidates.

    python automation/plan_episode.py <slug>

1. the animals of the episode (Gemini): for each, whether it lives today, the
   features that tell it apart and the traits that would betray a wrong animal
   -> episodes/<slug>/species.json, the checklist pipeline/speciescheck.py asks;
2. beats for every line (Gemini, in chunks): living animals are real VIDEO,
   always; extinct animals and lost landscapes are FLUX prompts with their
   anatomy spelled out; maps where places are named. The hook's lines get two or
   three short moving shots each;
3. the rules applied in code: at most config flux_frames_max FLUX frames (two
   seeds a prompt), the 8-10 strongest marked for LTX-Video animation (hook and
   climax first);
4. footage: for each video beat, candidates from Pixabay and Wikimedia Commons
   (PD / CC0 / CC BY only), the best first, up to three alternates; the species
   check at render time takes the first that is right;
-> shots.json, recon.json, footage.json, species.json, plan_report.json.

    python automation/plan_episode.py --standby <slug>

5. after the footage species check: a living animal whose passing real clips
   cannot fill 60 % of the time the narration tells of it gets FLUX/LTX
   reconstructions of that very species (owner 31.10), checked against its own
   checklist like every frame; the render shows them only where real video has
   run out (beats marked "standby"). Real video stays first. They come out of
   the episode's flux_frames_max (owner 02.11), in place of FLUX landscapes and
   extinct frames that are not animated - never on top of it.
"""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from core import delivery, gemini, textllm  # noqa: E402
from core.logger import get_logger  # noqa: E402

log = get_logger("plan")
CFG = json.loads((ROOT / "config" / "channel.json").read_text(encoding="utf-8"))
AMB = ["ocean", "beach", "forest", "swamp", "fire", "room", "harbour", "wind", "underwater"]
STYLE = ("photorealistic wildlife documentary frame, BBC natural history cinematography, natural vivid colours, "
         "soft natural light, high dynamic range, sharp detail, 35mm film look, no text, no people")

ANIMALS = """Hier ist der Sprechertext einer deutschen Urzeit-Doku. Nenne jedes Tier (oder jede Tiergruppe),
das im Text vorkommt und gezeigt werden soll. Antworte als JSON:
{{"animals": [{{"key": "<snake_case, englisch>", "name_de": "...", "latin": "...",
  "living": true|false,          // lebt die Art (oder eine nahe Art der Gruppe) heute noch?
  "features": ["<4-5 sichtbare Merkmale, die die Art von ähnlichen unterscheiden, englisch>"],
  "forbidden": ["<4 Merkmale, die ein falsches Tier verraten - z. B. Vogelkopf beim Flugsaurier, langer Hals beim Hadrosaurier, englisch>"],
  "video_query": "<englische Suchanfrage für echtes Stock-Video, nur wenn living>"}}]}}

TEXT:
{script}"""

BEATS = """Du planst die Bilder einer deutschen Urzeit-Doku (4K, ruhige BBC-Ästhetik). Für jede nummerierte
Zeile: was gezeigt wird. Regeln:
- Lebende Tiere (living=true in der Liste) NUR als echtes Video: {{"kind": "video", "species": "<key>",
  "query": "<englische Stock-Suche, 2-5 Wörter>", "dark": true|false}}.
- NIEMALS als FLUX-Bild: Diagramme, Infografiken, Stammbäume, DNA, Moleküle, Bildschirme, Schrift, Karten,
  abstrakte Visualisierungen - dafür echtes Stock-Video (kind "video").
- Ausgestorbene Tiere und verlorene Urzeit-Landschaften als FLUX-Bild: {{"kind": "flux", "species": "<key oder null>",
  "id": "<snake_case, eindeutig>", "prompt": "<englisch: das Tier mit genauer Anatomie aus der Merkmalliste, in seiner
  Umgebung, Kamerablick - niemals ein modernes Tier daneben, keine Menschen>", "motion": "<englisch, eine Bewegung für die
  Animation, z. B. 'swims slowly toward the camera'>"}}. Dieselbe id darf für eine spätere Zeile wiederkommen.
- Heutige Natur als Stellvertreter (Meer, Küste, Wald, Gestein, Tiefsee): {{"kind": "video", "species": null,
  "query": "<englisch>", "dark": true|false}}.
- Orte, die genannt werden: {{"kind": "map", "places": [{{"name": "<GROSS, deutsch>", "sub": "<Land/Region, deutsch>",
  "lat": <Zahl>, "lon": <Zahl>}}]}} - höchstens zwei Orte.
- DAS TIER IM BILD: erzählt eine Zeile von einem Tier (auch von seinem Fund, seinem Verhalten, seinen Verwandten),
  zeigen ALLE ihre Einstellungen dieses Tier (lebend: echtes Video; ausgestorben oder seine ausgestorbenen
  Verwandten: FLUX). Ein Kapitel über ein Tier zeigt es mindestens 60 % der Zeit.
- Meer, Himmel, Küste, Landschaft OHNE Tier: höchstens jede siebte Einstellung der ganzen Folge (<= 15 %).
- NIEMALS: DNA, Moleküle, Texturen (Sand, Schuppen-Makro, Papier), Büros, Archive, Menschen am Schreibtisch,
  Bildschirme, Schrift - statt dessen das Tier, um das es geht.
- In jeder Folge Urzeit-Tiere in Bewegung: mindestens 10 FLUX-Bilder ausgestorbener Tiere mit "motion", darunter
  mindestens 2 Dinosaurier; Bewegung geht vor Standbild. Etwa jedes dritte FLUX-Bild ein Wow-Moment: Jagd, Angriff,
  offenes Maul, das Tier frontal auf die Kamera zu, dramatisches Licht.
- Zeilen im Hook (section hook): 3 Einstellungen, nur Bewegtes MIT Tier (video eines Tiers des Themas oder flux
  eines ausgestorbenen Tiers mit "motion"), die stärksten Bilder der Folge.
- Sonst 1-2 Einstellungen pro Zeile. Abwechslung; kein Bild zweimal in Folge.
Dazu je Zeile "amb": eine von {amb}.
Antworte als JSON: {{"lines": [{{"n": <Nummer>, "amb": "...", "beats": [...]}}]}}

TIERE:
{animals}

ZEILEN:
{lines}"""


EXTINCT = """Diese deutsche Urzeit-Doku braucht mindestens {need} weitere Bilder der VERGANGENEN Welt (ausgestorbene
Tiere und verlorene Landschaften), die animiert werden. Wähle {need} Zeilen, die von der Vergangenheit erzählen
(Fossilien, frühere Erdzeitalter, Vorfahren und ausgestorbene Verwandte der heutigen Tiere, das Massenaussterben),
und schlage für jede ein FLUX-Bild vor: ein AUSGESTORBENES Tier (z. B. ausgestorbene Verwandte der Quastenflosser
wie Mawsonia oder Axelrodichthys, Ammoniten, Placodermen wie Dunkleosteus, Mosasaurier, Ichthyosaurier, Trilobiten,
Seeskorpione, frühe Pfeilschwanzkrebs-Verwandte, Dinosaurier) oder eine Urzeit-Landschaft, mit genauer Anatomie.
NIEMALS ein heute lebendes Tier ({living}), keine Diagramme, keine Schrift, keine Menschen.
Antworte als JSON: {{"frames": [{{"n": <Zeilennummer>, "id": "<snake_case, eindeutig>", "species": "<ausgestorbene Art, englisch>",
"prompt": "<englisch: Art, Anatomie, Umgebung, Licht, Kamerablick>", "motion": "<englisch, eine Bewegung>"}}]}}

SCHON GEPLANT: {have}

ZEILEN:
{lines}"""

THUMBS = """Plane drei YouTube-Vorschaubilder für die deutsche Urzeit-Doku "{title}". Vorbild sind die
erfolgreichsten Kanäle der Nische: KNALLIGE Farben, ein Tier groß im Bild (Kopf/Auge), sofort als urzeitlich
erkennbar, und ein Bild-Trick als Haken:
  "scale"      - ein winziger Taucher neben dem riesigen Tier, gelb eingekreist (Größenvergleich)
  "split"      - links das Fossil, rechts das lebende/rekonstruierte Tier
  "silhouette" - im Hintergrund die dunkle Silhouette eines T-Rex, das Tier groß davor
Jede Variante einen anderen Trick. Text: 2-3 deutsche Wörter, eines davon als Akzent (rot), wahr, neugierig.
Lebende Tiere (living=true) nur als echtes Foto ("base": "photo" mit englischer Bildsuche), ausgestorbene als
FLUX-Bild ("base": "flux" mit englischem Prompt: Kopf/Auge groß, knallige Farben, dramatisches Licht,
viel Raum für Text links oder rechts, keine Menschen, keine Schrift).
Antworte als JSON: {{"thumbs": [{{"id": "thumb_a|thumb_b|thumb_c", "hook": "scale|split|silhouette",
"animal": "<key>", "base": "photo|flux", "query": "<englisch, bei photo>", "prompt": "<englisch, bei flux>",
"fossil_query": "<englisch, Fossilfoto-Suche, nur bei split>", "words": ["Wort", "Wort"], "accent": <Index>}}]}}

TIERE:
{animals}"""


PLAN_VERSION = 7          # 31.10: the animal on screen (>= 60 % of its chapter), <= 15 % landscapes, a moving hook with animals; a lost chunk of lines asked again; older plans are made again
# FLUX draws animals and landscapes; a diagram, a DNA strand or a "visualisation" comes out as
# mush, and the species check turns it down. Such beats become stock footage instead.
ABSTRACT = re.compile(r"chart|diagram|graph|infograph|visuali[sz]ation|dna|helix|genetic|molecul|abstract|"
                      r"phylogen|schematic|screen|display|data|text|label|logo|map of", re.I)


def living_named(fid, prompt, species):
    """The living species a FLUX prompt or id names (by its English key words, Latin
    genus or German name), even when the planner gave the beat no species."""
    text = f"{fid} {prompt}".lower().replace("_", " ")
    for k, v in species.items():
        if not v.get("living"):
            continue
        toks = {t for t in k.lower().split("_") if len(t) > 3}
        toks |= {w for w in re.findall(r"[a-zäöüß]{5,}", v.get("name", "").lower())}
        if any(t in text for t in toks):
            return k
    return None


def abstract(fid, prompt):
    return bool(ABSTRACT.search(f"{fid} {prompt}"))


def chunks(seq, n):
    for i in range(0, len(seq), n):
        yield i, seq[i:i + n]


def footage_for(query, used, need_live, min_w=1280):
    from pipeline import footage
    cands = []
    try:
        cands += footage.pixabay(query, per_page=40)
    except Exception as e:                                               # noqa: BLE001
        log.warning("pixabay %r: %s", query, e)
    try:
        cands += footage.commons(query, limit=30)
    except Exception as e:                                               # noqa: BLE001
        log.warning("commons %r: %s", query, e)
    cands = [c for c in cands if c["id"] not in used and (c.get("width") or 0) >= min_w
             and (c.get("duration") or 0) >= 5 and (c.get("mime") or "video").startswith(("video", "application/ogg"))]
    # animals are rare on open licences: any resolution from 1280 up; the best first
    cands.sort(key=lambda c: (-(c["width"] >= 3840), -(c["width"] >= 1920), -min(c.get("duration") or 0, 30)))
    return cands[:4]


def main(slug):
    ep = ROOT / "episodes" / slug
    script = (ep / "script.txt").read_text(encoding="utf-8")
    lines = delivery.parse(script)
    # the animals, from the script in pieces a Groq request can carry, merged by key
    animals, plain = [], delivery.plain_text(script)
    for c0 in range(0, len(plain), 9000):
        for a in gemini.pick(textllm.ask(ANIMALS.format(script=plain[c0:c0 + 9000]), step="plan", temperature=0.2,
                                         max_tokens=2500), "animals"):
            if isinstance(a, dict) and a.get("key") and all(a["key"] != b.get("key") for b in animals):
                animals.append(a)
    animals = [a for a in animals if isinstance(a, dict) and a.get("key")]
    species = {a["key"]: {"name": f"{a.get('name_de')} ({a.get('latin')})", "features": (a.get("features") or [])[:5],
                          "forbidden": (a.get("forbidden") or [])[:5] + ["a modern animal in a prehistoric scene"],
                          "living": bool(a.get("living")), "video_query": a.get("video_query")} for a in animals}
    (ep / "species.json").write_text(json.dumps(species, indent=1, ensure_ascii=False), encoding="utf-8")
    living = {k for k, v in species.items() if v["living"]}
    alist = "\n".join(f"- {k}: {v['name']}, living={v['living']}; features: {'; '.join(v['features'])}"
                      for k, v in species.items())

    planned = {}
    for i0, part in chunks(lines, 20):
        numbered = "\n".join(f"{i0 + j} [{l.section}]: {l.text}" for j, l in enumerate(part))
        ans = textllm.ask(BEATS.format(amb=", ".join(AMB), animals=alist, lines=numbered), step="plan",
                          temperature=0.3, max_tokens=3000)
        for row in gemini.pick(ans, "lines"):
            if isinstance(row, dict) and str(row.get("n", "")).isdigit():
                planned[int(row["n"])] = row
    # 31.10: a chunk that came back empty left 21 lines without a picture; they are asked again, in smaller pieces
    for _ in range(2):
        lost = [i for i in range(len(lines)) if i not in planned]
        if not lost:
            break
        log.warning("%d lines came back without beats; asked again", len(lost))
        for k in range(0, len(lost), 10):
            numbered = "\n".join(f"{i} [{lines[i].section}]: {lines[i].text}" for i in lost[k:k + 10])
            try:
                ans = textllm.ask(BEATS.format(amb=", ".join(AMB), animals=alist, lines=numbered), step="plan",
                                  temperature=0.3, max_tokens=2500)
            except Exception as e:                                       # noqa: BLE001
                log.warning("beats again: %s", str(e)[:160])
                continue
            for row in gemini.pick(ans, "lines"):
                if isinstance(row, dict) and str(row.get("n", "")).isdigit() and int(row["n"]) in lost:
                    planned[int(row["n"])] = row
    # ---- the rules, in code
    flux, order = {}, []
    out_lines, footage_json, used = [], {}, set()
    for i, l in enumerate(lines):
        row = planned.get(i) or {"beats": [{"kind": "video", "species": None, "query": "ocean waves"}]}
        beats = []
        for k, b in enumerate(row.get("beats", [])[:3]):
            kind = b.get("kind")
            sp = b.get("species") if b.get("species") in species else None
            if kind == "flux" and sp in living:                  # a living animal is never drawn
                kind, b = "video", {"kind": "video", "species": sp, "query": species[sp]["video_query"] or sp}
            elif kind == "flux" and living_named(b.get("id") or "", b.get("prompt") or "", species):
                sp = living_named(b.get("id") or "", b.get("prompt") or "", species)
                kind, b = "video", {"kind": "video", "species": sp, "query": species[sp]["video_query"] or sp}
            elif kind == "flux" and abstract(b.get("id") or "", b.get("prompt") or ""):
                kind, b = "video", {"kind": "video", "species": None, "dark": True,
                                    "query": (b.get("id") or "deep sea").replace("_", " ")[:40]}
            if kind == "flux":
                fid = re.sub(r"[^a-z0-9_]", "_", (b.get("id") or f"f{i}_{k}").lower())[:40]
                if fid not in flux:
                    flux[fid] = {"prompt": b.get("prompt", ""), "species": sp, "motion": b.get("motion"),
                                 "section": l.section, "first": i}
                    order.append(fid)
                beats.append({"kind": "recon", "id": fid, "species": sp})
            elif kind == "map" and b.get("places"):
                beats.append({"kind": "anim", "anim": "map", "places": b["places"][:2]})
            else:
                need = f"v{i:03d}_{k}"
                q = b.get("query") or "ocean"
                cands = footage_for(q, used, sp in living)
                if not cands and sp in living and species[sp].get("video_query"):
                    cands = footage_for(species[sp]["video_query"], used, True)
                if not cands:
                    continue
                main = cands[0]
                used.add(main["id"])
                footage_json[need] = {**main, "alternates": [{kk: c.get(kk) for kk in (
                    "id", "url", "width", "height", "duration", "page", "author", "license", "license_url",
                    "source", "title", "tags")} for c in cands[1:]], "query": q}
                beats.append({"kind": "real", "need": need, "species": sp, "dark": bool(b.get("dark")),
                              "topic": q, "allow": CFG["species_check"]["allow"]})
        if not beats:
            beats = [{"kind": "anim", "anim": "map", "places": []}] if False else []
        out_lines.append({"at": " ".join(l.text.split()[:4]), "n": i, "section": l.section,
                          "amb": row.get("amb") if row.get("amb") in AMB else "wind",
                          **({"cut": True} if i == 0 or l.section != lines[i - 1].section else {}), "beats": beats})

    # 8-10 animated frames an episode (config motion.animated_per_episode): an episode about
    # living animals plans few FLUX frames of its own (29.10: three), so the past is added -
    # extinct relatives, earlier seas and coasts - on the lines that tell of it
    lo_anim = CFG["motion"]["animated_per_episode"][0]
    # the lines that tell of the past, short enough for one request
    past = re.compile(r"Millionen|Fossil|ausgestorb|Urzeit|Kreide|Devon|Perm|Trias|Jura|Kambri|Ordovizi|Silur|"
                      r"Karbon|Vorfahr|Massenausster|Erdzeitalter|versteinert|Dinosaur", re.I)
    numbered = "\n".join(f"{i} [{l.section}]: {l.text}" for i, l in enumerate(lines) if past.search(l.text))[:9000]
    by_n = {r["n"]: r for r in out_lines}
    for _ in range(3):                         # rounds, until there are enough
        if len(order) >= lo_anim + 2:
            break
        extra = gemini.pick(textllm.ask(EXTINCT.format(need=lo_anim + 2 - len(order), living=", ".join(sorted(living)),
                                                       have=", ".join(order) or "nichts", lines=numbered),
                                        step="plan", temperature=0.6, max_tokens=2500), "frames")
        for f in extra:
            if not isinstance(f, dict):
                continue
            fid = re.sub(r"[^a-z0-9_]", "_", str(f.get("id") or "").lower())[:40]
            n = int(f.get("n", -1)) if str(f.get("n", "")).lstrip("-").isdigit() else -1
            prompt, extinct = f.get("prompt", ""), str(f.get("species") or "")
            if not fid or fid in flux or n not in by_n or abstract(fid, prompt):
                continue
            # an extinct relative may name its group ("an extinct coelacanth, Mawsonia");
            # what may not be drawn is a living species itself
            if living_named(extinct, "", species) if extinct else living_named(fid, prompt, species):
                continue
            flux[fid] = {"prompt": prompt, "species": None, "motion": f.get("motion"),
                         "section": lines[n].section, "first": n, "extinct": extinct or None}
            order.append(fid)
            by_n[n]["beats"].append({"kind": "recon", "id": fid, "species": None})
        log.info("extinct frames of the past added; now %d FLUX prompts", len(order))

    # at most flux_frames_max frames: the prompts that come first in the hook and climax win
    meta = json.loads((ep / "meta.json").read_text(encoding="utf-8")) if (ep / "meta.json").exists() else {}
    thumbs = gemini.pick(textllm.ask(THUMBS.format(title=meta.get("title", slug), animals=alist), step="plan",
                                     temperature=0.7, max_tokens=1500), "thumbs")[:3]
    flux_thumbs = {t["id"]: t["prompt"] for t in thumbs if t.get("base") == "flux" and t.get("prompt")}
    # the thumbnails' FLUX bases (two seeds each) come out of the same 60 frames
    max_prompts = (CFG["motion"]["flux_frames_max"] - 2 * len(flux_thumbs)) // CFG["motion"]["flux_seeds"]
    prio = {"hook": 0, "climax": 1, "story": 2, "finale": 3}
    keep = sorted(order, key=lambda f: (prio.get(flux[f]["section"], 2), flux[f]["first"]))[:max_prompts]
    for row in out_lines:
        row["beats"] = [b for b in row["beats"] if b["kind"] != "recon" or b["id"] in keep]
    lo, hi = CFG["motion"]["animated_per_episode"]
    animate = [f for f in sorted(keep, key=lambda f: (prio.get(flux[f]["section"], 2), flux[f]["first"]))
               if flux[f]["species"] or flux[f].get("extinct")][:hi]
    if len(animate) < lo:
        animate += [f for f in keep if f not in animate][:lo - len(animate)]
    recon = {"_note": "FLUX prompts of this episode, written by automation/plan_episode.py; extinct animals and lost "
                      "landscapes only. 'animate' are the frames LTX-Video brings to life.",
             "style": STYLE, "seeds_per_prompt": CFG["motion"]["flux_seeds"],
             "prompts": {f: flux[f]["prompt"] for f in keep}, "species": {f: flux[f]["species"] for f in keep},
             "animate": {f: flux[f]["motion"] or "slow natural movement, gentle camera push in" for f in animate},
             "thumbs": flux_thumbs, "photos": {}, "two_seeds": []}
    (ep / "thumbs.json").write_text(json.dumps(thumbs, indent=1, ensure_ascii=False), encoding="utf-8")
    (ep / "recon.json").write_text(json.dumps(recon, indent=1, ensure_ascii=False), encoding="utf-8")
    (ep / "footage.json").write_text(json.dumps(footage_json, indent=1, ensure_ascii=False), encoding="utf-8")
    (ep / "shots.json").write_text(json.dumps({"_note": "Planned by automation/plan_episode.py from script.txt.",
                                               "lines": out_lines}, indent=1, ensure_ascii=False), encoding="utf-8")
    import hashlib
    rep = {"version": PLAN_VERSION, "script_sha": hashlib.sha1(script.encode("utf-8")).hexdigest()[:12],
           "lines": len(lines), "species": len(species), "living": sorted(living), "flux_prompts": len(keep),
           "flux_frames": len(keep) * CFG["motion"]["flux_seeds"], "animate": animate,
           "video_beats": len(footage_json), "lines_without_picture": [r["n"] for r in out_lines if not r["beats"]]}
    (ep / "plan_report.json").write_text(json.dumps(rep, indent=1), encoding="utf-8")
    log.info("plan: %s", rep)
    return 0


STANDBY = """Für eine deutsche Doku über lebende Fossilien fehlt echtes Video von {name}. Schreibe {n} verschiedene
englische FLUX-Prompts, die dieses HEUTE LEBENDE Tier fotorealistisch in seinem heutigen Lebensraum zeigen
(wie ein Tierfilm der BBC). Jeder Prompt: das Tier groß und vollständig erkennbar, mit genau diesen Merkmalen:
{features}. Niemals: {forbidden}; keine Menschen, keine Schrift, kein anderes Tier im Vordergrund, nichts Urzeitliches.
Jeder Prompt ein anderer Bildausschnitt: Kopf und Auge groß, ganzer Körper von der Seite, frontal auf die Kamera zu,
von unten, im Scheinwerferlicht eines Tauchboots, an einer Höhle oder Felswand, die Flossen/Beine in Bewegung.
Dazu je eine Bewegung für die Animation (englisch, z. B. "glides slowly toward the camera, fins paddling").
Was die Folge über das Tier erzählt (für Lebensraum, Tiefe, Licht):
{context}
Antworte als JSON {{"frames": [{{"prompt": "<englisch>", "motion": "<englisch>"}}]}}"""
VIEWS = ["close-up of the head and eye", "the whole body seen from the side", "swimming toward the camera",
         "seen from below against faint light", "lit by a submersible's lamp in the dark", "beside a rocky wall",
         "its paired fins moving", "three-quarter view, gliding"]
STANDBY_MARGIN = 0.65          # a little over the QA's 60 %: some seeds fail their species check
STANDBY_PASS = 0.75            # the share of seeds expected to pass it


def standby(slug):
    """Living animals the footage cannot show (owner 31.10): FLUX/LTX reconstructions of
    the species where its passing real clips fall short of the QA's 60 % - see the module
    docstring, step 5. Runs after the footage species check, before FLUX; once a plan
    (again when a plan of the stand-ins is over the frame cap)."""
    from collections import defaultdict
    import math
    from pipeline import docplan, speciescheck
    ep = ROOT / "episodes" / slug
    recon = json.loads((ep / "recon.json").read_text(encoding="utf-8"))
    spec = json.loads((ep / "shots.json").read_text(encoding="utf-8"))
    seeds = CFG["motion"]["flux_seeds"]
    cap = CFG["motion"]["flux_frames_max"]

    def frames_of():
        return len(recon["prompts"]) * seeds + 2 * len(recon.get("thumbs") or {})

    def drop(fid):
        for k in ("prompts", "species", "animate"):
            (recon.get(k) or {}).pop(fid, None)
        for e in spec["lines"]:
            e["beats"] = [b for b in e["beats"] if not (b["kind"] == "recon" and b.get("id") == fid)]
    if "standby" in recon:
        if frames_of() <= cap:
            log.info("%s: stand-ins already planned (%d)", slug, len(recon["standby"]))
            return 0
        # 02.11: stand-ins planned on top of the 60 frames (82); they are planned again inside them
        log.warning("%s: %d FLUX frames, over the %d an episode: stand-ins planned again", slug, frames_of(), cap)
        for fid in list(recon.pop("standby")):
            drop(fid)
    species = json.loads((ep / "species.json").read_text(encoding="utf-8"))
    footage = json.loads((ep / "footage.json").read_text(encoding="utf-8"))
    vpath = ep / "species_verdicts.json"
    known = json.loads(vpath.read_text(encoding="utf-8")) if vpath.exists() else {}
    meta = json.loads((ep / "meta.json").read_text(encoding="utf-8")) if (ep / "meta.json").exists() else {}
    lines = delivery.parse((ep / "script.txt").read_text(encoding="utf-8"))
    words = sum(len(l.text.split()) for l in lines)
    per_word = (meta.get("minutes_est") or words / 130) * 60 / max(1, words)
    living = {k for k, v in species.items() if v.get("living")}
    told, real, where = defaultdict(float), defaultdict(set), defaultdict(list)
    for e in spec["lines"]:
        n = e.get("n")
        for sp in {b["species"] for b in e["beats"] if b.get("species") in living}:
            told[sp] += len(lines[n].text.split()) * per_word
            where[sp].append(n)
        for b in e["beats"]:
            if b["kind"] != "real" or b.get("species") not in living or b["need"] not in footage:
                continue
            main = footage[b["need"]]
            for c in [main] + list(main.get("alternates") or []):
                if c["id"] in known and speciescheck.passes(known[c["id"]], b["species"], allow=b.get("allow") or ())[0]:
                    real[b["species"]].add(c["id"])
    per_prompt = seeds * docplan.REPEAT_MAX * docplan.TARGET_S * STANDBY_PASS
    want = {}
    for sp, t in told.items():
        short = STANDBY_MARGIN * t - len(real[sp]) * docplan.REPEAT_MAX * docplan.TARGET_S
        if t >= 30 and short > 0:
            want[sp] = math.ceil(short / per_prompt)
    # the stand-ins come out of the episode's flux_frames_max, not on top of them (owner 02.11): they
    # take the places of FLUX landscapes first, then of extinct frames LTX does not animate; the
    # animated extinct animals stay (the hook and the moving prehistory need them)
    replaced = []
    if sum(want.values()) * seeds > cap - frames_of():
        land = lambda f: docplan.is_landscape({"kind": "recon", "id": f, "topic": recon["prompts"][f],
                                                    "species": recon["species"].get(f)})  # noqa: E731
        anim = recon.get("animate") or {}
        order = ([f for f in recon["prompts"] if land(f) and f not in anim] +
                 [f for f in recon["prompts"] if land(f) and f in anim] +
                 [f for f in reversed(list(recon["prompts"])) if not land(f) and f not in anim])
        for f in order:
            if sum(want.values()) * seeds <= cap - frames_of():
                break
            drop(f)
            replaced.append(f)
    room = max(0, cap - frames_of()) // seeds
    if sum(want.values()) > room:
        k = room / sum(want.values())
        want = {sp: int(n * k) for sp, n in want.items()}
        for sp in sorted(want, key=lambda s: -told[s]):           # what is left over: the most told first
            if sum(want.values()) < room:
                want[sp] += 1
        want = {sp: n for sp, n in want.items() if n}
    report = {sp: {"told_s": round(told[sp]), "real_clips_passing": len(real[sp]), "prompts": want.get(sp, 0)}
              for sp in told}
    report["_frames"] = {"cap": cap, "replaced": replaced}
    log.info("%s: living animals, real video against the time told: %s", slug, report)
    recon["standby"] = {}
    rows = {e.get("n"): e for e in spec["lines"]}
    std = []
    for sp, n in want.items():
        v = species[sp]
        context = " ".join(lines[i].text for i in where[sp])[:2500]
        frames = []
        try:
            frames = gemini.pick(textllm.ask(STANDBY.format(name=v["name"], n=n, features="; ".join(v["features"]),
                                                            forbidden="; ".join(v["forbidden"]), context=context),
                                             step="plan", temperature=0.7, max_tokens=min(6000, 300 + 160 * n)),
                                 "frames")
        except Exception as e:                                           # noqa: BLE001
            log.warning("%s: stand-in prompts from the template (%s)", sp, str(e)[:160])
        frames = [f for f in frames if isinstance(f, dict) and f.get("prompt") and not abstract("", f["prompt"])][:n]
        for k in range(len(frames), n):                     # the template, when the model gave fewer
            frames.append({"prompt": f"{v['name']}, a living animal today, {', '.join(v['features'])}, "
                                     f"{VIEWS[k % len(VIEWS)]}, in its natural habitat",
                           "motion": "moves slowly and naturally, gentle camera push in"})
        for k, f in enumerate(frames):
            fid = f"standby_{sp}_{k:02d}"[:40]
            recon["prompts"][fid] = f["prompt"]
            recon["species"][fid] = sp
            recon["standby"][fid] = sp
            line = where[sp][min(len(where[sp]) - 1, k * len(where[sp]) // len(frames))]
            rows[line]["beats"].append({"kind": "recon", "id": fid, "species": sp, "standby": True})
            std.append((fid, f.get("motion") or "moves slowly and naturally, gentle camera push in"))
    # LTX: the animals first - extinct ones and the stand-ins share the clips; landscapes last
    hi = CFG["motion"]["animated_per_episode"][1]
    old = list((recon.get("animate") or {}).items())
    land = [(f, m) for f, m in old if docplan.is_landscape({"kind": "recon", "id": f, "topic": recon["prompts"].get(f),
                                                                    "species": recon["species"].get(f)})]
    animals = [x for x in old if x not in land]
    take_std = min(len(std), max(hi // 2, hi - len(animals)))
    recon["animate"] = dict((animals[:hi - take_std] + std[:take_std] + land)[:hi])
    recon["standby_report"] = report
    (ep / "recon.json").write_text(json.dumps(recon, indent=1, ensure_ascii=False), encoding="utf-8")
    (ep / "shots.json").write_text(json.dumps(spec, indent=1, ensure_ascii=False), encoding="utf-8")
    log.info("%s: %d stand-in prompts of living animals; animated: %s", slug, len(std), list(recon["animate"]))
    return 0


DINO = re.compile(r"dinosaur|tyrannosaur|t\.? ?rex\b|triceratops|edmontosaur|hadrosaur|sauropod|brachiosaur|"
                  r"diplodoc|stegosaur|ankylosaur|raptor|spinosaur|allosaur|iguanodon|parasaurolophus|theropod|"
                  r"ceratops|titanosaur|carnotaur|giganotosaur|argentinosaur|apatosaur", re.I)
NOT_DINO = re.compile(r"(ichthyo|mosa|plesio|ptero|plio|nothosaur)saur\w*", re.I)
# a tense picture - one kind of wow moment besides the moving ones (owner 02.11)
TENSE = re.compile(r"hunt|attack|strik|\bbit(e|es|ing)\b|jaws|teeth|chas|lung(e|es|ing)|roar|charg|predator|ambush|"
                   r"collaps|dying|fight|clash|stampede|flee|toward the camera|facing .{0,24}camera|close-up of the head|"
                   r"spotlight|lamp|dramatic|tense|menac", re.I)
DINO_LINE = re.compile(r"Dinosaur|Kreide|Asteroid|66 Millionen|Massenausster|Einschlag", re.I)
DINO_FRAMES = [
    ("cretaceous_trex_stride", "A Tyrannosaurus rex striding through a misty Late Cretaceous conifer forest, massive head "
     "with forward-facing eyes, short two-fingered arms, heavy tail held straight out behind, pebbly grey-brown skin, "
     "low camera angle", "strides forward and turns its head toward the camera, jaws slightly open"),
    ("cretaceous_triceratops_herd", "A small herd of Triceratops at the edge of a Late Cretaceous river, three facial "
     "horns, a broad solid bony frill, bulky four-legged bodies, morning haze, eye-level wildlife shot",
     "the herd walks slowly forward, heads lowered, dust rising"),
    ("cretaceous_edmontosaurus_flee", "A group of Edmontosaurus, duck-billed hadrosaurs without crests, running across "
     "a Late Cretaceous floodplain under a darkening sky, long heavy tails held straight out",
     "the animals run across the frame, tails swinging"),
]
DINO_MIN = 2


def is_dino(fid, prompt):
    return bool(DINO.search(NOT_DINO.sub("", f"{fid} {prompt}".replace("_", " "))))


def motion(slug):
    """Motion first inside the 60 FLUX frames and 10 LTX clips (owner 02.11): at least
    DINO_MIN dinosaurs among the animated frames - a dinosaur frame the plan has, or a new
    one on a line about the dinosaurs' time, in the place of a still stand-in, its LTX slot
    from a landscape or a non-dinosaur extinct animal - and the wow pictures marked
    (recon.json "wow": every animated frame, every tense FLUX animal). Once a plan."""
    from pipeline import docplan
    ep = ROOT / "episodes" / slug
    recon = json.loads((ep / "recon.json").read_text(encoding="utf-8"))
    spec = json.loads((ep / "shots.json").read_text(encoding="utf-8"))
    lines = delivery.parse((ep / "script.txt").read_text(encoding="utf-8"))
    seeds, cap = CFG["motion"]["flux_seeds"], CFG["motion"]["flux_frames_max"]
    hi = CFG["motion"]["animated_per_episode"][1]
    anim = recon.setdefault("animate", {})
    prompts = recon["prompts"]
    land = lambda f: docplan.is_landscape({"kind": "recon", "id": f, "topic": prompts.get(f),
                                                  "species": recon["species"].get(f)})   # noqa: E731

    def frames_of():
        return len(prompts) * seeds + 2 * len(recon.get("thumbs") or {})
    have = [f for f in anim if is_dino(f, prompts.get(f, ""))]
    added = []
    if len(have) < DINO_MIN:
        spare = [f for f in prompts if f not in anim and is_dino(f, prompts[f])]
        used_lines = {b.get("id"): e.get("n") for e in spec["lines"] for b in e["beats"]}
        dino_lines = [i for i, l in enumerate(lines) if DINO_LINE.search(l.text)] or list(range(len(lines)))
        for fid, prompt, mot in DINO_FRAMES:
            if len(have) + len(spare) >= DINO_MIN:
                break
            if fid in prompts:
                continue
            if frames_of() + seeds > cap:            # room: a still stand-in first, then a still extinct frame
                victims = [f for f in reversed(list(prompts)) if f not in anim and f in (recon.get("standby") or {})] + \
                          [f for f in reversed(list(prompts)) if f not in anim and not is_dino(f, prompts[f])]
                if not victims:
                    break
                v = victims[0]
                for k in ("prompts", "species", "standby"):
                    (recon.get(k) or {}).pop(v, None)
                for e in spec["lines"]:
                    e["beats"] = [b for b in e["beats"] if not (b["kind"] == "recon" and b.get("id") == v)]
            prompts[fid] = prompt
            recon["species"][fid] = None
            n = dino_lines[len(added) % len(dino_lines)]
            row = next(e for e in spec["lines"] if e.get("n") == n)
            row["beats"].append({"kind": "recon", "id": fid, "species": None})
            spare.append(fid)
            added.append(fid)
            recon.setdefault("motions", {})[fid] = mot
        for fid in spare[:DINO_MIN - len(have)]:
            if len(anim) >= hi:                       # its LTX slot: a landscape, then a non-dinosaur extinct animal
                give = [f for f in reversed(list(anim)) if land(f)] + \
                       [f for f in reversed(list(anim)) if not is_dino(f, prompts.get(f, ""))
                        and f not in (recon.get("standby") or {})] + \
                       [f for f in reversed(list(anim)) if not is_dino(f, prompts.get(f, ""))]
                if not give:
                    break
                anim.pop(give[0])
            anim[fid] = (recon.get("motions") or {}).get(fid) or "moves slowly and naturally, gentle camera push in"
    recon["wow"] = sorted(set(anim) | {f for f in prompts if not land(f) and TENSE.search(
        f"{prompts[f]} {anim.get(f, '')} {f.replace('_', ' ')}")})
    recon["motion_report"] = {"dinosaurs_animated": [f for f in anim if is_dino(f, prompts.get(f, ""))],
                              "dinosaur_frames_added": added, "wow_pictures": len(recon["wow"]),
                              "frames": frames_of(), "animated": len(anim)}
    (ep / "recon.json").write_text(json.dumps(recon, indent=1, ensure_ascii=False), encoding="utf-8")
    (ep / "shots.json").write_text(json.dumps(spec, indent=1, ensure_ascii=False), encoding="utf-8")
    log.info("%s: motion pass %s", slug, recon["motion_report"])
    return 0


def stale(slug):
    """A plan of an older planner, or made for another script than the one there now."""
    import hashlib
    rp = ROOT / "episodes" / slug / "plan_report.json"
    if not rp.exists():
        return True
    r = json.loads(rp.read_text(encoding="utf-8"))
    sha = hashlib.sha1((ROOT / "episodes" / slug / "script.txt").read_bytes().decode("utf-8").encode("utf-8")
                       ).hexdigest()[:12]
    return r.get("version") != PLAN_VERSION or ("script_sha" in r and r["script_sha"] != sha)


if __name__ == "__main__":
    if sys.argv[1] == "--if-stale":
        sys.exit(main(sys.argv[2]) if stale(sys.argv[2]) else 0)
    if sys.argv[1] == "--standby":
        sys.exit(standby(sys.argv[2]) or motion(sys.argv[2]))
    sys.exit(main(sys.argv[1]))
