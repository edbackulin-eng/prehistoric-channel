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
"""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from core import delivery, gemini  # noqa: E402
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
- Zeilen im Hook (section hook): 2-3 Einstellungen, nur Bewegtes (video oder flux), die stärksten Bilder der Folge.
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


PLAN_VERSION = 4          # 29.10: no abstract frames, no living animal drawn even when the planner named no species; a plan of an older version is made again
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
    animals = gemini.pick(gemini.ask(ANIMALS.format(script=delivery.plain_text(script)[:60000]), temperature=0.2),
                          "animals")
    animals = [a for a in animals if isinstance(a, dict) and a.get("key")]
    species = {a["key"]: {"name": f"{a.get('name_de')} ({a.get('latin')})", "features": (a.get("features") or [])[:5],
                          "forbidden": (a.get("forbidden") or [])[:5] + ["a modern animal in a prehistoric scene"],
                          "living": bool(a.get("living")), "video_query": a.get("video_query")} for a in animals}
    (ep / "species.json").write_text(json.dumps(species, indent=1, ensure_ascii=False), encoding="utf-8")
    living = {k for k, v in species.items() if v["living"]}
    alist = "\n".join(f"- {k}: {v['name']}, living={v['living']}; features: {'; '.join(v['features'])}"
                      for k, v in species.items())

    planned = {}
    for i0, part in chunks(lines, 40):
        numbered = "\n".join(f"{i0 + j} [{l.section}]: {l.text}" for j, l in enumerate(part))
        ans = gemini.ask(BEATS.format(amb=", ".join(AMB), animals=alist, lines=numbered), temperature=0.3)
        for row in gemini.pick(ans, "lines"):
            if isinstance(row, dict) and str(row.get("n", "")).isdigit():
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
    if len(order) < lo_anim + 2:
        numbered = "\n".join(f"{i} [{l.section}]: {l.text}" for i, l in enumerate(lines))
        extra = gemini.ask(EXTINCT.format(need=lo_anim + 2 - len(order), living=", ".join(sorted(living)),
                                          have=", ".join(order) or "nichts", lines=numbered[:60000]),
                           temperature=0.5)
        extra = gemini.pick(extra, "frames")
        by_n = {r["n"]: r for r in out_lines}
        for f in extra:
            fid = re.sub(r"[^a-z0-9_]", "_", str(f.get("id") or "").lower())[:40]
            n = int(f.get("n", -1)) if str(f.get("n", "")).lstrip("-").isdigit() else -1
            if not fid or fid in flux or n not in by_n or abstract(fid, f.get("prompt", "")) \
                    or living_named(fid, f.get("prompt", ""), species):
                continue
            flux[fid] = {"prompt": f["prompt"], "species": None, "motion": f.get("motion"),
                         "section": lines[n].section, "first": n, "extinct": f.get("species")}
            order.append(fid)
            by_n[n]["beats"].append({"kind": "recon", "id": fid, "species": None})
        log.info("extinct frames of the past added; now %d FLUX prompts", len(order))

    # at most flux_frames_max frames: the prompts that come first in the hook and climax win
    meta = json.loads((ep / "meta.json").read_text(encoding="utf-8")) if (ep / "meta.json").exists() else {}
    thumbs = gemini.pick(gemini.ask(THUMBS.format(title=meta.get("title", slug), animals=alist), temperature=0.7),
                         "thumbs")[:3]
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
    rep = {"version": PLAN_VERSION, "lines": len(lines), "species": len(species), "living": sorted(living), "flux_prompts": len(keep),
           "flux_frames": len(keep) * CFG["motion"]["flux_seeds"], "animate": animate,
           "video_beats": len(footage_json), "lines_without_picture": [r["n"] for r in out_lines if not r["beats"]]}
    (ep / "plan_report.json").write_text(json.dumps(rep, indent=1), encoding="utf-8")
    log.info("plan: %s", rep)
    return 0


def stale(slug):
    rp = ROOT / "episodes" / slug / "plan_report.json"
    return not rp.exists() or json.loads(rp.read_text(encoding="utf-8")).get("version") != PLAN_VERSION


if __name__ == "__main__":
    if sys.argv[1] == "--if-stale":
        sys.exit(main(sys.argv[2]) if stale(sys.argv[2]) else 0)
    sys.exit(main(sys.argv[1]))
