"""Is it the right animal, and does the picture belong here? Asked of every shot.

For each beat of shots.json that shows a picture - real footage, a real photo,
a FLUX reconstruction - Gemini sees the candidate (three frames of a clip, or
the still) with the narration line it sits under, and answers two questions:

  1. species: is this really <species>? A checklist of the features that tell
     the species apart (a coelacanth's three-lobed tail with its small middle
     lobe and fleshy lobed fins; a horseshoe crab's smooth horseshoe shell and
     straight tail spine, no claws; a tuatara's olive-grey skin and crest of
     spines, never bright green; an Edmontosaurus's duck bill and short neck),
     each feature answered seen / not seen / not visible, and the traits that
     would give a wrong animal away (a bird's head on a pterosaur, a long neck
     on a hadrosaur);
  2. relevance: does it fit the line, with nothing off topic in it (candles,
     books where no books are spoken of, people, text, logos, modern objects).

A candidate passes when the animal is the right one, no feature is seen to be
wrong, nothing forbidden is seen, and it fits. A beat's candidates are tried in
order - the chosen clip and its alternates, the photo and its alternates, the
reconstruction's three seeds - and the passing ones become its picks, which
pipeline/docplan.py uses in that order (a long line takes the second one as its
second shot). A beat with no passing candidate is dropped: the shot before it
runs on, so the scene is replaced by its neighbour rather than shown wrong.

Everything asked and answered goes to species_check.json. Needs this channel's
own GEMINI_API_KEY_PH; without it nothing is filtered and the log says so.
"""
import base64
import io
import json
import os
import re
import time
import urllib.error
import urllib.request
from pathlib import Path

from PIL import Image

from core.logger import get_logger

log = get_logger("species")

ROOT = Path(__file__).resolve().parent.parent
MODELS = ("gemini-flash-latest", "gemini-flash-lite-latest")
URL = "https://generativelanguage.googleapis.com/v1beta/models/{m}:generateContent"
# 12.11: was 1 - "a line's second shot is another moment of the same clip" - but since 07.11 another
# moment of a clip is the same source (twice a film, never within 3 min: pipeline/fragments.py), so a
# long line had nothing for its second shot and ran on (megalodon: 15 s a shot). Two passing clips a beat.
WANT_PASSING = 2
_SPENT = set()            # models out of quota for the rest of this run

SCENE = {"name": "a prehistoric scene (no particular animal asked for)", "features": [],
         "forbidden": ["a modern animal (bird, mammal, modern fish) in a prehistoric scene", "people",
                       "buildings, roads or vehicles"]}

SPECIES = {
    "crocodile": {
        "name": "a crocodilian (crocodile, alligator or caiman)",
        "features": ["armoured back with rows of bony scutes", "long flattened snout",
                     "eyes and nostrils raised on top of the head", "short sprawling legs",
                     "long muscular tail flattened from side to side"],
        "forbidden": ["a lizard's body (monitor lizard, iguana)", "a toy, statue or drawing"]},
    "limulus": {
        "name": "a horseshoe crab (Limulus polyphemus or another living horseshoe crab)",
        "features": ["smooth horseshoe-shaped domed front shell (carapace)",
                     "hinged rear shell section with short spines along its edges",
                     "one long, straight, rigid tail spine (telson)"],
        "forbidden": ["claws held out in front like a true crab", "visible antennae",
                      "a ghost crab, shore crab or any true crab", "a horseshoe (metal)"]},
    "nautilus": {
        "name": "a chambered nautilus (Nautilus or Allonautilus)",
        "features": ["smooth coiled external shell", "brown or orange stripes on a pale shell",
                     "many thin tentacles without suckers around the head", "a leathery hood above the eye"],
        "forbidden": ["an octopus or squid without an external shell", "an ammonite with ribbed fossil shell",
                      "a snail", "only an empty shell"]},
    "tuatara": {
        "name": "a tuatara (Sphenodon punctatus)",
        "features": ["olive-grey, grey-brown or dull greenish-grey skin",
                     "a crest of soft pale spines along the neck and back", "a large head with a blunt snout",
                     "small granular scales with paler spots"],
        "forbidden": ["bright green or vivid colours like an iguana, anole or chameleon",
                      "a dewlap or throat fan", "an iguana, gecko, chameleon or water dragon",
                      "a skeleton or preserved specimen"]},
    "latimeria": {
        "name": "a coelacanth (Latimeria) - alive, preserved, or a faithful museum model",
        "features": ["heavy body covered in thick large scales",
                     "paired fins on fleshy lobed stalks, like small limbs",
                     "a three-lobed tail with a small extra lobe in the middle",
                     "steel-blue to brownish colour with irregular pale blotches (preserved specimens may be brown or grey)"],
        "forbidden": ["an ordinary ray-finned fish (grouper, bass, cod, carp)", "a shark or whale",
                      "a cartoon or toy"]},
    "coelacanth_fossil": {
        "name": "a fossil fish in rock, from the coelacanth's ancient lineage",
        "features": ["the outline of a fish preserved in stone", "fins and tail visible in the rock"],
        "forbidden": ["a living fish", "a modern illustration"]},
    "edmontosaurus": {
        "name": "Edmontosaurus, a large duck-billed hadrosaur dinosaur",
        "features": ["a broad flat duck-like bill at the front of a long low head",
                     "no crest, horn or frill on the head",
                     "a short thick neck, much shorter than the body",
                     "a bulky body and a long heavy tail held straight out behind",
                     "three-toed hind feet and smaller front legs"],
        "forbidden": ["a long neck like a sauropod, giraffe or ostrich", "a bird's head or feathers",
                      "a sail, spikes or armour plates on the back", "a mix with a modern animal"]},
    "quetzalcoatlus": {
        "name": "Quetzalcoatlus or another giant azhdarchid pterosaur",
        "features": ["a huge straight pointed toothless beak", "a very long stiff neck",
                     "wings of bare skin membrane stretched from one very long finger to the hind legs",
                     "a body covered in short fuzz, not feathers"],
        "forbidden": ["a bird's head or a bird's body", "feathered wings", "a bat or a dragon",
                      "teeth in the beak", "a long bony tail with a vane"]},
    "devonian": {
        "name": "a Devonian sea with extinct fishes: a lobe-finned fish like Eusthenopteron and/or the armoured placoderm Dunkleosteus",
        "features": ["a lobe-finned fish with fleshy-stalked paired fins and a three-lobed tail",
                     "or an armoured fish with bony head plates and blade-like jaws",
                     "an ancient reef of sponges, horn corals and crinoids"],
        "forbidden": ["modern reef fish (clownfish, perch, snapper, tropical fish)", "a modern shark",
                      "modern branching hard corals as the main reef"]},
}
BUILTIN = frozenset(SPECIES)

PROMPT = """You check pictures for a German nature documentary about "living fossils".
The narration at this point says (German): "{line}"
{topic}
{species_block}
Look at the image(s) - {what} - and answer with JSON only:
{{"subject": "<what is really shown, a few words>",
 "species_match": true|false|null,
 "features": {{{features_json}}},
 "forbidden_seen": [<every item from the forbidden list that you do see>],
 "fits_narration": true|false,   // true if it shows what "The picture should show" describes, or is a fair documentary stand-in for the line (a sea for a voyage, a harbour coast for a harbour); false only if it contradicts or distracts from it
 "off_topic_objects": [<objects that do not belong under this line: candles, books (unless books are spoken of), people, text, logos, watermarks, modern objects, buildings, vehicles>],
 "other_animal": "<an animal clearly shown that is neither the species asked for nor named or meant by the narration line (a goldfish, a crab, a jellyfish under a line about something else), or null>",
 "people": true|false,
 "text_or_logo": true|false,
 "museum_display": true|false,   // a preserved specimen, taxidermy, a model or a museum display case
 "note": "<a few words>"}}
For every feature answer "yes" (clearly seen), "no" (seen to be different) or "not visible".
species_match is null when no species is asked for."""


def _key():
    return os.environ.get("GEMINI_API_KEY_PH", "").strip()


# 12.11 (megalodon, 0 of 20 FLUX frames passed): the planner's checklist named what only a fossil or a
# tape measure shows - "V-shaped notch at the base of the tooth root", "body 9-18 m", "small size
# comparable to a great white" - which no picture of a living animal can answer, so every frame was
# "too few features visible" or "forbidden: small size". A checklist asks only what a picture shows.
MEASURED = re.compile(r"\d+(?:[.,]\d+)?\s*(?:-|to|bis)?\s*\d*\s*(?:m|cm|mm|meters?|metres?|feet|ft|tons?|tonnes?|kg)\b|"
                      r"estimated|\bsize\b|comparable|compared to|than (?:a|an|the) (?:modern|living)|weigh|"
                      r"\broot\b|notch|posterior teeth|internal|microscop|bite force|lifespan|growth rings?|"
                      r"\bdna\b|genetic", re.I)
# a feature of the living animal is never its fossil; a fossil or a skeleton in the picture is still
# a wrong picture, so the forbidden list keeps those
INVISIBLE = re.compile(MEASURED.pattern + r"|fossil|bone|skelet|vertebra|cartilag", re.I)
# the planner added this to every animal's forbidden list: a FLUX megalodon is a shark and a real great
# white is a modern animal, so it turned down the very animals asked for; a prehistoric scene keeps it (SCENE)
NOT_FOR_ANIMALS = re.compile(r"modern animal in a prehistoric scene", re.I)


def visible(sp):
    """The checklist with only what a picture can show (features and forbidden traits)."""
    sp = dict(sp)
    sp["features"] = [f for f in sp.get("features") or [] if not INVISIBLE.search(str(f))]
    sp["forbidden"] = [f for f in sp.get("forbidden") or []
                       if not MEASURED.search(str(f)) and not NOT_FOR_ANIMALS.search(str(f))]
    return sp


def load_episode_species(ep):
    """The episode's own checklists (episodes/<slug>/species.json), as a picture can answer them."""
    p = Path(ep) / "species.json"
    if p.exists():
        SPECIES.update({k: visible(v) for k, v in json.loads(p.read_text(encoding="utf-8")).items()
                        if k not in BUILTIN})
    SPECIES["prehistoric_scene"] = SCENE


def extinct(species):
    sp = SPECIES.get(species or "")
    return bool(sp) and sp.get("living") is False


def topic_of(beat, species):
    """What the picture should show. A FLUX frame of the episode's extinct animal shows it alive: 12.11,
    megalodon frames were turned down as "does not fit" under lines about its vertebrae and bite marks,
    though a documentary shows the animal while it tells of its fossils."""
    if beat.get("topic"):
        return beat["topic"]
    if beat.get("kind") == "recon" and extinct(species):
        return (f"{SPECIES[species]['name']}, alive, as reconstructed for this film - the animal this part of "
                f"the film tells of (shown alive also while the narration speaks of its fossils or finds)")
    return None


PEOPLE_WORDS = ["people", "person", "man", "woman", "men", "women", "scientist", "researcher", "fisherman",
                "fishermen", "diver", "worker", "palaeontologist", "paleontologist"]
SCALE_WORDS = ["diver", "people", "person", "boat", "hull", "ship", "vessel", "kayak", "torch"]


def allow_of(beat):
    """Words for what belongs in this beat's picture. 12.11: a beat planned to show people ("fishermen
    pulling a net", "scientist examining a fossil") was turned down for its people, and a scene planned
    with a diver or a boat for scale (plan_episode.SCENE_POSES) for its diver - 176 such refusals."""
    allow = list(beat.get("allow") or ())
    if beat.get("people"):
        allow += PEOPLE_WORDS
    if beat.get("kind") == "recon" and beat.get("scene"):
        allow += SCALE_WORDS
    return allow


def vkey(cid, species):
    """Where a verdict is kept: a clip asked for an animal under its own key per animal (12.11: kept by
    the clip alone, a clip judged as a whale was read again for a great white - 314 of 886 readings
    in megalodon answered another question than the one asked)."""
    return f"{cid}|{species}" if species and species != "prehistoric_scene" and species in SPECIES else cid


QA_REJECTS = set()          # 12.11: sources the QA of a render found showing the wrong animal or unlicensed


def load_rejects(ep):
    """episodes/<slug>/qa_rejects.json: what a failed QA turned down; a rebuild leaves it out."""
    p = Path(ep) / "qa_rejects.json"
    QA_REJECTS.clear()
    if p.exists():
        QA_REJECTS.update(json.loads(p.read_text(encoding="utf-8")))


def saved(known, cid, species):
    """The kept verdict that answers this candidate for this species, or None (to be asked)."""
    if str(cid) in QA_REJECTS:
        return {"model": "reused", "reused_from": "qa", "reused_why": "turned down by the film's QA",
                "species_match": False}
    k = vkey(cid, species)
    v = known.get(k)
    if v is not None or k == cid:
        return v
    v = known.get(cid)
    if v is None:
        return None
    if v.get("missing") or v.get("model") in ("siglip", "reused"):
        return v                                    # no download, or the CPU's / another episode's no
    if v.get("species_asked"):
        return v if v["species_asked"] == species else None
    feats = set((v.get("features") or {}).keys())
    if feats and feats == set(SPECIES[species].get("features") or []):
        return v
    # a FLUX frame belongs to one beat, so its old verdict was asked for this animal - unless its
    # checklist is another animal's of this episode
    if "__s" in cid and feats and not any(feats == set(sp.get("features") or []) for sp in SPECIES.values()):
        return v
    return None


def _b64(path, side=768):
    im = Image.open(path).convert("RGB")
    im.thumbnail((side, side))
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode()


def ask(key, images, line, species=None, topic=None, archive=False, what="one still image", recon=False):
    block, feats = _species_block(species, archive, recon)
    text = PROMPT.format(line=line, topic=f"The picture should show: {topic}." if topic else "",
                         species_block=block, what=what, features_json=feats)
    parts = [{"text": text}] + [{"inline_data": {"mime_type": "image/jpeg", "data": _b64(p)}} for p in images]
    payload = {"contents": [{"parts": parts}],
               "generationConfig": {"temperature": 0, "responseMimeType": "application/json"}}
    last = None
    for model in MODELS:
        if model in _SPENT:
            continue
        for attempt in range(2):
            try:
                from core import gemini as _g
                _g._pace(model, 1800 * len(images))               # the model's minute (config rpm/tpm)
                req = urllib.request.Request(URL.format(m=model), data=json.dumps(payload).encode(),
                                             headers={"Content-Type": "application/json", "x-goog-api-key": key})
                with urllib.request.urlopen(req, timeout=90) as r:
                    d = json.loads(r.read())
                _g._WINDOW.setdefault(model, []).append((time.time(), 1800 * len(images)))
                out = "".join(p.get("text", "") for p in d["candidates"][0]["content"]["parts"])
                from core import llm_usage
                um = d.get("usageMetadata", {})
                llm_usage.record(model, um.get("promptTokenCount"), um.get("candidatesTokenCount"), "species_check")
                v = json.loads(out[out.find("{"): out.rfind("}") + 1])
                v["model"] = model
                return v
            except urllib.error.HTTPError as e:
                last = f"HTTP {e.code}"
                if e.code == 429 and attempt == 0:
                    time.sleep(8)
                    continue
                if e.code == 429:
                    # 28.10: waiting out 429s took two minutes a picture and the job ran
                    # out of time; a model that is out of quota is set aside for the run
                    _SPENT.add(model)
                break                                   # 404 / quota: try the next model
            except Exception as e:                      # noqa: BLE001
                last = f"{type(e).__name__}: {str(e)[:120]}"
                time.sleep(5)
    return {"error": last}


# owner 08.11: several clips a request (megalodon needed ~390 requests, ~250 a day). Off (1) until
# scripts/validate_batch.py has shown >= 95 % agreement with the saved single-clip verdicts; then
# config/channel.json "species_batch_clips" says how many
BATCH_CLIPS = int(json.loads((ROOT / "config" / "channel.json").read_text(encoding="utf-8"))
                  .get("species_batch_clips", 1))
BATCH_INTRO = """You check pictures for a German nature documentary about "living fossils".
Below are {n} separate CLIPS, each with its own narration line, its own species (or none) and its
own image(s). Judge every clip on its own, as if it were the only one: never let one clip's
species, features or pictures influence the answer for another.
Answer with JSON only: {{"results": [ {{"i": <clip number>, <the answer object for that clip>}}, ... ]}}
with one entry for every clip, in order."""
BATCH_CLIP = """CLIP {i}.
The narration at this point says (German): "{line}"
{topic}
{species_block}
Its image(s) follow - {what}. Its answer object:
{{"subject": "<what is really shown, a few words>",
 "species_match": true|false|null,
 "features": {{{features_json}}},
 "forbidden_seen": [<every item from this clip's forbidden list that you do see>],
 "fits_narration": true|false,   // true if it shows what "The picture should show" describes, or is a fair documentary stand-in for the line; false only if it contradicts or distracts from it
 "off_topic_objects": [<objects that do not belong under this line: candles, books (unless books are spoken of), people, text, logos, watermarks, modern objects, buildings, vehicles>],
 "other_animal": "<an animal clearly shown that is neither the species asked for nor named or meant by the narration line (a goldfish, a crab, a jellyfish under a line about something else), or null>",
 "people": true|false, "text_or_logo": true|false,
 "museum_display": true|false,   // a preserved specimen, taxidermy, a model or a museum display case
 "note": "<a few words>"}}
For every feature answer "yes" (clearly seen), "no" (seen to be different) or "not visible".
species_match is null when no species is asked for."""


RECON_NOTE = ("\nThis animal is EXTINCT: the image is a reconstruction of it, drawn for this film. Judge whether "
              "it shows this animal as reconstructed from the features above; that it looks like its living "
              "relatives is not wrong in itself (an extinct shark is drawn as a shark). Judge size only where "
              "something in the picture gives the scale.")


def _species_block(species, archive, recon=False):
    sp = SPECIES.get(species) if species else None
    if not sp:
        return "No particular species is asked for: judge relevance only.", ""
    block = (f"Species asked for: {sp['name']}.\nFeatures to check: " + "; ".join(sp["features"]) +
             ".\nForbidden - any of these means it is the wrong animal or a bad picture: " +
             "; ".join(sp["forbidden"]) + ".")
    if recon and sp.get("living") is False:
        block += RECON_NOTE
    if archive:
        block += ("\nThis is a historical photograph (possibly black and white, with people): judge only "
                  "whether the animal in it is this species; people and old print are fine here.")
    return block, ", ".join(f'"{f}": "yes|no|not visible"' for f in sp["features"])


def ask_batch(key, items):
    """items: [{"images", "line", "species", "topic", "archive", "what"}] (at most BATCH_CLIPS) ->
    [verdict] in the same order, each as ask() gives it ({"error": ...} when it could not be had)."""
    parts = [{"text": BATCH_INTRO.format(n=len(items))}]
    n_img = 0
    for i, it in enumerate(items):
        block, feats = _species_block(it.get("species"), it.get("archive"), it.get("recon"))
        parts.append({"text": BATCH_CLIP.format(
            i=i, line=it["line"], topic=f"The picture should show: {it['topic']}." if it.get("topic") else "",
            species_block=block, what=it.get("what", "one still image"), features_json=feats)})
        parts += [{"inline_data": {"mime_type": "image/jpeg", "data": _b64(p)}} for p in it["images"]]
        n_img += len(it["images"])
    payload = {"contents": [{"parts": parts}],
               "generationConfig": {"temperature": 0, "responseMimeType": "application/json"}}
    last = None
    for model in MODELS:
        if model in _SPENT:
            continue
        for attempt in range(2):
            try:
                from core import gemini as _g
                _g._pace(model, 1800 * n_img)
                req = urllib.request.Request(URL.format(m=model), data=json.dumps(payload).encode(),
                                             headers={"Content-Type": "application/json", "x-goog-api-key": key})
                with urllib.request.urlopen(req, timeout=180) as r:
                    d = json.loads(r.read())
                _g._WINDOW.setdefault(model, []).append((time.time(), 1800 * n_img))
                out = "".join(p.get("text", "") for p in d["candidates"][0]["content"]["parts"])
                from core import llm_usage
                um = d.get("usageMetadata", {})
                llm_usage.record(model, um.get("promptTokenCount"), um.get("candidatesTokenCount"), "species_check")
                res = json.loads(out[out.find("{"): out.rfind("}") + 1]).get("results", [])
                by = {}
                for r_ in res:
                    if isinstance(r_, dict) and isinstance(r_.get("i"), int):
                        by[r_["i"]] = r_
                verdicts = []
                for i in range(len(items)):
                    v = by.get(i)
                    if not isinstance(v, dict):
                        verdicts.append({"error": "no answer for this clip in the batch"})
                        continue
                    v = {k: x for k, x in v.items() if k != "i"}
                    v.update({"model": model, "batch": len(items)})
                    verdicts.append(v)
                return verdicts
            except urllib.error.HTTPError as e:
                last = f"HTTP {e.code}"
                if e.code == 429 and attempt == 0:
                    time.sleep(8)
                    continue
                if e.code == 429:
                    _SPENT.add(model)
                break
            except Exception as e:                      # noqa: BLE001
                last = f"{type(e).__name__}: {str(e)[:120]}"
                time.sleep(5)
    return [{"error": last} for _ in items]


def passes(v, species=None, archive=False, allow=()):
    """(ok, why). `allow`: words for objects that belong in this beat's pictures
    (a museum specimen's label and case, the bait cage of the only nautilus footage)."""
    allow = [a.lower() for a in allow]
    if "error" in v:
        return None, v["error"]
    if v.get("missing"):                            # the clip never downloaded: the next one is tried
        return False, "frames missing"
    if v.get("model") == "reused":                  # turned down in another episode, for no line's reason
        return False, f"turned down before ({v.get('reused_from')}): {v.get('reused_why')}"
    if v.get("model") == "siglip":                  # turned down on the CPU (pipeline/prefilter.py)
        return False, f"CPU: clearly not it ({v.get('score')})"
    why = []
    if species:
        # a prehistoric scene asks for no animal: Gemini answers null there, and the
        # forbidden list decides (30.10: all 20 FLUX frames failed on that null, 0 LTX clips)
        scene = species == "prehistoric_scene"
        if v.get("species_match") is not True and not (scene and v.get("species_match") is None):
            why.append("not the species")
        if not archive:
            feats = v.get("features") or {}
            answer = {f: str(a).strip().lower() for f, a in feats.items()}
            bad = [f for f, a in answer.items() if a in ("no", "false")]
            seen = [f for f, a in answer.items() if a in ("yes", "true")]
            if bad:
                why.append("wrong: " + "; ".join(bad))
            if feats and len(seen) < max(1, len(feats) // 2):
                why.append("too few features visible")
        if v.get("forbidden_seen"):
            why.append("forbidden: " + "; ".join(map(str, v["forbidden_seen"])))
    if v.get("fits_narration") is False:
        why.append("does not fit the line")
    # 12.11 (megalodon's QA "match": a goldfish, a crab, a jellyfish passed as scenery under lines that
    # name none of them): an animal the line does not tell of is a wrong picture (asked since 12.11)
    other = v.get("other_animal")
    if other and str(other).strip().lower() not in ("null", "none", "no", "false", "") and \
            not any(w in str(other).lower() for w in allow):
        why.append(f"an animal the line does not name: {other}")
    ok_words = list(allow) + (["people", "person", "text", "man", "woman"] if archive else [])
    off = [o for o in (v.get("off_topic_objects") or []) if not any(w in str(o).lower() for w in ok_words)]
    if off:
        why.append("off topic: " + "; ".join(map(str, off)))
    if not archive and ((v.get("people") and "people" not in allow) or (v.get("text_or_logo") and "text" not in allow)):
        why.append("people or text in shot")
    return not why, ", ".join(why)


# ------------------------------------------------------------ candidates
ORDER = {}                # cid -> the prefilter's score on its poster: the order the clips are tried in


def use_scores(slug):
    """The episode's prefilter scores (episodes/<slug>/prefilter_scores.json), for candidates()."""
    ORDER.clear()
    p = ROOT / "episodes" / slug / "prefilter_scores.json"
    if p.exists():
        ORDER.update(json.loads(p.read_text(encoding="utf-8")))
    return ORDER


def candidates(beat, footage, layers):
    """[(candidate id, kind, source)] in the order they are tried. Owner 10.11: a beat's stock clips
    in the order of the prefilter's score (best first; a clip with no score keeps its place after
    them), the walk stopping at the first one Gemini passes."""
    k = beat["kind"]
    if k == "real":
        main = footage[beat["need"]]
        cs = [main] + list(main.get("alternates") or [])
        if ORDER and any(c.get("id") in ORDER for c in cs):
            scored = sorted((c for c in cs if c.get("id") in ORDER), key=lambda c: -ORDER[c["id"]])
            cs = scored + [c for c in cs if c.get("id") not in ORDER]
        return [(c["id"], "real", c) for c in cs]
    if k == "photo":
        return [(i, "photo", layers / f"{i}.jpg") for i in [beat["need"]] + list(beat.get("alts") or [])]
    if k == "recon":
        return [(f"{beat['id']}__s{s}", "recon", layers / f"{beat['id']}__s{s}.jpg") for s in range(3)]
    return []


_REUSE = {}


def _reuse_index(slug):
    """cid -> [(episode, verdict)] of every other episode's saved verdicts (built once a run)."""
    if slug not in _REUSE:
        idx = {}
        for vf in (ROOT / "episodes").glob("*/species_verdicts.json"):
            if vf.parent.name == slug:
                continue
            try:
                for k, v in json.loads(vf.read_text(encoding="utf-8")).items():
                    if isinstance(v, dict) and v.get("model") not in ("siglip", None) and "error" not in v:
                        idx.setdefault(k.split("|")[0], []).append((vf.parent.name, v))
            except (OSError, ValueError):
                continue
        _REUSE[slug] = idx
    return _REUSE[slug]


def _asked_for(v):
    """The species a saved verdict was asked for: kept since 10.11, before that read off its features."""
    if v.get("species_asked"):
        return v["species_asked"]
    feats = set((v.get("features") or {}).keys())
    return next((k for k, sp in SPECIES.items() if feats and feats == set(sp.get("features") or [])), None)


def reused_fail(slug, cid, species, beat):
    """Owner 10.11: a clip another episode's check turned down for a reason no line changes - the
    wrong animal for the same species, a logo or text, people where none are allowed - is turned
    down again without a request. A pass is never reused: it depends on the line."""
    allow = [a.lower() for a in allow_of(beat)]
    for other, v in _reuse_index(slug).get(cid, []):
        same = species and species != "prehistoric_scene" and _asked_for(v) == species
        why = ("not the species" if same and v.get("species_match") is False
               else "text or logo" if v.get("text_or_logo") and "text" not in allow and not beat.get("archive")
               else "people" if v.get("people") and "people" not in allow and not beat.get("archive") else None)
        if why:
            return {**{k: v.get(k) for k in ("subject", "species_match", "people", "text_or_logo", "note")},
                    "model": "reused", "reused_from": other, "reused_why": why}
    return None


def beat_kind(beat):
    """The prefilter's kind of beat: an animal asked for, or a scene with none."""
    return "species" if beat.get("species") and beat.get("species") != "prehistoric_scene" else "scene"


def cpu_verdict(cid, src, beat, footage):
    """A verdict from SigLIP on the clip's poster frame when it is clear; None when Gemini has to say."""
    from pipeline import prefilter
    th = prefilter.thresholds(beat_kind(beat))
    if th is None or beat.get("archive") or not prefilter.available():
        return None
    pp = prefilter.poster(src, cid)
    if pp is None:
        return None
    query = beat.get("topic") or (footage.get(beat.get("need")) or {}).get("query")
    x = prefilter.scores([(prefilter.text_for(query=query), pp)])[0]
    # owner 08.11: the CPU only turns clips down - a species is never taken without Gemini (the
    # calibration had 8 passing clips), whatever accept threshold the config might hold
    d = prefilter.decide(x, (None, th[1]))
    if d != "reject":
        return None
    return {"model": "siglip", "cpu": d, "score": round(x, 4), "kind": beat_kind(beat), "poster": str(pp),
            "subject": query, "note": "decided on the CPU (pipeline/prefilter.py)"}


def _score_all(slug, shots, footage):
    """The prefilter's score of every stock candidate on its poster, kept in
    episodes/<slug>/prefilter_scores.json (the order candidates() tries them in)."""
    from pipeline import prefilter
    p = ROOT / "episodes" / slug / "prefilter_scores.json"
    have = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
    vf = ROOT / "episodes" / slug / "species_verdicts.json"
    if not p.exists() and vf.exists() and any(
            isinstance(v, dict) and str(v.get("model", "")).startswith("gemini")
            for v in json.loads(vf.read_text(encoding="utf-8")).values()):
        # its check began in the planned order (megalodon): a new order now would ask again what is judged
        return have
    if not prefilter.available():
        return have
    todo, seen = [], set()
    for line in shots:
        for beat in line["beats"]:
            if beat["kind"] != "real" or beat["need"] in seen or beat["need"] not in footage:
                continue
            seen.add(beat["need"])
            main = footage[beat["need"]]
            query = beat.get("topic") or main.get("query")
            for c in [main] + list(main.get("alternates") or []):
                if c.get("id") and c["id"] not in have:
                    pp = prefilter.poster(c, c["id"])
                    if pp:
                        todo.append((c["id"], prefilter.text_for(query=query), pp))
    for (cid, _t, _p), s in zip(todo, prefilter.scores([(t_, p_) for _c, t_, p_ in todo])):
        have[cid] = round(s, 4)
    p.write_text(json.dumps(have, indent=0), encoding="utf-8")
    log.info("prefilter: %d stock clips of %s scored (%d new)", len(have), slug, len(todo))
    return have


def prefilter_pass(slug):
    """Before the budget is asked: every stock clip the CPU can decide is decided, so the
    estimate (automation/llm_budget.py) counts only what Gemini still has to see. Returns the
    number of clips decided."""
    ep = ROOT / "episodes" / slug
    shots = json.loads((ep / "shots.json").read_text(encoding="utf-8"))["lines"]
    footage = json.loads((ep / "footage.json").read_text(encoding="utf-8"))
    load_episode_species(ep)
    vpath = ep / "species_verdicts.json"
    known = json.loads(vpath.read_text(encoding="utf-8")) if vpath.exists() else {}
    _score_all(slug, shots, footage)
    use_scores(slug)
    n, seen = 0, set()
    for line in shots:
        for beat in line["beats"]:
            if beat["kind"] != "real" or beat["need"] in seen or beat["need"] not in footage:
                continue
            seen.add(beat["need"])
            for cid, kind, src in candidates(beat, footage, None):
                v = saved(known, cid, beat.get("species"))
                if v is None:
                    v = reused_fail(slug, cid, beat.get("species"), beat) or cpu_verdict(cid, src, beat, footage)
                    if v is None:
                        break                          # doubtful: Gemini sees it (and what follows) in the check
                    known[vkey(cid, beat.get("species"))] = v
                    n += 1
                ok, _why = passes(v, beat.get("species"), bool(beat.get("archive")), allow_of(beat))
                if ok:
                    break
    vpath.write_text(json.dumps(known, indent=1, ensure_ascii=False), encoding="utf-8")
    log.info("prefilter: %d stock clips of %s decided on the CPU", n, slug)
    return n


def _beats_to_check(spec, timeline, kinds):
    """[(beat, line text, species)] in the order check() walks them (each beat key once)."""
    out, seen = [], set()
    for entry in spec:
        line = entry["at"]
        if timeline:
            n = entry.get("n")
            line = (timeline[n]["text"] if n is not None and n < len(timeline) else
                    next((l["text"] for l in timeline if l["text"].startswith(entry["at"])), line))
        for beat in entry["beats"]:
            if beat["kind"] == "anim" or (kinds and beat["kind"] not in kinds):
                continue
            bk = beat_key(beat)
            if bk in seen:
                continue
            seen.add(bk)
            species = beat.get("species")
            if beat["kind"] == "recon" and species not in SPECIES:
                species = "prehistoric_scene"
            out.append((beat, line, species))
    return out


def batch_prefill(key, beats, footage, layers, out_dir, known, vpath):
    """Gemini's verdicts for the clips check() will ask about, BATCH_CLIPS a request: every beat's
    next undecided candidate (the CPU's clear misses skipped), until each beat has its passing
    clips or none is left. check() then finds them all judged and asks nothing more."""
    state = {beat_key(b): 0 for b, _l, _s in beats}
    asked = 0
    while True:
        pending = []
        for beat, line, species in beats:
            bk = beat_key(beat)
            want = 2 if beat.get("standby") else WANT_PASSING
            n_pass, tried = 0, set()
            for cid, kind, src in candidates(beat, footage, layers):
                if cid in tried:
                    continue
                tried.add(cid)
                v = saved(known, cid, species)
                if v is None and kind == "real":
                    v = cpu_verdict(cid, src, beat, footage)
                    if v:
                        known[vkey(cid, species)] = v
                if v is None:
                    if state[bk] >= 0:
                        pending.append((beat, line, species, cid, kind, src))
                    break
                ok, _w = passes(v, species, bool(beat.get("archive")), allow_of(beat))
                if ok:
                    n_pass += 1
                    if n_pass >= want:
                        break
        if not pending:
            break
        progress = False
        for k in range(0, len(pending), BATCH_CLIPS):
            group = []
            for beat, line, species, cid, kind, src in pending[k:k + BATCH_CLIPS]:
                fr = _frames(cid, src, out_dir)
                if not fr:
                    known[cid] = {"missing": True, "note": "no frames (the clip did not download)"}
                    progress = True                     # the beat's next clip comes in the next round
                    continue
                group.append((beat, cid, {"images": fr, "line": line, "species": species,
                                          "topic": topic_of(beat, species), "archive": bool(beat.get("archive")),
                                          "recon": kind == "recon",
                                          "what": "three frames from the start, middle and end of one video clip"
                                          if kind == "real" else "one still image"}))
            if not group:
                continue
            vs = ask_batch(key, [g[2] for g in group])
            asked += 1
            for (beat, cid, it), v in zip(group, vs):
                if "error" in v:
                    state[beat_key(beat)] = -1          # unknown: check() asks it on its own, or later
                else:
                    v["species_asked"] = it["species"]
                    known[vkey(cid, it["species"])] = v
                    progress = True
            if asked % 5 == 0:
                vpath.write_text(json.dumps(known, indent=1, ensure_ascii=False), encoding="utf-8")
        if not progress:
            break
    vpath.write_text(json.dumps(known, indent=1, ensure_ascii=False), encoding="utf-8")
    log.info("species check: %d requests of up to %d clips", asked, BATCH_CLIPS)
    return asked


def beat_key(beat):
    return beat.get("id") or beat["need"]


def _frames(cid, src, work):
    if isinstance(src, Path):
        return [src] if src.exists() else []
    from pipeline import footage as fmod, select
    url = src["url"]
    path = url
    if "commons" in (src.get("source") or "").lower() or "wikimedia" in url:
        ext = Path(url.split("?")[0]).suffix
        try:
            path = str(fmod.download(url, ROOT / "cache" / "media" / f"{cid}{ext}"))
        except Exception as e:                                  # noqa: BLE001
            log.warning("%s: download failed (%s)", cid, e)
            return []
    return select.frames(path, src.get("duration") or 12, work / "frames", cid.replace("/", "_"))


def check(slug, layers, out_dir, timeline=None, kinds=None):
    """Writes species_picks.json and species_check.json into out_dir; returns the picks.
    kinds: only these beat kinds (the FLUX frames are checked on their own
    before animation). Verdicts already given for a picture
    (episodes/<slug>/species_verdicts.json) are reused, not asked again."""
    ep = ROOT / "episodes" / slug
    spec = json.loads((ep / "shots.json").read_text(encoding="utf-8"))["lines"]
    footage = json.loads((ep / "footage.json").read_text(encoding="utf-8"))
    # the episode's own checklists, written with its plan (automation/plan_episode.py)
    load_episode_species(ep)
    load_rejects(ep)
    layers, out_dir = Path(layers), Path(out_dir)
    key = _key()
    picks, logrows, seen, museum = {}, [], {}, []
    vpath = ep / "species_verdicts.json"
    known = json.loads(vpath.read_text(encoding="utf-8")) if vpath.exists() else {}
    asked = [0]
    use_scores(slug)
    if key and BATCH_CLIPS > 1:
        # owner 08.11: the clips asked BATCH_CLIPS a request first; the walk below then reads them
        batch_prefill(key, _beats_to_check(spec, timeline, kinds), footage, layers, out_dir, known, vpath)
    for entry in spec:
        line = entry["at"]
        if timeline:
            n = entry.get("n")
            line = (timeline[n]["text"] if n is not None and n < len(timeline) else
                    next((l["text"] for l in timeline if l["text"].startswith(entry["at"])), line))
        for beat in entry["beats"]:
            if beat["kind"] == "anim" or (kinds and beat["kind"] not in kinds):
                continue
            bk = beat_key(beat)
            if bk in picks:                          # the same frame or clip planned again later
                continue
            species, archive = beat.get("species"), bool(beat.get("archive"))
            if beat["kind"] == "recon" and species not in SPECIES:
                species = "prehistoric_scene"       # no modern animals in a prehistoric scene
            row = {"beat": bk, "kind": beat["kind"], "line": line, "species": species, "candidates": []}
            passing = []
            tried = set()
            # a stand-in of a living animal: both seeds, each a picture of its own (docplan)
            want = 2 if beat.get("standby") else WANT_PASSING
            for cid, kind, src in candidates(beat, footage, layers):
                if len(passing) >= want:
                    break
                if cid in tried:                     # an alternate listed twice
                    continue
                tried.add(cid)
                label = source_label(kind, src)
                allow = allow_of(beat)
                if key and saved(known, cid, species) is not None:
                    # judged before (an earlier run, a rerun) for this very animal: no download, no request
                    v = saved(known, cid, species)
                    ok, why = passes(v, species, archive, allow)
                    if ok and v.get("museum_display"):
                        museum.append(cid)
                    row["candidates"].append({"id": cid, "pass": ok, "why": why, "answer": v, "source": label,
                                              "cached": True})
                    if ok:
                        passing.append(cid)
                    continue
                if key and kind == "real":
                    # owner 08.11: SigLIP on the CPU first - a clear miss needs no Gemini; 10.11: nor a
                    # clip another episode turned down for a reason no line changes
                    v = reused_fail(slug, cid, species, beat) or cpu_verdict(cid, src, beat, footage)
                    if v:
                        known[vkey(cid, species)] = v
                        ok, why = passes(v, species, archive, allow)
                        row["candidates"].append({"id": cid, "pass": ok, "why": why, "answer": v, "source": label,
                                                  "frame": v.get("poster")})
                        log.info("%-22s %-34s %s %s", bk, cid, "PASS" if ok else "fail", why)
                        if ok:
                            passing.append(cid)
                        continue
                if key and all(m in _SPENT for m in MODELS):
                    # the day's quota is spent: no download for an answer nobody can give today
                    row["candidates"].append({"id": cid, "verdict": "unknown", "why": "quota spent", "source": label})
                    break
                fr = _frames(cid, src, out_dir)
                if not fr:
                    # 09.11: kept as unusable, so the budget walks on to the alternate instead of
                    # counting this clip as still to ask (megalodon waited a day on 22 such beats)
                    known[cid] = {"missing": True, "note": "no frames (the clip did not download)"}
                    row["candidates"].append({"id": cid, "verdict": "missing"})
                    continue
                if not key:
                    passing.append(cid)
                    row["candidates"].append({"id": cid, "verdict": "not checked (no GEMINI_API_KEY_PH)",
                                              "source": label, "frame": str(fr[len(fr) // 2])})
                    continue
                what = "three frames from the start, middle and end of one video clip" if kind == "real" \
                    else "one still image"
                v = ask(key, fr, line, species, topic_of(beat, species), archive, what, recon=kind == "recon")
                if "error" not in v:
                    v["species_asked"] = species
                    known[vkey(cid, species)] = v
                    asked[0] += 1
                    if asked[0] % 10 == 0:       # kept as it goes: a run that dies later loses nothing
                        vpath.write_text(json.dumps(known, indent=1, ensure_ascii=False), encoding="utf-8")
                ok, why = passes(v, species, archive, allow)
                if ok is None:
                    # the checker failed (429, quota, a timeout), not the picture: no verdict,
                    # and nothing kept or dropped on it - the beat is asked again later
                    row["candidates"].append({"id": cid, "verdict": "unknown", "why": why, "source": label})
                    continue
                if ok and v.get("museum_display"):
                    museum.append(cid)
                row["candidates"].append({"id": cid, "pass": ok, "why": why, "answer": v, "source": label,
                                          "frame": str(fr[len(fr) // 2])})
                log.info("%-22s %-34s %s %s", bk, cid, "PASS" if ok else "fail", why)
                if ok:
                    passing.append(cid)
                time.sleep(1.5)
            judged = [c for c in row["candidates"] if "pass" in c]
            if not passing and not judged and row["candidates"]:
                # 04.11 (megalodon): every frame missing on the runner, or the checker out of
                # quota, read as "scene dropped" for all 28 beats. Unknown is not rejected:
                # no pick is written, so the next run asks again
                row["action"] = "unknown"
                logrows.append(row)
                continue
            picks[bk] = passing
            row["picks"] = passing
            row["action"] = ("kept" if passing and passing[0] == candidates(beat, footage, layers)[0][0]
                             else "replaced by an alternate" if passing else "scene dropped")
            logrows.append(row)
    picks["_museum"] = museum                   # docplan shows at most three of these
    vpath.write_text(json.dumps(known, indent=1, ensure_ascii=False), encoding="utf-8")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "species_picks.json").write_text(json.dumps(picks, indent=1), encoding="utf-8")
    summary = {"checked_with": MODELS[0] if key else None,
               "skipped": None if key else "GEMINI_API_KEY_PH is not set: nothing was filtered",
               "beats": len(logrows), "kept": sum(r["action"] == "kept" for r in logrows),
               "alternate": sum(r["action"] == "replaced by an alternate" for r in logrows),
               "dropped": [r["beat"] for r in logrows if r["action"] == "scene dropped"],
               "unknown": [r["beat"] for r in logrows if r["action"] == "unknown"],
               "rows": logrows}
    (out_dir / "species_check.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False), encoding="utf-8")
    try:
        grid(summary, out_dir / "species_grid.jpg")
    except Exception as e:                                       # noqa: BLE001
        log.warning("species grid failed: %s", e)
    log.info("species check: %d beats, %d kept, %d alternates, dropped %s, unknown %s; %d new requests",
             summary["beats"], summary["kept"], summary["alternate"], summary["dropped"], summary["unknown"], asked[0])
    return picks


def source_label(kind, src):
    """FLUX / Commons / NOAA / Pixabay, for the grid."""
    if kind == "recon":
        return "FLUX"
    if kind == "photo":
        return "Commons"
    text = f"{src.get('source', '')} {src.get('author', '')} {src.get('url', '')}".lower()
    if "noaa" in text or "oceanexplorer" in text:
        return "NOAA"
    return "Pixabay" if "pixabay" in text else "Commons"


def grid(summary, out_path, tile_w=360, cols=6):
    """Every candidate that was looked at: green frame = passed and used in
    order, red = turned down (replaced by the next, or the scene dropped)."""
    from PIL import ImageDraw
    tiles = []
    for row in summary["rows"]:
        for c in row["candidates"]:
            if not c.get("frame") or not Path(c["frame"]).exists():
                continue
            im = Image.open(c["frame"]).convert("RGB")
            im.thumbnail((tile_w, tile_w))
            th = int(tile_w * 9 / 16)
            t = Image.new("RGB", (tile_w + 12, th + 64), (18, 18, 18))
            ok = c.get("pass", True)
            col = (40, 200, 70) if ok else (225, 40, 40)
            ImageDraw.Draw(t).rectangle([0, 0, tile_w + 11, th + 11], outline=col, width=6)
            t.paste(im.crop((0, 0, min(im.width, tile_w), min(im.height, th))), (6, 6))
            d = ImageDraw.Draw(t)
            d.text((6, th + 14), f"{c.get('source', '?')} | {row['beat']}"[:52], fill=(255, 255, 255))
            d.text((6, th + 30), str(c["id"])[:52], fill=(200, 200, 200))
            d.text((6, th + 46), ("PASS" if ok else "FAIL: " + str(c.get("why", ""))[:44]), fill=col)
            tiles.append(t)
    if not tiles:
        return None
    w, h = tiles[0].size
    sheet = Image.new("RGB", (cols * w, ((len(tiles) + cols - 1) // cols) * h), (0, 0, 0))
    for i, t in enumerate(tiles):
        sheet.paste(t, ((i % cols) * w, (i // cols) * h))
    sheet.save(out_path, quality=86)
    return out_path


def check_thumbnail(path, species, words):
    """The same question for a thumbnail base, plus the style it has to have."""
    key = _key()
    if not key:
        return {"skipped": "no GEMINI_API_KEY_PH"}
    v = ask(key, [path], " ".join(words), species,
            "one animal, close on its head or eye, against a dark background, lit from the side or behind")
    ok, why = passes(v, species)
    return {"pass": True if ok is None else ok, "why": why, "answer": v}
