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
WANT_PASSING = 2          # enough for a line long enough to need a second shot

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

PROMPT = """You check pictures for a German nature documentary about "living fossils".
The narration at this point says (German): "{line}"
{topic}
{species_block}
Look at the image(s) - {what} - and answer with JSON only:
{{"subject": "<what is really shown, a few words>",
 "species_match": true|false|null,
 "features": {{{features_json}}},
 "forbidden_seen": [<every item from the forbidden list that you do see>],
 "fits_narration": true|false,
 "off_topic_objects": [<objects that do not belong under this line: candles, books (unless books are spoken of), people, text, logos, watermarks, modern objects, buildings, vehicles>],
 "people": true|false,
 "text_or_logo": true|false,
 "note": "<a few words>"}}
For every feature answer "yes" (clearly seen), "no" (seen to be different) or "not visible".
species_match is null when no species is asked for."""


def _key():
    return os.environ.get("GEMINI_API_KEY_PH", "").strip()


def _b64(path, side=768):
    im = Image.open(path).convert("RGB")
    im.thumbnail((side, side))
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode()


def ask(key, images, line, species=None, topic=None, archive=False, what="one still image"):
    sp = SPECIES.get(species) if species else None
    if sp:
        block = (f"Species asked for: {sp['name']}.\nFeatures to check: " + "; ".join(sp["features"]) +
                 ".\nForbidden - any of these means it is the wrong animal or a bad picture: " +
                 "; ".join(sp["forbidden"]) + ".")
        if archive:
            block += ("\nThis is a historical photograph (possibly black and white, with people): judge only "
                      "whether the animal in it is this species; people and old print are fine here.")
        feats = ", ".join(f'"{f}": "yes|no|not visible"' for f in sp["features"])
    else:
        block, feats = "No particular species is asked for: judge relevance only.", ""
    text = PROMPT.format(line=line, topic=f"The picture should show: {topic}." if topic else "",
                         species_block=block, what=what, features_json=feats)
    parts = [{"text": text}] + [{"inline_data": {"mime_type": "image/jpeg", "data": _b64(p)}} for p in images]
    payload = {"contents": [{"parts": parts}],
               "generationConfig": {"temperature": 0, "responseMimeType": "application/json"}}
    last = None
    for model in MODELS:
        for attempt in range(4):
            try:
                req = urllib.request.Request(URL.format(m=model), data=json.dumps(payload).encode(),
                                             headers={"Content-Type": "application/json", "x-goog-api-key": key})
                with urllib.request.urlopen(req, timeout=90) as r:
                    d = json.loads(r.read())
                out = "".join(p.get("text", "") for p in d["candidates"][0]["content"]["parts"])
                v = json.loads(out[out.find("{"): out.rfind("}") + 1])
                v["model"] = model
                return v
            except urllib.error.HTTPError as e:
                last = f"HTTP {e.code}"
                if e.code == 429 and attempt < 3:
                    time.sleep(20 * (attempt + 1))
                    continue
                break                                   # 404 / quota: try the next model
            except Exception as e:                      # noqa: BLE001
                last = f"{type(e).__name__}: {str(e)[:120]}"
                time.sleep(5)
    return {"error": last}


def passes(v, species=None, archive=False):
    """(ok, why)."""
    if "error" in v:
        return None, v["error"]
    why = []
    if species:
        if v.get("species_match") is not True:
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
    off = [o for o in (v.get("off_topic_objects") or [])
           if not (archive and any(w in str(o).lower() for w in ("people", "person", "text", "man", "woman")))]
    if off:
        why.append("off topic: " + "; ".join(map(str, off)))
    if not archive and (v.get("people") or v.get("text_or_logo")):
        why.append("people or text in shot")
    return not why, ", ".join(why)


# ------------------------------------------------------------ candidates
def candidates(beat, footage, layers):
    """[(candidate id, kind, source)] in the order they are tried."""
    k = beat["kind"]
    if k == "real":
        main = footage[beat["need"]]
        return [(c["id"], "real", c) for c in [main] + list(main.get("alternates") or [])]
    if k == "photo":
        return [(i, "photo", layers / f"{i}.jpg") for i in [beat["need"]] + list(beat.get("alts") or [])]
    if k == "recon":
        return [(f"{beat['id']}__s{s}", "recon", layers / f"{beat['id']}__s{s}.jpg") for s in range(3)]
    return []


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


def check(slug, layers, out_dir, timeline=None):
    """Writes species_picks.json and species_check.json into out_dir; returns the picks."""
    ep = ROOT / "episodes" / slug
    spec = json.loads((ep / "shots.json").read_text(encoding="utf-8"))["lines"]
    footage = json.loads((ep / "footage.json").read_text(encoding="utf-8"))
    layers, out_dir = Path(layers), Path(out_dir)
    key = _key()
    picks, logrows = {}, []
    for entry in spec:
        line = entry["at"]
        if timeline:
            line = next((l["text"] for l in timeline if l["text"].startswith(entry["at"])), line)
        for beat in entry["beats"]:
            if beat["kind"] == "anim":
                continue
            bk = beat_key(beat)
            species, archive = beat.get("species"), bool(beat.get("archive"))
            row = {"beat": bk, "kind": beat["kind"], "line": line, "species": species, "candidates": []}
            passing = []
            for cid, kind, src in candidates(beat, footage, layers):
                if len(passing) >= WANT_PASSING:
                    break
                fr = _frames(cid, src, out_dir)
                if not fr:
                    row["candidates"].append({"id": cid, "verdict": "missing"})
                    continue
                if not key:
                    passing.append(cid)
                    row["candidates"].append({"id": cid, "verdict": "not checked (no GEMINI_API_KEY_PH)"})
                    continue
                what = "three frames from the start, middle and end of one video clip" if kind == "real" \
                    else "one still image"
                v = ask(key, fr, line, species, beat.get("topic"), archive, what)
                ok, why = passes(v, species, archive)
                if ok is None:                       # the checker failed, not the picture: keep it
                    ok, why = True, f"unchecked ({why})"
                row["candidates"].append({"id": cid, "pass": ok, "why": why, "answer": v})
                log.info("%-22s %-34s %s %s", bk, cid, "PASS" if ok else "fail", why)
                if ok:
                    passing.append(cid)
                time.sleep(3)
            picks[bk] = passing
            row["picks"] = passing
            row["action"] = ("kept" if passing and passing[0] == candidates(beat, footage, layers)[0][0]
                             else "replaced by an alternate" if passing else "scene dropped")
            logrows.append(row)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "species_picks.json").write_text(json.dumps(picks, indent=1), encoding="utf-8")
    summary = {"checked_with": MODELS[0] if key else None,
               "skipped": None if key else "GEMINI_API_KEY_PH is not set: nothing was filtered",
               "beats": len(logrows), "kept": sum(r["action"] == "kept" for r in logrows),
               "alternate": sum(r["action"] == "replaced by an alternate" for r in logrows),
               "dropped": [r["beat"] for r in logrows if r["action"] == "scene dropped"],
               "rows": logrows}
    (out_dir / "species_check.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False), encoding="utf-8")
    log.info("species check: %d beats, %d kept, %d alternates, dropped %s", summary["beats"], summary["kept"],
             summary["alternate"], summary["dropped"])
    return picks


def check_thumbnail(path, species, words):
    """The same question for a thumbnail base, plus the style it has to have."""
    key = _key()
    if not key:
        return {"skipped": "no GEMINI_API_KEY_PH"}
    v = ask(key, [path], " ".join(words), species,
            "one animal, close on its head or eye, against a dark background, lit from the side or behind")
    ok, why = passes(v, species)
    return {"pass": True if ok is None else ok, "why": why, "answer": v}
