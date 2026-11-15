"""Smoke checks of the real-footage format (owner 15.11), run by scripts/smoke.py.

No network, no model: the plan is cut from a made-up pool with the text model switched off (the
line tags then come from the animal names in the text), a still and a test-pattern film are
rendered at 640x360, the QA and the credits are run on what came out.
"""
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _pool():
    animals = [{"key": "mosasaurus", "name_de": "Mosasaurus", "latin": "Mosasaurus", "extinct": True},
               {"key": "monitor_lizard", "name_de": "Waran", "latin": "Varanus", "extinct": False,
                "relative_of": "mosasaurus"}]
    items = []

    def add(i, kind, cat, animal, lic="CC BY-SA 4.0", dur=12):
        items.append({"id": f"{kind}-{i}", "source": "Wikimedia Commons" if kind == "still" else "Pixabay",
                      "kind": kind, "url": f"https://example.org/{kind}-{i}", "full_url": f"https://example.org/{i}.jpg",
                      "width": 3840, "height": 2160, "duration": dur if kind == "video" else None,
                      "license": lic if kind == "still" else "Pixabay Content License", "author": f"Author {i}",
                      "page": f"https://example.org/page/{i}", "title": f"File:Thing {i}.jpg",
                      "desc": {"category": cat, "animal": animal, "what": f"{cat} {animal}", "cgi": False,
                               "text": False, "quality": 4}, "usable": True})
    for i in range(14):
        add(i, "still", ["fossil", "skeleton", "palaeoart"][i % 3], "mosasaurus")
    for i in range(14, 26):
        add(i, "video", "living_animal", "monitor_lizard")
    for i in range(26, 40):
        add(i, "video", ["landscape", "excavation", "museum"][i % 3], "none")
    add(40, "video", "living_animal", "other:cat")
    add(41, "still", "portrait", "none", lic="Public domain")
    items[-1]["title"] = "File:Georges Cuvier portrait 1831.jpg"
    add(42, "still", "portrait", "none", lic="Public domain")
    items[-1]["title"] = "File:Somebody else.jpg"
    for i in range(43, 49):
        add(i, "still", "archive_photo", "none", lic="Public domain")
        items[-1]["title"] = f"File:Maastricht quarry {1900 + i}.jpg"
    add(49, "video", "excavation", "none")             # shown by another film of the channel already
    return {"slug": "smoke-real", "animals": animals, "items": items}


def run():
    out = []
    from core import textllm
    from pipeline import realplan, realpool
    tmp = Path(tempfile.mkdtemp())
    ep = tmp / "episodes" / "smoke-real"
    ep.mkdir(parents=True)
    pool = _pool()
    (ep / "pool.json").write_text(json.dumps(pool), encoding="utf-8")
    texts = ["Der Mosasaurus jagte in den Meeren der Kreidezeit.", "Forscher fanden seine Knochen in Maastricht.",
             "Heute leben seine Verwandten, die Warane, auf Inseln.", "Der Waran schwimmt gut.",
             "Das Meer war damals warm und flach.", "Der Mosasaurus war bis zu 17 Meter lang.",
             "Die Grabungen dauern bis heute an.", "Am Ende verschwand der Mosasaurus mit den Dinosauriern.",
             "In Paris erkannte Georges Cuvier, was man gefunden hatte."]
    tl, t = [], 0.0
    for i, x in enumerate(texts):
        tl.append({"text": x, "kind": "normal", "section": "hook" if i == 0 else "story", "start": t, "end": t + 13})
        t += 14
    other = tmp / "episodes" / "other-film"
    other.mkdir(parents=True)
    (other / "footage.json").write_text(json.dumps({"video-49": {"id": "video-49"}}), encoding="utf-8")
    real_root, real_ask, pool_root = realplan.ROOT, textllm.ask, realpool.ROOT
    realplan.ROOT = realpool.ROOT = tmp

    def no_llm(*a, **k):
        raise RuntimeError("no model in the smoke test")
    textllm.ask = no_llm
    real_tag = realplan.tag_lines

    def tags(timeline, animals):                       # names as the text model would give them
        t_ = real_tag(timeline, animals)
        t_[-1]["names"] = ["Georges Cuvier", "Paris"]
        return t_
    realplan.tag_lines = tags
    try:
        plan = realplan.build("smoke-real", tl, t, parts=2)
    finally:
        realplan.ROOT, textllm.ask, realpool.ROOT, realplan.tag_lines = real_root, real_ask, pool_root, real_tag
    shots = plan["shots"]
    ids = [s["pool_id"] for s in shots]
    out.append(("real: each picture once", len(ids) == len(set(ids)) and plan["checks"]["repeats"] == 0,
                f"{len(ids)} shots, {len(set(ids))} pictures"))
    spec = json.loads((ep / "shots.json").read_text(encoding="utf-8"))
    named = {e["n"]: {b["species"] for b in e["beats"]} for e in spec["lines"]}
    wrong = [(s["start"], s["species"]) for s in shots if s.get("species") and named.get(s["line"])
             and s["species"] not in named[s["line"]]]
    out.append(("real: never another animal than the line names", not wrong, str(wrong[:5])))
    lizard_off = [s["start"] for s in shots if s.get("species") == "monitor_lizard"
                  and "monitor_lizard" not in named.get(s["line"], set())]
    out.append(("real: the living relative only where the line speaks of it", not lizard_off, str(lizard_off)))
    out.append(("real: no picture of an unrelated animal", all(not str(s.get("species") or "").startswith("other")
                                                             and s["pool_id"] != "video-40" for s in shots), ""))
    cont = all(abs(a["end"] - b["start"]) < 1e-6 for a, b in zip(shots, shots[1:]))
    out.append(("real: the picture covers the whole narration", cont and shots[0]["start"] == 0.0
                and abs(shots[-1]["end"] - plan["total_s"]) < 1e-6, f"{shots[-1]['end']} vs {plan['total_s']}"))
    out.append(("real: no FLUX, LTX or 2.5D in the plan", all(s["kind"] in ("real", "photo") for s in shots)
                and not any(str(s.get("need", "")).startswith("ltx-") for s in shots), ""))
    out.append(("real: every shot carries its licence", all(s.get("license") for s in shots), ""))
    last = len(texts) - 1
    out.append(("real: the portrait of the person the line names, under that line",
                any(s["pool_id"] == "still-41" and s["line"] == last for s in shots)
                and not any(s["pool_id"] == "still-42" for s in shots), str([(s["line"], s["pool_id"]) for s in shots
                                                                              if s.get("category") == "portrait"])))
    out.append(("real: a picture another film showed is not shown again", "video-49" not in ids, ""))
    out.append(("real: archive pictures used", plan["checks"].get("archive_share", 0) > 0,
                str(plan["checks"].get("archive_share"))))

    # the vision model's verdicts: renders, text and moving paintings are out
    base = {"kind": "video", "source": "Pixabay", "desc": {"category": "living_animal", "animal": "x", "cgi": False,
                                                          "text": False, "quality": 4}}
    cgi = {**base, "desc": {**base["desc"], "cgi": True}}
    txt = {**base, "desc": {**base["desc"], "text": True}}
    art = {**base, "desc": {**base["desc"], "category": "palaeoart"}}
    still_art = {**art, "kind": "still", "source": "Wikimedia Commons"}
    out.append(("real: CGI, text and a moving painting are refused, a still painting taken",
                realpool.usable(base) and not realpool.usable(cgi) and not realpool.usable(txt)
                and not realpool.usable(art) and realpool.usable(still_art), ""))
    out.append(("real: licences - CC BY-SA and PD in, NC and ND out",
                bool(realpool.LICENCE_OK.search("CC BY-SA 4.0")) and bool(realpool.LICENCE_OK.search("Public domain"))
                and bool(realpool.LICENCE_BAD.search("CC BY-NC 2.0")) and bool(realpool.LICENCE_BAD.search("CC BY-ND 3.0"))
                and not realpool.LICENCE_BAD.search("CC BY-SA 3.0"), ""))

    # a portrait photograph on the 16:9 plate, then a still and a test film rendered
    from automation import build_doc
    from pipeline import docrender
    w = tmp / "render"
    (w / "layers").mkdir(parents=True)
    (w / "media").mkdir()
    Image.fromarray(np.random.default_rng(1).integers(0, 255, (1200, 800, 3), np.uint8)).save(w / "portrait.jpg")
    build_doc.still_16x9(w / "portrait.jpg", w / "layers" / "still-0.jpg")
    out.append(("real: a portrait photo on the 16:9 plate", Image.open(w / "layers" / "still-0.jpg").size == (4224, 2376), ""))
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc2=size=1280x720:rate=30",
                    "-t", "6", "-pix_fmt", "yuv420p", str(w / "media" / "video-0.mp4")], check=True)
    rshots = [{"kind": "photo", "image": "still-0", "move": "push_in", "start": 0.0, "end": 3.0, "transition": "cut",
               "line": 0, "text": "x"},
              {"kind": "real", "clip": {"id": "video-0", "duration": 6.0, "at": 0.08}, "start": 3.0, "end": 6.0,
               "transition": "fade", "line": 1, "text": "y"}]
    rplan = {"slug": "smoke-real", "total_s": 6.0, "xfade_s": 0.6, "shots": rshots, "parts": [[0, 2]]}
    docrender.SIZES["smoke"] = (640, 360)
    r = docrender.render_part(rplan, 0, {"media": w / "media", "layers": w / "layers"}, w / "tmp", w / "parts", res="smoke")
    out.append(("real: a still and a film rendered without layers", abs(r["seconds"] - 6.0) < 0.2, str(r["seconds"])))

    # the QA's licence rule and the scores on a real plan
    from automation import qa
    bad = {"format": "real", "shots": [{"kind": "photo", "pool_id": "x", "license": None}], "music": []}
    good = {"format": "real", "shots": [{"kind": "photo", "pool_id": "y", "license": "CC BY-SA 4.0"},
                                        {"kind": "real", "clip": {"id": "z"}, "license": "Pixabay Content License"}],
            "music": []}
    out.append(("real QA: a picture without a licence blocks", not qa.license_ok(bad, "none")[0]
                and qa.license_ok(good, "none")[0], qa.license_ok(bad, "none")[1]))
    chk = [{"name": n, "ok": False, "detail": "x"} for n in ("ltx", "dinosaurs", "motion", "wow", "scenes", "pace")]
    rc = qa.real_checks(chk, {"checks": {"repeats": 0, "unique_pictures": 9}})
    out.append(("real QA: the FLUX-only scores are gone, uniqueness measured",
                [c["name"] for c in rc] == ["pace", "unique"] and rc[-1]["ok"], str([c["name"] for c in rc])))
    shutil.rmtree(tmp, ignore_errors=True)
    return out


if __name__ == "__main__":
    bad = 0
    for name, ok, detail in run():
        print(f"[{'ok' if ok else 'FAIL'}] {name} {detail}")
        bad += not ok
    sys.exit(1 if bad else 0)
