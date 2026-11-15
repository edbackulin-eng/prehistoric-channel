"""credits.json, LICENSES.md and the description's credit block, from the
records the edit actually used - so a clip cannot be shown without a licence.

    python automation/credits.py living-fossils
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

MODELS = [
    {"what": "Reconstructions", "name": "FLUX.1-schnell (4-bit NF4 build magespace/FLUX.1-schnell-bnb-nf4)", "by": "Black Forest Labs", "license": "Apache-2.0"},
    {"what": "Upscaling", "name": "Real-ESRGAN x4plus", "by": "Xintao Wang et al.", "license": "BSD-3-Clause"},
    {"what": "Depth for 2.5D", "name": "Depth Anything V2 Small", "by": "Lihe Yang et al.", "license": "Apache-2.0"},
    {"what": "Background fill for 2.5D", "name": "LaMa (big-lama)", "by": "Samsung AI", "license": "Apache-2.0"},
    {"what": "Subject masks for 2.5D", "name": "BiRefNet / IS-Net (rembg)", "by": "ZhengPeng7 / Xuebin Qin et al.", "license": "MIT / Apache-2.0"},
    {"what": "Narration (German)", "name": "Piper 1.8, voice de_DE-thorsten-high", "by": "Open Home Foundation / Thorsten Mueller", "license": "engine GPL-3.0 (output not covered), voice CC0"},
    {"what": "Animated reconstructions", "name": "LTX-Video 2B 0.9.x (diffusers weights, Lightricks/LTX-Video)", "by": "Lightricks", "license": "LTX-Video OpenRAIL-M style licence: commercial use of outputs allowed, use restrictions apply"},
]
VOICES = [
    {"what": "Narration", "name": "Piper voice de_DE-thorsten-high (trained on the Thorsten-Voice dataset)",
     "by": "Thorsten Mueller / Open Home Foundation", "license": "voice and dataset CC0 1.0",
     "url": "https://huggingface.co/rhasspy/piper-voices/tree/main/de/de_DE/thorsten/high"},
]
FONTS = [{"name": n, "license": "SIL Open Font License 1.1"} for n in
         ("Montserrat", "Cormorant Garamond", "Anton")]
MAPS = {"name": "Blue Marble Next Generation (Dec 2004, topography and bathymetry)", "by": "NASA Earth Observatory",
        "license": "Public domain (NASA)", "url": "https://visibleearth.nasa.gov/images/73909"}


REPO_URL = "https://github.com/edbackulin-eng/prehistoric-channel/blob/main"
DESC_CREDITS_MAX = 2400        # characters of the description the credit block may take (YouTube: 5000 in all)


def main_real(slug, plan):
    """owner 15.11, real footage: every clip and picture of the film with its source, author and licence.
    CC BY / CC BY-SA need title, author, source and licence - in full in episodes/<slug>/CREDITS.md
    (linked from the description), and the authors with their licences in the description itself."""
    ep = ROOT / "episodes" / slug
    pool = {it["id"]: it for it in json.loads((ep / "pool.json").read_text(encoding="utf-8"))["items"]}
    used = []
    for s in plan["shots"]:
        it = pool.get(s.get("pool_id"))
        if it and it["id"] not in {u["id"] for u in used}:
            used.append(it)
    tr = ROOT / "work" / slug / "release" / "thumbnails_report.json"
    trep = json.loads(tr.read_text(encoding="utf-8")) if tr.exists() else {}
    best = ((trep.get("choice") or {}).get("best") or "") if isinstance(trep, dict) else ""
    for v in (trep.get("variants") or []) if isinstance(trep, dict) else []:
        sid = (v.get("source") or {}).get("id")
        if sid in pool and sid not in {u["id"] for u in used} and best.endswith(v["id"].split("_", 1)[1] + ".jpg"):
            used.append(pool[sid])
    used_music = set(plan.get("music") or [])
    music = [t for t in json.loads((ROOT / "assets/music/library.json").read_text(encoding="utf-8"))["tracks"]
             if t["title"] in used_music]
    amb_lib = json.loads((ROOT / "assets/ambience/library.json").read_text(encoding="utf-8"))["beds"]
    used_amb = {s.get("amb") for s in plan["shots"]}
    amb = [{"scene": k, **v} for k, v in amb_lib.items() if k in used_amb and "file" in v]
    rec = [{k: it.get(k) for k in ("id", "source", "title", "tags", "author", "license", "license_url", "page")}
           for it in used]
    credits = {"format": "real", "footage": rec, "music": music, "ambience": amb, "voices": VOICES, "fonts": FONTS}
    out = ROOT / "work" / slug / "release"
    out.mkdir(parents=True, exist_ok=True)
    for d in (out, ep):
        (d / "credits.json").write_text(json.dumps(credits, indent=1, ensure_ascii=False), encoding="utf-8")

    def name(r):
        return (r.get("title") or "").replace("File:", "").rsplit(".", 1)[0][:70] or (r.get("tags") or "")[:50]

    md = [f"# Bildnachweise / Credits: {slug}", "",
          "Alle Filmaufnahmen und Bilder sind echtes Material aus freien Quellen; die Lizenz steht jeweils auf der "
          "Seite der Quelle. Keine KI-Bilder, keine Animation.", "", "| Titel | Urheber | Lizenz | Quelle |",
          "|---|---|---|---|"]
    for r in rec:
        md.append(f"| {name(r).replace('|', '/')} | {str(r.get('author') or '').replace('|', '/')[:80]} | "
                  f"{r.get('license')} | [{r.get('source')}]({r.get('page')}) |")
    md += ["", "## Musik", ""] + [f"- \"{t['title']}\" Kevin MacLeod (incompetech.com), CC BY 4.0" for t in music]
    md += ["", "## Ton", ""] + [f"- {a['title'].replace('File:', '')} - {a['author']} - {a['license']} - {a['page']}"
                                for a in amb]
    md += ["", "## Stimme", "", "- Thorsten-Voice (Thorsten Müller), CC0 - Piper de_DE-thorsten-high"]
    (ep / "CREDITS.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    (out / "LICENSES.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    desc = ["Bildnachweise (Titel, Urheber, Lizenz und Link zu jeder Aufnahme): "
            f"{REPO_URL}/episodes/{slug}/CREDITS.md"]
    by = [r for r in rec if str(r.get("license") or "").upper().startswith("CC BY")]
    line = "Unter CC BY / CC BY-SA: " + "; ".join(f"{str(r.get('author') or 'unbekannt')[:40]} ({r['license']})"
                                                    for r in by)
    if len(line) > DESC_CREDITS_MAX:
        line = line[:DESC_CREDITS_MAX].rsplit(";", 1)[0] + "; weitere siehe Link."
    if by:
        desc.append(line)
    srcs = sorted({r.get("source") for r in rec})
    desc.append("Weiteres Material: " + ", ".join(srcs) + " (gemeinfrei, CC0, Pixabay-/Pexels-Lizenz).")
    for t in music:
        desc.append(f'Musik: "{t["title"]}" Kevin MacLeod (incompetech.com), licensed under Creative Commons: '
                    f'By Attribution 4.0 License, http://creativecommons.org/licenses/by/4.0/')
    for a in amb:
        if a["license"].upper().startswith("CC BY"):
            desc.append(f'Ton: "{a["title"].replace("File:", "")}" von {a["author"]}, {a["license"]} - {a["page"]}')
    desc.append("Sprecherstimme: Thorsten-Voice (Thorsten Müller, CC0).")
    (out / "description_credits.txt").write_text("\n".join(desc), encoding="utf-8")
    print(f"{len(rec)} pictures credited ({len(by)} CC BY/BY-SA) -> CREDITS.md, description_credits.txt")


def main(slug):
    ep = ROOT / "episodes" / slug
    plan = json.loads((ROOT / "work" / slug / "plan.json").read_text(encoding="utf-8"))
    if plan.get("format") == "real":
        return main_real(slug, plan)
    footage = json.loads((ep / "footage.json").read_text(encoding="utf-8"))
    recon = json.loads((ep / "recon.json").read_text(encoding="utf-8"))
    used_ids = {s["clip"]["id"] for s in plan["shots"] if s["kind"] == "real"}
    clips = {}
    for c in footage.values():
        for x in [c] + (c.get("alternates") or []):
            if x.get("id") in used_ids and x["id"] not in clips:
                clips[x["id"]] = {k: x.get(k) for k in ("source", "title", "tags", "author", "license", "license_url",
                                                        "page", "url")}
                clips[x["id"]]["source"] = x.get("source") or (
                    "Pixabay" if x["id"].startswith("pixabay") else "Wikimedia Commons")
                clips[x["id"]]["license"] = x.get("license") or (
                    "Pixabay Content License" if x["id"].startswith("pixabay") else None)
    photos = [recon["photos"][s["image"]] for s in plan["shots"] if s["kind"] == "photo" and s["image"] in recon["photos"]]
    tr = ROOT / "work" / slug / "release" / "thumbnails_report.json"
    trep = json.loads(tr.read_text(encoding="utf-8")) if tr.exists() else {}
    thumbs = [r["source"] for r in (trep.get("variants", []) if isinstance(trep, dict) else trep)
              if r.get("source") and r["source"].get("source") != "FLUX.1-schnell"]
    photos = list({p["page"]: p for p in photos}.values())
    used_music = set(plan.get("music") or [])
    music = [t for t in json.loads((ROOT / "assets/music/library.json").read_text(encoding="utf-8"))["tracks"]
             if t["title"] in used_music or (not used_music and t.get("role") == "bed")]
    amb_lib = json.loads((ROOT / "assets/ambience/library.json").read_text(encoding="utf-8"))["beds"]
    used_amb = {s.get("amb") for s in plan["shots"]}
    amb = [{"scene": k, **v} for k, v in amb_lib.items() if k in used_amb and "file" in v]
    credits = {"footage": list(clips.values()), "photos": photos, "music": music, "ambience": amb,
               "maps": MAPS, "models": MODELS, "voices": VOICES, "fonts": FONTS, "thumbnails": thumbs,
               "reconstructions": [s["image"] for s in plan["shots"] if s["kind"] == "recon"]}
    out = ROOT / "work" / slug / "release"
    out.mkdir(exist_ok=True)
    (out / "credits.json").write_text(json.dumps(credits, indent=1, ensure_ascii=False), encoding="utf-8")
    (ep / "credits.json").write_text(json.dumps(credits, indent=1, ensure_ascii=False), encoding="utf-8")

    attrib = [c for c in clips.values() if c["license"] and c["license"].upper().startswith("CC BY")]
    attrib += [p for p in photos if p["license"].upper().startswith("CC BY")]
    attrib += [t for t in thumbs if (t.get("license") or "").upper().startswith("CC BY")]
    desc = ["Filmmaterial und Bilder:"]
    for c in attrib:
        desc.append(f"- {c.get('title') or (c.get('tags') or '')[:60]} von {c['author']}, {c['license']} - {c['page']}")
    desc.append("- Weiteres Material: Pixabay (Pixabay-Inhaltslizenz); NOAA Ocean Exploration und NASA "
                "(gemeinfrei); Wikimedia Commons (gemeinfrei / CC0).")
    for t in music:
        desc.append(f'Musik: "{t["title"]}" Kevin MacLeod (incompetech.com), licensed under Creative Commons: '
                    f'By Attribution 4.0 License, http://creativecommons.org/licenses/by/4.0/')
    for a in amb:
        if a["license"].upper().startswith("CC BY"):
            desc.append(f'Ton: "{a["title"].replace("File:", "")}" von {a["author"]}, {a["license"]} - {a["page"]}')
    desc.append("Karten: NASA Earth Observatory, Blue Marble Next Generation.")
    desc.append("Sprecherstimme: Thorsten-Voice (Thorsten Müller, CC0).")
    desc.append("Rekonstruktionen ausgestorbener Tiere sind Illustrationen auf Grundlage veröffentlichter Forschung.")
    (out / "description_credits.txt").write_text("\n".join(desc), encoding="utf-8")

    lic = ["# Licences\n", "Every third-party element in the test video and why it may be used in a "
           "monetised YouTube video. Generated by automation/credits.py from what the edit used.\n",
           "## Music\n"]
    for t in music:
        lic.append(f"- **{t['title']}**, Kevin MacLeod - {t['license']} ({t['license_url']}). Commercial use "
                   "allowed, attribution in the description (see description_credits.txt).")
    lic.append("\n## Atmosphere\n")
    for a in amb:
        lic.append(f"- {a['title'].replace('File:', '')} - {a['author']} - {a['license']} - {a['page']}")
    lic.append("- Wind and underwater beds, all effects: synthesised in code (pipeline/docsound.py).")
    lic.append("\n## Footage and photographs\n")
    for c in clips.values():
        lic.append(f"- {c['source']}: {c.get('title') or (c.get('tags') or '')[:60]} - {c['author']} - {c['license']} "
                   f"- {c['page']}")
    for p in photos:
        lic.append(f"- Wikimedia Commons: {p['title']} - {p['author']} - {p['license']} - {p['page']}")
    lic.append("\n## Thumbnail photographs\n")
    for t in thumbs:
        lic.append(f"- {t.get('source')}: {t.get('title')} - {t.get('author')} - {t.get('license')} - {t.get('page')}")
    lic.append("\n## Voices\n")
    for v in VOICES:
        lic.append(f"- {v['what']}: {v['name']} ({v['by']}) - {v['license']} - {v['url']}")
    lic.append(f"\n## Maps\n\n- {MAPS['name']}, {MAPS['by']} - {MAPS['license']} - {MAPS['url']}")
    lic.append("\n## Models (all permit commercial use of their output)\n")
    for m in MODELS:
        lic.append(f"- {m['what']}: {m['name']} ({m['by']}) - {m['license']}")
    lic.append("\n## Fonts\n")
    for f in FONTS:
        lic.append(f"- {f['name']} - {f['license']}")
    (out / "LICENSES.md").write_text("\n".join(lic) + "\n", encoding="utf-8")
    (ROOT / "LICENSES.md").write_text("\n".join(lic) + "\n", encoding="utf-8")
    print(f"{len(clips)} clips, {len(photos)} photos, {len(amb)} beds -> test/credits.json, LICENSES.md")


if __name__ == "__main__":
    main(sys.argv[1])
