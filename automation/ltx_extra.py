"""One or more LTX clips more for an episode whose GPU run is done (ltx-extra.yml).

    python automation/ltx_extra.py prepare <slug> <frame id>[,<frame id>...]
    python automation/ltx_extra.py merge <slug>

prepare: each FLUX frame (from the episode run's artifact, gpu/results_recon/raw)
is put to the species check again with the narration line its beat now sits on
(episodes/<slug>/shots.json), and only a frame that passes is written into
gpu/animate_job.json; the clips the episode already has are set aside.
merge: after gpu/kaggle_once.py animate, the clips set aside come back beside the
new one, so the render gets them all from this run's artifact.

04.11, lebende-fossilien: QA wants two LTX clips of dinosaurs; the T. rex frame had
been put on a line about a fisherman, so the check had said "does not fit the line".
"""
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from core.logger import get_logger  # noqa: E402

log = get_logger("ltx-extra")
GPU = ROOT / "gpu"
KEEP = GPU / "_clips_before"


def line_of(slug, beat_id):
    spec = json.loads((ROOT / "episodes" / slug / "shots.json").read_text(encoding="utf-8"))["lines"]
    for e in spec:
        if any(b.get("id") == beat_id for b in e["beats"]):
            return e["at"], e.get("topic") or ""
    raise SystemExit(f"{beat_id} is on no line of {slug}'s shots.json")


def prepare(slug, frames):
    from pipeline import speciescheck
    recon = json.loads((ROOT / "episodes" / slug / "recon.json").read_text(encoding="utf-8"))
    key = speciescheck._key()
    raw = GPU / "results_recon" / "raw"
    job = []
    for fid in frames:
        base = fid.split("__s")[0]
        img = next(raw.glob(f"{fid}.*"), None)
        if img is None:
            raise SystemExit(f"{fid}: no FLUX frame in {raw}")
        line, topic = line_of(slug, base)
        if not key:
            raise SystemExit("no Gemini key: the frame cannot be checked, and no GPU hour goes on an unchecked one")
        v = speciescheck.ask(key, [img], line, "prehistoric_scene", topic or recon["prompts"][base])
        ok, why = speciescheck.passes(v, "prehistoric_scene")
        log.info("%s on %r: %s %s", fid, line[:60], "PASS" if ok else "fail", why)
        if ok is None:
            log.error("%s could not be checked (%s); try again later", fid, why)
            return 3
        if not ok:
            raise SystemExit(f"{fid} fails the species check on its line: {why}")
        job.append({"id": fid, "prompt": recon["prompts"][base],
                    "motion": (recon.get("animate") or {}).get(base) or "moves slowly and naturally, gentle camera push in"})
    cfg = json.loads((ROOT / "config" / "channel.json").read_text(encoding="utf-8"))
    n_frames = int(cfg["qa"].get("ltx_frames", 97))
    (GPU / "animate_job.json").write_text(json.dumps(
        {"slug": slug, "frames": job, "steps": 30, "num_frames": n_frames,
         "budget_s": 300 + (400 if n_frames > 97 else 240) * len(job)}, indent=1), encoding="utf-8")
    clips = GPU / "results_animate" / "clips"
    if clips.exists():
        KEEP.mkdir(parents=True, exist_ok=True)
        for f in clips.glob("*.mp4"):
            shutil.copy(f, KEEP / f.name)
    log.info("job: %s; %d clips set aside", [j["id"] for j in job], len(list(KEEP.glob("*.mp4"))) if KEEP.exists() else 0)
    return 0


def merge(slug):
    clips = GPU / "results_animate" / "clips"
    clips.mkdir(parents=True, exist_ok=True)
    new = sorted(p.name for p in clips.glob("*.mp4"))
    for f in KEEP.glob("*.mp4") if KEEP.exists() else []:
        if not (clips / f.name).exists():
            shutil.copy(f, clips / f.name)
    log.info("%s: new %s; %d clips in all", slug, new, len(list(clips.glob("*.mp4"))))
    return 0 if new else 1


if __name__ == "__main__":
    cmd, slug = sys.argv[1], sys.argv[2]
    sys.exit(prepare(slug, sys.argv[3].split(",")) if cmd == "prepare" else merge(slug))
