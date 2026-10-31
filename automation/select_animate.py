"""Which FLUX frames LTX-Video animates: the ones that are the right animal.

    python automation/select_animate.py <slug>

The species check runs on the FLUX frames alone (pipeline/speciescheck.py,
kinds=recon); its verdicts are kept in episodes/<slug>/species_verdicts.json so
the render's full check does not ask about them again. Of the prompts the plan
marked for animation (recon.json "animate", hook and climax first), the first
passing seed of each goes to gpu/animate_job.json - the job the Kaggle kernel
gpu/kernels/animate_job.py reads.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from core.logger import get_logger  # noqa: E402
from pipeline import speciescheck  # noqa: E402

log = get_logger("animate")


def main(slug):
    ep = ROOT / "episodes" / slug
    recon = json.loads((ep / "recon.json").read_text(encoding="utf-8"))
    raw = ROOT / "gpu" / "results_recon" / "raw"
    out = ROOT / "work" / slug / "flux_check"
    picks = speciescheck.check(slug, raw, out, kinds=("recon",))
    frames = []
    for fid, motion in (recon.get("animate") or {}).items():
        ok = [c for c in picks.get(fid, [])]
        if not ok:
            log.info("%s: no seed passed the species check; not animated", fid)
            continue
        frames.append({"id": ok[0], "prompt": recon["prompts"][fid], "motion": motion})
    # a planned frame that failed gives its clip to another passing animal of the episode
    # (the living animals' stand-ins first), never past the plan's count
    cfg = json.loads((ROOT / "config" / "channel.json").read_text(encoding="utf-8"))
    from pipeline import docplan
    spare = sorted((f for f in recon.get("prompts", {}) if f not in (recon.get("animate") or {}) and picks.get(f)
                    and not docplan.is_landscape({"kind": "recon", "id": f, "topic": recon["prompts"][f]})),
                   key=lambda f: 0 if f in (recon.get("standby") or {}) else 1)
    for fid in spare[:max(0, len(recon.get("animate") or {}) - len(frames))]:
        frames.append({"id": picks[fid][0], "prompt": recon["prompts"][fid],
                       "motion": "moves slowly and naturally, gentle camera push in"})
        log.info("%s animated in place of a planned frame that failed", fid)
    job = {"slug": slug, "frames": frames, "steps": 30}
    (ROOT / "gpu" / "animate_job.json").write_text(json.dumps(job, indent=1), encoding="utf-8")
    log.info("animate %d of %d planned frames", len(frames), len(recon.get("animate") or {}))
    need = int(cfg["qa"]["ltx_min"])
    if len(frames) < need:
        # the render needs that many clips (QA): without enough frames no GPU hour is spent on it
        log.error("only %d frames to animate, the render needs %d LTX clips: the plan needs more FLUX animals",
                  len(frames), need)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
