"""Measure a render run and forecast a 25-minute episode: render time and GPU.

    python automation/forecast.py <slug> <res> <run_id> [--gpu-report kaggle/results_recon/report.json]

Render time comes from GitHub's own job timestamps (the run's jobs API): the
picture is "render" from the first part starting to the merge finishing;
prepare (downloads, narration, plan) is reported apart, because it scales with
the number of clips rather than with minutes of picture.

The forecast keeps parts at the size measured here (about 30 s of picture each)
and runs them GitHub_CONCURRENCY at a time - 20 jobs at once on a free plan -
so a 25-minute film is ceil(parts / 20) waves of one part's wall time.

GPU: the reconstruction kernel's own stage timings (report.json) give seconds
per image for FLUX, Real-ESRGAN and the depth/LaMa layers, plus fixed loading.
A 25-minute episode needs the same density of reconstructions as this test.
"""
import json
import math
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TARGET_S = 25 * 60
GITHUB_CONCURRENCY = 20


def _t(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def jobs(run_id, repo):
    out = subprocess.run(["gh", "api", f"repos/{repo}/actions/runs/{run_id}/jobs?per_page=100"],
                         capture_output=True, text=True, check=True).stdout
    return json.loads(out)["jobs"]


def render_times(run_id, repo, plan, parts_dir):
    js = jobs(run_id, repo)
    parts = [j for j in js if j["name"].startswith("part")]
    prep = next(j for j in js if j["name"] == "prepare")
    merge = next((j for j in js if j["name"] == "merge"), None)
    start = min(_t(j["started_at"]) for j in parts)
    part_walls = [(_t(j["completed_at"]) - _t(j["started_at"])).total_seconds() for j in parts]
    reps = [json.loads(p.read_text()) for p in sorted(Path(parts_dir).glob("part_*.json"))]
    pic = [r["seconds"] for r in reps]
    now_end = _t(merge["completed_at"]) if merge and merge.get("completed_at") else datetime.now(start.tzinfo)
    return {
        "prepare_s": round((_t(prep["completed_at"]) - _t(prep["started_at"])).total_seconds()),
        "render_s": round((now_end - start).total_seconds()),
        "parts": len(parts), "part_wall_s": [round(w) for w in part_walls],
        "part_picture_s": [round(p, 1) for p in pic],
        "part_render_s": [r["render_s"] for r in reps],
        "picture_s": plan["total_s"],
    }


def forecast_render(m, plan):
    per_part = sum(m["part_picture_s"]) / max(1, len(m["part_picture_s"]))
    n_parts = math.ceil(TARGET_S / per_part)
    waves = math.ceil(n_parts / GITHUB_CONCURRENCY)
    part_wall = max(m["part_wall_s"])                      # the slowest part sets the pace of a wave
    merge_s = max(0, m["render_s"] - part_wall)            # join + sound + mux, grows with length
    merge_25 = merge_s * TARGET_S / plan["total_s"]
    total = waves * part_wall + merge_25
    return {"parts": n_parts, "waves": waves, "part_wall_s": part_wall,
            "merge_s": round(merge_25), "render_s": round(total),
            "render_min": round(total / 60, 1),
            "with_prepare_min": round((total + m["prepare_s"] * TARGET_S / plan["total_s"]) / 60, 1)}


def forecast_week(gpu_report, cpu_summary=None, samples=None, videos=2, frames=60, minutes=25):
    """GPU hours a week for `videos` episodes of `minutes`, Kaggle doing FLUX
    alone (at most `frames` an episode), per voice option; and what the CPU
    stages cost on the runners."""
    r = json.loads(Path(gpu_report).read_text())
    imgs = r.get("images", [])
    n = max(1, len(imgs))
    flux_per = sum(i["s"] for i in imgs) / n
    # what the kernel spends besides the frames themselves: installs, weights, warm-up
    overhead = max(0.0, (r.get("total_s") or 0) - sum(i["s"] for i in imgs)) if r.get("flux_only") else 300.0
    flux_ep_h = (overhead + frames * flux_per) / 3600
    out = {"flux_s_per_frame": round(flux_per, 1), "kernel_overhead_s": round(overhead),
           "frames_per_episode": frames, "flux_gpu_h_per_episode": round(flux_ep_h, 2),
           "flux_gpu_h_per_week": round(videos * flux_ep_h, 2), "voices": {}}
    for key in ("A", "B", "C"):
        v = (samples or {}).get(key) or {}
        if key == "B":
            ep = 0.0
        elif "gpu_s_per_audio_min" in v:
            ep = (v["gpu_s_per_audio_min"] * minutes + (v.get("model_load_s") or 0) + 240) / 3600   # +4 min installs
        else:
            out["voices"][key] = {"missing": True}
            continue
        out["voices"][key] = {"voice_gpu_h_per_episode": round(ep, 2),
                              "gpu_h_per_week": round(videos * (flux_ep_h + ep), 2)}
    if cpu_summary:
        per = (cpu_summary.get("per_flux_frame_s") or {}).get("total_s")
        allf = (cpu_summary.get("per_frame_all_s") or {}).get("total_s")
        cpu_s = frames * (per or 0)
        out["cpu_layers"] = {"s_per_flux_frame": per, "s_per_frame_incl_photos": allf, "flux_frames_per_episode": frames,
                             "cpu_min_per_episode_flux": round(cpu_s / 60, 1),
                             "wall_min_on_12_runners": round(cpu_s / 12 / 60, 1),
                             "note": "real photographs add their own s_per_frame each; they need no GPU"}
    return out


if __name__ == "__main__":
    slug, res, run_id = sys.argv[1:4]
    repo = "edbackulin-eng/prehistoric-channel"
    W = ROOT / "work" / slug
    plan = json.loads((W / "plan.json").read_text(encoding="utf-8"))
    m = render_times(run_id, repo, plan, W / f"parts_{res}")
    rep = {"res": res, "measured": m, "forecast_25min": forecast_render(m, plan)}
    gp = W / "inputs" / "kaggle_recon_report.json"
    cp = W / "inputs" / "cpu_layers_report.json"
    sp = W / "samples_report.json"
    if gp.exists():
        rep["week_2x25min"] = forecast_week(gp, json.loads(cp.read_text()) if cp.exists() else None,
                                            json.loads(sp.read_text(encoding="utf-8")) if sp.exists() else None)
    (W / f"timing_{res}.json").write_text(json.dumps(rep, indent=1), encoding="utf-8")
    print(json.dumps(rep, indent=1))
    over = rep["forecast_25min"]["render_min"] > 60
    Path(W / "forecast_over_hour.txt").write_text("yes" if over else "no")
