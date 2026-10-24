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


def forecast_gpu(gpu_report, plan, voice_report=None):
    r = json.loads(Path(gpu_report).read_text())
    imgs = r.get("images", [])
    n = max(1, len(imgs))
    st = r.get("stages", {})
    flux_per = sum(i["s"] for i in imgs) / n
    flux_fixed = max(0, st.get("flux", 0) - flux_per * n)
    esr_per = st.get("esrgan", 0) / n
    layers = r.get("layers", [])
    lay_per = st.get("layers", 0) / max(1, len(layers))
    recon_shots = sum(1 for s in plan["shots"] if s["kind"] == "recon")
    scale = TARGET_S / plan["total_s"]
    n25 = math.ceil(recon_shots * scale)
    thumbs = 6
    sec = flux_fixed + (n25 + thumbs) * (flux_per + esr_per) + n25 * lay_per + 300   # +5 min installs
    out = {"test_kernel_s": r.get("total_s"), "images_test": n, "flux_s_per_image": round(flux_per, 1),
           "esrgan_s_per_image": round(esr_per, 1), "layers_s_per_image": round(lay_per, 1),
           "recon_images_25min": n25 + thumbs, "recon_gpu_min_25min": round(sec / 60, 1)}
    if voice_report and Path(voice_report).exists():
        v = json.loads(Path(voice_report).read_text())
        speech = v.get("seconds") or 0
        # the test kernel's wall time is not in its report; use the ledger entry if present
        out["chatterbox_audio_s_test"] = round(speech, 1)
    return out


if __name__ == "__main__":
    slug, res, run_id = sys.argv[1:4]
    repo = "edbackulin-eng/prehistoric-channel"
    W = ROOT / "work" / slug
    plan = json.loads((W / "plan.json").read_text(encoding="utf-8"))
    m = render_times(run_id, repo, plan, W / f"parts_{res}")
    rep = {"res": res, "measured": m, "forecast_25min": forecast_render(m, plan)}
    gp = W / "inputs" / "kaggle_recon_report.json"
    if gp.exists():
        rep["gpu_forecast_25min"] = forecast_gpu(gp, plan)
    (W / f"timing_{res}.json").write_text(json.dumps(rep, indent=1), encoding="utf-8")
    print(json.dumps(rep, indent=1))
    over = rep["forecast_25min"]["render_min"] > 60
    Path(W / "forecast_over_hour.txt").write_text("yes" if over else "no")
