"""One-off: this episode's GPU work on Kaggle, in What If's and Closed Tabs' free hours.

    python gpu/kaggle_once.py recon living-fossils
    python gpu/kaggle_once.py animate <slug>
    python gpu/kaggle_once.py fetch [recon|animate]  # outputs only, for the render

The channel's GPU home is Modal (gpu/modal_app.py); this exists because Modal
runs no GPU without a card, and the test should not wait for one. It uses the
same kernel code (gpu/kernels/) on the shared Kaggle account, and it goes
strictly last:
  - never in Closed Tabs' window (20:00-04:00 UTC), never on a What If build
    day (Mon, Tue, Thu, Fri, Sat) before 17:30 UTC - its worst-case finish;
  - only after no What If or Closed Tabs kernel has been queued or running for
    15 minutes, and no What If build is in progress on GitHub;
  - only if the account's week stays within 27.5 h: Closed Tabs + What If (as
    Closed Tabs' own kaggle/gpu_budget.py measures them, fetched from its
    repository, so the accounting is theirs) + What If's next build in reserve
    + this channel's own week (state/gpu_ledger.json) + this job. The 25 h the
    two of them keep between themselves is theirs and unchanged;
  - only FLUX runs here (gpu/kernels/recon_job.py flux_source); everything
    after it runs on GitHub's CPUs (pipeline/layers_cpu.py);
  - every kernel run is written to state/kaggle_once.json and to
    state/gpu_ledger.json, the ledger in the format the other channels publish
    theirs ({ref, start, end, hours}), so the shared account can be read whole;
  - and if either of them pushed in the same minute, this kernel is deleted
    and the wait starts again.
Exit codes: 0 done, 3 not yet (the workflow restarts itself), other = failure.
"""
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "gpu" / "kernels"))
sys.path.insert(0, str(ROOT))
OTHERS = ("whatif", "closedtabs")
CT_RAW = "https://raw.githubusercontent.com/edbackulin-eng/closed-tabs/main/kaggle/"
JOB_HOURS = {"recon": 0.75, "animate": 1.2}     # recon is re-estimated from its frame count in main()
ACCOUNT_LIMIT_H = 27.5                        # the whole Kaggle week this channel may run into
FLUX_S_PER_FRAME = 33.0                       # measured on a T4, 28.10 (4-bit, 4 steps, 1360x768)
FLUX_OVERHEAD_H = 0.12                        # install, weights, warm-up
MAX_FLUX_FRAMES = 60                          # per episode: the GPU budget is planned on this
LEDGER = ROOT / "state" / "gpu_ledger.json"
SLUGS = {"recon": "prehistoric-recon", "animate": "prehistoric-animate"}
LTX_H_PER_CLIP = 0.10                         # LTX-Video 97 frames on a T4, estimated; measured per run
IDLE_S = 15 * 60
MAX_WAIT_S = float(os.environ.get("GPU_MAX_WAIT_S", 5 * 3600))
WHATIF_DAYS = {0, 1, 3, 4, 5}          # Mon Tue Thu Fri Sat


def say(msg):
    print(f"[{datetime.now(timezone.utc):%H:%M:%S}] {msg}", flush=True)


def kaggle(*args, timeout=120):
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"}
    try:
        r = subprocess.run([sys.executable, "-m", "kaggle", *args], capture_output=True, text=True,
                           encoding="utf-8", errors="replace", env=env, timeout=timeout)
        return r.stdout + r.stderr
    except subprocess.TimeoutExpired:
        return ""


def user():
    return os.environ.get("KAGGLE_USERNAME") or json.loads(
        (Path.home() / ".kaggle" / "kaggle.json").read_text())["username"]


def status(ref):
    s = kaggle("kernels", "status", ref).lower()
    if "kernelworkerstatus" not in s:
        return "unknown"
    return s.rsplit("kernelworkerstatus.", 1)[1].split('"')[0].strip()


def kernels():
    out = kaggle("kernels", "list", "--mine", "--page-size", "60", "--sort-by", "dateRun", "--csv")
    lines = out.splitlines()
    head = next((i for i, l in enumerate(lines) if l.startswith("ref,")), None)
    if head is None:
        return None
    hdr = lines[head].split(",")
    it = hdr.index("lastRunTime")
    rows = []
    for l in lines[head + 1:]:
        c = l.split(",")
        try:
            t = datetime.fromisoformat(c[it].strip()).replace(tzinfo=timezone.utc)
        except (ValueError, IndexError):
            continue
        rows.append((c[0].strip(), t))
    return rows


def others_busy():
    rows = kernels()
    if rows is None:
        return ["kernel list unavailable"]
    recent = datetime.now(timezone.utc) - timedelta(hours=12)
    busy = []
    for ref, t in rows:
        if t < recent or not ref.split("/")[-1].startswith(OTHERS):
            continue
        st = status(ref)
        if st in ("running", "queued", "pending", "unknown"):
            busy.append(f"{ref} ({st})")
    return busy


def whatif_building():
    req = urllib.request.Request("https://api.github.com/repos/edbackulin-eng/whatif-channel/actions/"
                                 "workflows/build-video.yml/runs?per_page=5",
                                 headers={"User-Agent": "prehistoric", "Accept": "application/vnd.github+json",
                                          **({"Authorization": f"Bearer {os.environ['GH_TOKEN']}"}
                                             if os.environ.get("GH_TOKEN") else {})})
    try:
        runs = json.loads(urllib.request.urlopen(req, timeout=30).read())["workflow_runs"]
    except Exception:                                            # noqa: BLE001
        return True                                              # unknown counts as busy
    return any(r["status"] in ("in_progress", "queued", "waiting") for r in runs)


def in_their_hours(now=None):
    now = now or datetime.now(timezone.utc)
    if now.hour >= 20 or now.hour < 4:
        return "Closed Tabs' window (20:00-04:00 UTC)"
    if now.weekday() in WHATIF_DAYS and now.hour < 17 or (now.weekday() in WHATIF_DAYS and now.hour == 17
                                                          and now.minute < 30):
        return "a What If build day, before 17:30 UTC"
    return None


_CT = {}


def budget(job_hours):
    """Closed Tabs' own accounting, run from its repository. Its code is
    fetched once per run; its What If log cache lives as long as the run."""
    if "gb" not in _CT:
        d = Path(tempfile.mkdtemp())
        for f in ("gpu_budget.py", "gpu_shared.py"):
            with urllib.request.urlopen(CT_RAW + f, timeout=30) as r:
                (d / f).write_bytes(r.read())
        sys.path.insert(0, str(d))
        import importlib
        _CT["gb"], _CT["d"] = importlib.import_module("gpu_budget"), d
    gb, d = _CT["gb"], _CT["d"]
    gb.LEDGER = d / "ledger.json"
    with urllib.request.urlopen("https://raw.githubusercontent.com/edbackulin-eng/closed-tabs/main/state/"
                                "gpu_ledger.json", timeout=30) as r:
        gb.LEDGER.write_bytes(r.read())
    gb.WHATIF_CACHE = d / "whatif_gpu.json"
    ct, wi = gb.closed_tabs_hours(), gb.whatif_hours()
    own = own_week_hours()
    total = ct + wi + gb.WHATIF_HOURS_PER_RUN + own + job_hours
    line = (f"week: Closed Tabs {ct:.2f} h + What If ~{wi:.2f} h + What If reserve "
            f"{gb.WHATIF_HOURS_PER_RUN:.1f} h + dinosaurs so far {own:.2f} h + this job {job_hours:.2f} h "
            f"= {total:.2f} h of {ACCOUNT_LIMIT_H} h")
    return total <= ACCOUNT_LIMIT_H, line


def own_week_hours(now=None):
    since = (now or datetime.now(timezone.utc)) - timedelta(days=7)
    rows = json.loads(LEDGER.read_text(encoding="utf-8")) if LEDGER.exists() else []
    return sum(r["hours"] for r in rows if datetime.fromisoformat(r["start"]) >= since)


def record(ref, started, finished):
    """The run into this channel's ledger, in the shared format."""
    rows = json.loads(LEDGER.read_text(encoding="utf-8")) if LEDGER.exists() else []
    rows.append({"ref": ref, "start": started.isoformat(), "end": finished.isoformat(),
                 "hours": round((finished - started).total_seconds() / 3600, 3)})
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    LEDGER.write_text(json.dumps(rows, indent=1), encoding="utf-8")


def kernel_watchdog():
    import watchdog
    return watchdog.WATCHDOG


def job_line(job):
    """JOB as a line of Python. Not json.dumps pasted in: JSON's true/false/null
    are not Python (28.10: the German voice kernel died on "required": true)."""
    return f"import json as _json\nJOB = _json.loads({json.dumps(json.dumps(job, ensure_ascii=False))})"


def build(task, slug, folder):
    folder.mkdir(parents=True, exist_ok=True)
    if task == "recon":
        import recon_job
        spec = json.loads((ROOT / "episodes" / slug / "recon.json").read_text(encoding="utf-8"))
        # FLUX only: real photographs are fetched and layered on the CPU later
        job = {k: spec[k] for k in ("style", "prompts", "thumbs")}
        job["two_seeds"] = spec.get("two_seeds", [])
        job["seeds_per_prompt"] = n = spec.get("seeds_per_prompt", 1)
        # the ids this job makes: an earlier run's output (other prompts) never counts
        want = [f"{k}__s{i}" for k in job["prompts"] for i in range(max(n, 2 if k in job["two_seeds"] else 1))]
        want += [f"{k}__s{i}" for k in job["thumbs"] for i in range(2)]
        if len(want) > MAX_FLUX_FRAMES:
            raise SystemExit(f"{len(want)} FLUX frames asked for; the plan allows {MAX_FLUX_FRAMES} an episode")
        # The Hugging Face token goes into the notebook at submit time and into
        # nothing else: no job file, no repository (as whatif-channel's
        # kaggle/push.py does it). The kernel is private, and the token is only
        # used if the weight mirror cannot be found.
        token = os.environ.get("HF_TOKEN", "").strip()
        if token:
            job["hf_token"] = token
        cell = f"{job_line(job)}\n{recon_job.flux_source()}"
        sources = [f"{user()}/whatif-flux-weights"]          # the 4-bit FLUX mirror, read only
    elif task == "animate":
        # LTX-Video image-to-video on the FLUX frames the species check passed
        import animate_job
        job = json.loads((ROOT / "gpu" / "animate_job.json").read_text(encoding="utf-8"))
        want = [f["id"] for f in job["frames"]]
        cell = f"{job_line(job)}\n{kernel_watchdog()}\n{animate_job.SOURCE}"
        sources = [f"{user()}/{SLUGS['recon']}"]           # the FLUX frames, read only
    else:
        raise SystemExit(f"unknown task {task}")
    nb = {"cells": [{"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                     "source": cell.splitlines(keepends=True)}],
          "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
          "nbformat": 4, "nbformat_minor": 5}
    (folder / "notebook.ipynb").write_text(json.dumps(nb), encoding="utf-8")
    meta = {"id": f"{user()}/{SLUGS[task]}", "title": SLUGS[task], "code_file": "notebook.ipynb",
            "language": "python", "kernel_type": "notebook", "is_private": True, "enable_gpu": True,
            "enable_internet": True, "dataset_sources": [], "competition_sources": [],
            "kernel_sources": sources}
    (folder / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta["id"], want


def lost_race(ref):
    """True if What If or Closed Tabs pushed within 5 minutes of us: we yield."""
    time.sleep(20)
    rows = kernels() or []
    mine = next((t for r, t in rows if r == ref), datetime.now(timezone.utc))
    for r, t in rows:
        if r.split("/")[-1].startswith(OTHERS) and abs((t - mine).total_seconds()) < 300:
            if status(r) in ("running", "queued", "pending"):
                say(f"{r} pushed alongside; withdrawing {ref}")
                kaggle("kernels", "delete", ref, "-y")
                return True
    return False


def wait_done(ref, timeout_s):
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        st = status(ref)
        if st == "complete":
            return True
        if st in ("error", "cancelacknowledged", "cancelrequested"):
            say(f"{ref} ended: {st}")
            return False
        time.sleep(60)
    say(f"{ref} still {status(ref)} after {timeout_s / 3600:.1f} h")
    return False


def validate(task, out, want):
    """(ok, why). A kernel counts as done only by what it produced: Kaggle marks
    a session COMPLETE even when every image in it failed (25.10: FLUX hit a
    401, the run still read as a success, and voice and render went ahead with
    nothing to show)."""
    rp = out / "report.json"
    if not rp.exists():
        return False, "no report.json in the kernel output"
    r = json.loads(rp.read_text(encoding="utf-8"))
    errors = r.get("errors") or []
    if task == "animate":
        # animation is a bonus: what came out is used, a frame that failed stays a still
        clips = [c for c in r.get("clips", []) if c.get("ok")]
        ok = "frames_found" in r and (r["frames_found"] > 0 or not want)
        return ok, f"{len(clips)}/{len(want)} clips, errors: {(r.get('errors') or [])[:3]}"
    if task == "recon":
        ids = set(want) if isinstance(want, (list, tuple, set)) else None
        n_want = len(ids) if ids is not None else want
        got = len([i for i in (r.get("images") or []) if ids is None or i.get("id") in ids])
        files = sum(len(list((out / d).glob("*.jpg"))) for d in ("raw", "up") if (out / d).exists())
        need = math.ceil(0.9 * n_want)
        ok = got >= need and not errors
        return ok, (f"{got}/{n_want} images (need {need}), {files} image files, "
                    f"weights: {r.get('weights_source')}, errors: {errors[:3]}")
    return False, f"unknown task {task}"


def pull(ref, out):
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True, exist_ok=True)
    say(kaggle("kernels", "output", ref, "-p", str(out), timeout=1800)[-300:])


def main(task, slug):
    """0: done and valid. 3: no free slot yet (the workflow retries). 1: the
    kernel ran and its output is not good enough - the chain stops here."""
    ref = f"{user()}/{SLUGS[task]}"
    out = ROOT / "gpu" / f"results_{task}"
    usage_p = ROOT / "state" / "kaggle_once.json"
    usage = json.loads(usage_p.read_text()) if usage_p.exists() else []
    folder = Path(tempfile.mkdtemp())
    _, want = build(task, slug, folder)
    job_h = JOB_HOURS.get(task, 0.5)
    if task == "recon":
        job_h = round(FLUX_OVERHEAD_H + len(want) * FLUX_S_PER_FRAME / 3600, 2)
    if task == "animate":
        if not want:
            say("nothing to animate")
            return 0
        job_h = round(0.2 + len(want) * LTX_H_PER_CLIP, 2)
    st = status(ref)
    started = None
    if st == "complete":
        pull(ref, out)
        ok, why = validate(task, out, want)
        if ok:
            say(f"{ref} already finished and its output is good: {why}")
            return 0
        say(f"{ref} finished earlier but its output is not good enough ({why}); running it again")
        st = None
    if st in ("running", "queued", "pending"):
        say(f"{ref} already {st}; waiting for it")
        started = datetime.now(timezone.utc)
    else:
        t0, idle_since = time.time(), None
        while True:
            if time.time() - t0 > MAX_WAIT_S:
                say("no free slot in this run; the workflow will try again")
                return 3
            why = in_their_hours()
            if why:
                idle_since = None
                say(f"waiting: {why}")
                time.sleep(15 * 60)
                continue
            ok, line = budget(job_h)
            if not ok:
                idle_since = None
                say(f"waiting: {line}")
                time.sleep(15 * 60)
                continue
            busy = others_busy() or (["What If build on GitHub"] if whatif_building() else [])
            if busy:
                idle_since = None
                say(f"waiting: busy {', '.join(busy)}")
                time.sleep(60)
                continue
            idle_since = idle_since or time.time()
            if time.time() - idle_since < IDLE_S:
                time.sleep(60)
                continue
            say(line)
            say(kaggle("kernels", "push", "-p", str(folder)).strip())
            started = datetime.now(timezone.utc)
            if lost_race(ref):
                idle_since, started = None, None
                continue
            break
    finished = wait_done(ref, 3 * 3600)
    pull(ref, out)
    ok, why = validate(task, out, want) if finished else (False, f"kernel ended as {status(ref)}")
    ended = datetime.now(timezone.utc)
    usage.append({"task": task, "ref": ref, "start": started.isoformat(), "end": ended.isoformat(),
                  "hours": round((ended - started).total_seconds() / 3600, 3), "ok": ok, "check": why})
    record(ref, started, ended)
    usage_p.parent.mkdir(exist_ok=True)
    usage_p.write_text(json.dumps(usage, indent=1))
    say(("output good: " if ok else "OUTPUT NOT GOOD ENOUGH - stopping the chain: ") + why)
    return 0 if ok else 1


def fetch(tasks=("recon", "animate")):
    """Pull the kernels' outputs: FLUX frames and LTX clips, for the CPU layers matrix."""
    for task in tasks:
        out = ROOT / "gpu" / f"results_{task}"
        out.mkdir(parents=True, exist_ok=True)
        say(kaggle("kernels", "output", f"{user()}/{SLUGS[task]}", "-p", str(out), timeout=1800)[-300:])
    return 0


if __name__ == "__main__":
    if sys.argv[1] == "fetch":
        sys.exit(fetch(tuple(sys.argv[2:]) or ("recon", "animate")))
    sys.exit(main(sys.argv[1], sys.argv[2]))
