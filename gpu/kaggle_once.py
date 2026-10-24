"""One-off: this episode's GPU work on Kaggle, in What If's and Closed Tabs' free hours.

    python gpu/kaggle_once.py recon living-fossils
    python gpu/kaggle_once.py voice living-fossils
    python gpu/kaggle_once.py fetch                  # outputs only, for the render

The channel's GPU home is Modal (gpu/modal_app.py); this exists because Modal
runs no GPU without a card, and the test should not wait for one. It uses the
same kernel code (gpu/kernels/) on the shared Kaggle account, and it goes
strictly last:
  - never in Closed Tabs' window (20:00-04:00 UTC), never on a What If build
    day (Mon, Tue, Thu, Fri, Sat) before 17:30 UTC - its worst-case finish;
  - only after no What If or Closed Tabs kernel has been queued or running for
    15 minutes, and no What If build is in progress on GitHub;
  - only if the shared week stays under 25 h with What If's next build held in
    reserve - measured by Closed Tabs' own kaggle/gpu_budget.py, fetched from
    its repository, so the accounting is theirs, not a copy;
  - and if either of them pushed in the same minute, this kernel is deleted
    and the wait starts again.
Exit codes: 0 done, 3 not yet (the workflow restarts itself), other = failure.
"""
import json
import os
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
JOB_HOURS = {"recon": 0.75, "voice": 0.3}
SLUGS = {"recon": "prehistoric-recon", "voice": "prehistoric-voice"}
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
    total = ct + wi + gb.WHATIF_HOURS_PER_RUN + job_hours
    line = (f"week: Closed Tabs {ct:.2f} h + What If ~{wi:.2f} h + What If reserve "
            f"{gb.WHATIF_HOURS_PER_RUN:.1f} h + this job {job_hours:.2f} h = {total:.2f} h")
    return total <= gb.WEEKLY_LIMIT_H, line


def build(task, slug, folder):
    folder.mkdir(parents=True, exist_ok=True)
    if task == "recon":
        import recon_job
        spec = json.loads((ROOT / "episodes" / slug / "recon.json").read_text(encoding="utf-8"))
        job = {k: spec[k] for k in ("style", "seeds", "prompts", "thumbs", "photos")}
        job["two_seeds"] = spec.get("two_seeds", [])
        cell = f"JOB = {json.dumps(job, indent=1)}\n{recon_job.SOURCE}"
        sources = [f"{user()}/whatif-flux-weights"]          # the 4-bit FLUX mirror, read only
    else:
        import voice_job
        sys.path.insert(0, str(ROOT / "gpu"))
        import modal_app
        job = modal_app.voice_job_dict(slug)
        cell = f"JOB = {json.dumps(job, indent=1)}\n{voice_job.WATCHDOG}\n{voice_job.body()}"
        sources = []
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
    return meta["id"]


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


def main(task, slug):
    ref = f"{user()}/{SLUGS[task]}"
    out = ROOT / "gpu" / f"results_{task}"
    usage_p = ROOT / "state" / "kaggle_once.json"
    usage = json.loads(usage_p.read_text()) if usage_p.exists() else []
    st = status(ref)
    started = None
    if st == "complete" and any(u["task"] == task and u.get("ok") for u in usage):
        say(f"{ref} already finished")
    elif st in ("running", "queued", "pending"):
        say(f"{ref} already {st}; waiting for it")
        started = datetime.now(timezone.utc)
    else:
        folder = Path(tempfile.mkdtemp())
        build(task, slug, folder)
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
            ok, line = budget(JOB_HOURS[task])
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
    ok = True
    if started:
        ok = wait_done(ref, 3 * 3600)
        usage.append({"task": task, "ref": ref, "start": started.isoformat(),
                      "end": datetime.now(timezone.utc).isoformat(),
                      "hours": round((datetime.now(timezone.utc) - started).total_seconds() / 3600, 3),
                      "ok": ok})
        usage_p.parent.mkdir(exist_ok=True)
        usage_p.write_text(json.dumps(usage, indent=1))
    out.mkdir(parents=True, exist_ok=True)
    say(kaggle("kernels", "output", ref, "-p", str(out), timeout=1800)[-500:])
    return 0 if ok else 1


def fetch():
    """Pull both kernels' outputs (for the render's prepare step)."""
    for task in ("recon", "voice"):
        out = ROOT / "gpu" / f"results_{task}"
        out.mkdir(parents=True, exist_ok=True)
        say(kaggle("kernels", "output", f"{user()}/{SLUGS[task]}", "-p", str(out), timeout=1800)[-300:])
    return 0


if __name__ == "__main__":
    if sys.argv[1] == "fetch":
        sys.exit(fetch())
    sys.exit(main(sys.argv[1], sys.argv[2]))
