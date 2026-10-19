"""Push a GPU job to Kaggle, wait for it, and pull the results back.

Kaggle gives roughly 30 GPU-hours a week on a P100 or two T4s. That is far more
than this channel needs (a video's images are minutes of GPU time), and it is the
only free way to run FLUX and Chatterbox at all on a machine with no CUDA.

Needs ~/.kaggle/kaggle.json (Kaggle > Settings > API > Create New Token).
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BUILD = ROOT / "build"
RESULTS = ROOT / "results"


def _notebook(job):
    """A one-cell notebook: the job dict, then the template body.

    The Hugging Face token is injected here, at submit time, and never stored in
    a job file. Those files are generated artefacts that would otherwise sit in
    the repository with a live credential in them - and this repository is
    public, so that is a leak, not an inconvenience.
    """
    from job_template import SOURCE
    job = dict(job)
    token = os.environ.get("HF_TOKEN") or _env_token()
    if token and job.get("image_prompts"):
        job["hf_token"] = token
    cell = f"JOB = {json.dumps(job, indent=2)}\n\n{SOURCE}"
    return {
        "cells": [{"cell_type": "code", "metadata": {}, "execution_count": None,
                   "outputs": [], "source": cell.splitlines(keepends=True)}],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python",
                           "name": "python3"},
            "language_info": {"name": "python", "version": "3.11.0"},
        },
        "nbformat": 4, "nbformat_minor": 5,
    }


def _env_token():
    """HF_TOKEN from .env, for local runs. In CI it arrives as a secret."""
    env = ROOT.parent / ".env"
    if not env.exists():
        return None
    for line in env.read_text(encoding="utf-8-sig").splitlines():
        key, _, value = line.partition("=")
        if key.strip() == "HF_TOKEN":
            return value.strip().strip(chr(34)).strip(chr(39)) or None
    return None


def _metadata(username, slug, title):
    return {
        "id": f"{username}/{slug}",
        "title": title,
        "code_file": "notebook.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": True,   # needed to pull model weights from Hugging Face
        "dataset_sources": [],
        "competition_sources": [],
        # The mirrored weights ride in as a kernel input, so a generation run
        # never contacts huggingface.co.
        "kernel_sources": ([f"{username}/whatif-flux-weights"]
                           if os.environ.get("FLUX_MIRROR", "1") != "0" else []),
    }


def _run(args, check=True):
    # The Kaggle CLI prints non-ASCII; under a Windows cp1251 console it crashed
    # while writing a downloaded log, after the kernel itself had finished.
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"}
    try:
        result = subprocess.run(args, capture_output=True, text=True, encoding="utf-8",
                                errors="replace", env=env, timeout=120)
    except subprocess.TimeoutExpired:
        if check:
            raise RuntimeError(f"{' '.join(args)} timed out")
        return subprocess.CompletedProcess(args, 1, "", "timed out")
    if check and result.returncode != 0:
        raise RuntimeError(f"{' '.join(args)} failed:\n{result.stdout}\n{result.stderr}")
    return result


def username():
    # The environment comes first. A CI runner has no home full of dotfiles:
    # credentials arrive as KAGGLE_USERNAME and KAGGLE_KEY. Reading only
    # ~/.kaggle/kaggle.json made every cloud build die at the first GPU
    # submission, after twenty minutes of useful work had already happened.
    env_user = os.environ.get("KAGGLE_USERNAME")
    if env_user:
        return env_user
    cfg = Path.home() / ".kaggle" / "kaggle.json"
    if cfg.exists():
        return json.loads(cfg.read_text())["username"]
    # RuntimeError, not SystemExit: SystemExit derives from BaseException and
    # slips straight through `except Exception`, so a caller that meant to
    # degrade gracefully gets killed instead.
    raise RuntimeError(
        "No Kaggle credentials: set KAGGLE_USERNAME and KAGGLE_KEY, "
        f"or save kaggle.json to {cfg}")


def _slugify(title):
    """Kaggle derives the kernel slug from the TITLE, not from the id you send.

    Passing a title that slugifies to something else silently creates the kernel
    at the other slug, and every later status/output call then fails with a
    permission error that looks like an auth problem. So the slug is computed
    from the title here and the id is made to match it.
    """
    cleaned = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")
    return re.sub(r"-{2,}", "-", cleaned)


# Kernels of both channels share one Kaggle account and its GPU quota. Closed Tabs
# never submits while either channel has a kernel queued or running. The What If
# repository is not changed, so the lock is one-sided: the daily schedule keeps
# Closed Tabs well clear of What If's 06:00 UTC build, and this check covers the
# rest (a late What If run, a manual one, a Closed Tabs retry).
SHARED_PREFIXES = ("whatif", "closedtabs", "prehistoric")
BUSY_WORDS = ("running", "queued", "pending")
GPU_WAIT_TIMEOUT_S = 4 * 3600


WHATIF_RUNS = ("https://api.github.com/repos/edbackulin-eng/whatif-channel/actions/"
               "workflows/build-video.yml/runs?per_page=5")


def whatif_building():
    """Is a What If build running or queued? Its build needs the GPU for hours,
    and a Closed Tabs kernel started first would make it wait until it times out
    (17.10: the Saturday What If build was cut off that way)."""
    import urllib.request
    headers = {"User-Agent": "prehistoric-channel", "Accept": "application/vnd.github+json"}
    if os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN"):
        headers["Authorization"] = f"Bearer {os.environ.get('GH_TOKEN') or os.environ.get('GITHUB_TOKEN')}"
    try:
        req = urllib.request.Request(WHATIF_RUNS, headers=headers)
        with urllib.request.urlopen(req, timeout=30) as r:
            runs = json.loads(r.read().decode())["workflow_runs"]
    except Exception as e:                                       # noqa: BLE001
        print(f"  could not see What If's builds ({e}); going by the kernels alone", flush=True)
        return False
    return any(r["status"] in ("in_progress", "queued", "waiting") for r in runs)


def busy_kernels():
    """Refs of kernels of any channel that are queued, running or unknown."""
    import gpu_shared
    return [r for r, _, _ in gpu_shared.busy_kernels()]


def wait_for_free_gpu(poll_s=60, timeout_s=GPU_WAIT_TIMEOUT_S):
    """Block until no kernel of either channel is queued or running."""
    started = time.time()
    while True:
        busy = busy_kernels()
        if not busy and whatif_building():
            busy = ["What If build running (GitHub Actions)"]
        if not busy:
            return
        if time.time() - started > timeout_s:
            raise RuntimeError(f"Kaggle stayed busy for {timeout_s // 3600} h: {busy}")
        print(f"  Kaggle busy with {', '.join(busy)}; waiting {poll_s}s", flush=True)
        time.sleep(poll_s)


IDLE_FOR_S = 15 * 60


def submit_when_idle(build_dir, job_hours, poll_s=60, timeout_s=20 * 3600):
    """Push a prepared kernel folder when the shared week has room and the GPU
    has been free IDLE_FOR_S; if another channel won a push race, wait again.
    The lock itself is kaggle/gpu_shared.py, identical in all three channels."""
    import gpu_budget
    import gpu_shared
    started, last_why = time.time(), None
    while time.time() - started < timeout_s:
        ok, why = gpu_budget.check(job_hours)
        if why != last_why:
            print(f"  [{time.strftime('%H:%M:%S')}] {why}", flush=True)
            last_why = why
        if not ok:
            time.sleep(15 * 60)
            continue
        if not gpu_shared.wait_for_free(IDLE_FOR_S, poll_s, max_wait_s=2 * 3600,
                                        log=lambda m: print(f"  [{time.strftime('%H:%M:%S')}]{m}", flush=True)):
            continue
        if whatif_building():
            print("  What If build in progress; waiting", flush=True)
            time.sleep(poll_s)
            continue
        meta = json.loads((Path(build_dir) / "kernel-metadata.json").read_text(encoding="utf-8"))
        out = _run([sys.executable, "-m", "kaggle", "kernels", "push", "-p", str(build_dir)])
        print(out.stdout.strip(), flush=True)
        if gpu_shared.after_push(meta["id"], log=lambda m: print(m, flush=True)):
            return meta["id"]
    raise RuntimeError("the GPU never came free inside the time-out")


def publish_ledger():
    """Commit and push state/gpu_ledger.json so the other channels count these hours."""
    repo = ROOT.parent
    for cmd in (["git", "add", "state/gpu_ledger.json"],
                ["git", "commit", "-m", "gpu ledger: a Kaggle job finished"],
                ["git", "pull", "--rebase", "-q", "origin", "main"],
                ["git", "push", "-q", "origin", "HEAD:main"]):
        r = subprocess.run(cmd, cwd=repo, capture_output=True, text=True)
        if r.returncode and cmd[1] != "commit":
            print(f"  ledger not published ({' '.join(cmd)}): {r.stderr[-300:]}", flush=True)
            return False
    return True


def push(job, slug, title, job_hours=None, force_window=False):
    import gpu_budget
    ok, why = gpu_budget.check(job_hours or gpu_budget.DEFAULT_JOB_H)
    if not ok:
        raise RuntimeError(f"not submitting to Kaggle: {why}")
    print(why)
    wait_for_free_gpu()
    BUILD.mkdir(parents=True, exist_ok=True)
    derived = _slugify(title)
    if derived != slug:
        log_slug = slug
        slug = derived
        print(f"note: Kaggle slugifies the title, so the kernel id becomes "
              f"{slug!r}, not {log_slug!r}")
    user = username()
    (BUILD / "notebook.ipynb").write_text(
        json.dumps(_notebook(job), indent=1), encoding="utf-8")
    (BUILD / "kernel-metadata.json").write_text(
        json.dumps(_metadata(user, slug, title), indent=2), encoding="utf-8")
    out = _run([sys.executable, "-m", "kaggle", "kernels", "push", "-p", str(BUILD)])
    print(out.stdout.strip())
    return f"{user}/{slug}"


def wait(ref, poll_s=30, timeout_s=3 * 3600):
    """Block until the kernel stops running, and log its GPU time to the ledger."""
    import gpu_budget
    from datetime import datetime, timezone
    began = datetime.now(timezone.utc)
    try:
        return _wait(ref, poll_s, timeout_s)
    finally:
        gpu_budget.record(ref, began, datetime.now(timezone.utc))


def _wait(ref, poll_s, timeout_s):
    started = time.time()
    while time.time() - started < timeout_s:
        result = _run([sys.executable, "-m", "kaggle", "kernels", "status", ref],
                      check=False)
        text = (result.stdout + result.stderr).strip()
        status = text.lower()
        elapsed = int(time.time() - started)
        print(f"  [{elapsed:5d}s] {text[:110]}", flush=True)
        # Only the status line counts. A dropped connection prints a traceback
        # that mentions errors without the kernel having failed at all.
        if "kernelworkerstatus.complete" in status:
            return True
        if "kernelworkerstatus.error" in status or "kernelworkerstatus.cancel" in status:
            return False
        time.sleep(poll_s)
    print("timed out waiting for the kernel")
    return False


def fetch(ref, dest):
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    _run([sys.executable, "-m", "kaggle", "kernels", "output", ref, "-p", str(dest)])
    files = sorted(p for p in dest.rglob("*") if p.is_file())
    print(f"\npulled {len(files)} file(s) into {dest}:")
    for f in files:
        print(f"  {f.relative_to(dest)}  ({f.stat().st_size / 1e6:.2f} MB)")
    report = dest / "report.json"
    if report.exists():
        print("\n==== report.json ====")
        print(report.read_text())
    return dest


def main():
    ap = argparse.ArgumentParser(description="run a GPU job on Kaggle")
    ap.add_argument("job_file", help="JSON file with the job spec")
    ap.add_argument("--slug", default="closedtabs-gpu-job")
    ap.add_argument("--title", default="WhatIf GPU job")
    ap.add_argument("--no-wait", action="store_true")
    args = ap.parse_args()

    job = json.loads(Path(args.job_file).read_text(encoding="utf-8"))
    ref = push(job, args.slug, args.title)
    print(f"pushed {ref}\nhttps://www.kaggle.com/code/{ref}")
    if args.no_wait:
        return 0
    print("\nwaiting for the run to finish...")
    if not wait(ref):
        print("kernel did not complete; check the link above")
        return 1
    fetch(ref, RESULTS / args.slug)
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))
    sys.exit(main())
