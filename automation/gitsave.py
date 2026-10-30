"""Commit and push a few files at once, with retries: records that must never be
lost between two steps (the YouTube id of an upload above all).

    from automation import gitsave
    gitsave.save(["episodes/x/uploaded.json", "state/episodes.json"], "Uploaded x")

Outside GitHub Actions (no GITHUB_ACTIONS) it only writes nothing and returns
False: a local run keeps its files and the owner commits them.
"""
import os
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _git(*args, check=True):
    return subprocess.run(["git", *args], cwd=ROOT, check=check, capture_output=True, text=True)


def save(paths, message, tries=6):
    if not os.environ.get("GITHUB_ACTIONS"):
        return False
    _git("config", "user.name", "github-actions[bot]")
    _git("config", "user.email", "41898282+github-actions[bot]@users.noreply.github.com")
    _git("add", "--", *[str(p) for p in paths], check=False)
    if _git("diff", "--cached", "--quiet", check=False).returncode == 0:
        return True
    _git("commit", "-q", "-m", message)
    for i in range(tries):
        pulled = _git("pull", "-q", "--rebase", "--autostash", "-X", "theirs", "origin", "main", check=False)
        if pulled.returncode != 0:
            _git("rebase", "--abort", check=False)
        elif _git("push", "-q", "origin", "HEAD:main", check=False).returncode == 0:
            return True
        time.sleep(5 * (i + 1))
    raise RuntimeError(f"could not push {message!r} after {tries} tries")
