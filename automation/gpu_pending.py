"""GPU work handed to the watchdog until the free Kaggle window opens (owner 06.11).

    python automation/gpu_pending.py relay <workflow> key=value ...   # a workflow's relay step
    python automation/gpu_pending.py list

The dinosaurs may use the shared Kaggle GPU only when What If and Closed Tabs do not: on
Wednesday and Sunday from 04:00 UTC, on What If's build days (Mon, Tue, Thu, Fri, Sat) from
17:30 UTC, and never from 20:00 (gpu/kaggle_once.py in_their_hours). A run used to wait up
to 4 h 10 for that and then restart itself, so its waits ran on a 4 h 10 rhythm of their own
that had nothing to do with the window: a run could end at 17:20, the next start at 17:25,
ready only after its setup, and on 05.11 the relays ran from 11:43 to 09:00 next day.

Now gpu/kaggle_once.py works out when the window opens. A run that would still be waiting
when its own wait ends does not wait at all: it writes the opening to gpu/next_open.txt and
exits 3. Its relay step calls this, which puts the restart into state/gpu_pending.json for
LEAD before the opening instead of starting it at once. The watchdog (every 15 minutes)
starts what is due, the run does its setup and kaggle_once sleeps the rest to the minute
the window opens. Without a gpu/next_open.txt (busy, budget, a race) the relay is as before:
at once.
"""
import json
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PENDING = ROOT / "state" / "gpu_pending.json"
NEXT_OPEN = ROOT / "gpu" / "next_open.txt"
LEAD = timedelta(minutes=20)          # setup before the GPU step: checkout, pip, artifacts, checks


def ts(s):
    return datetime.fromisoformat(str(s).replace("Z", "+00:00"))


def load():
    return json.loads(PENDING.read_text(encoding="utf-8")) if PENDING.exists() else []


def save(rows):
    PENDING.parent.mkdir(parents=True, exist_ok=True)
    PENDING.write_text(json.dumps(rows, indent=1) + "\n", encoding="utf-8")


def add(workflow, inputs, opens, why=""):
    """One pending restart a workflow and episode: a newer one replaces it."""
    rows = [r for r in load() if not (r["workflow"] == workflow and r["inputs"].get("slug") == inputs.get("slug"))]
    rows.append({"workflow": workflow, "inputs": inputs, "opens": opens.strftime("%Y-%m-%dT%H:%M:%SZ"),
                 "at": (opens - LEAD).strftime("%Y-%m-%dT%H:%M:%SZ"), "why": why})
    save(rows)
    return rows[-1]


def take_due(now=None):
    """The pending restarts that are due, removed from the file."""
    now = now or datetime.now(timezone.utc)
    rows = load()
    due = [r for r in rows if ts(r["at"]) <= now]
    if due:
        save([r for r in rows if r not in due])
    return due


def waiting_for(workflow, now=None):
    """Pending restarts of this workflow (the watchdog starts no second build meanwhile)."""
    return [r for r in load() if r["workflow"] == workflow]


def dispatch(repo, workflow, inputs):
    args = ["gh", "workflow", "run", workflow, "-R", repo]
    for k, v in inputs.items():
        args += ["-f", f"{k}={v}"]
    subprocess.run(args, check=True)


def relay(repo, workflow, inputs):
    opens = NEXT_OPEN.read_text(encoding="utf-8").strip() if NEXT_OPEN.exists() else ""
    if not opens:
        dispatch(repo, workflow, inputs)
        return "relayed now"
    row = add(workflow, inputs, ts(opens), "the free GPU window opens after this run's wait")
    return f"pending: the watchdog starts it at {row['at']} (the window opens {row['opens']})"


def main(argv):
    import os
    if argv[:1] == ["relay"] and len(argv) >= 2:
        repo = os.environ.get("GITHUB_REPOSITORY", "edbackulin-eng/prehistoric-channel")
        inputs = dict(a.split("=", 1) for a in argv[2:])
        print(relay(repo, argv[1], inputs))
        return 0
    if argv[:1] == ["list"]:
        print(json.dumps(load(), indent=1))
        return 0
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
