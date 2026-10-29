"""The watchdog: keeps the German channel running with nobody watching.

    python automation/watchdog.py            (from .github/workflows/watchdog.yml)

Acts only through GitHub Actions and the YouTube API. Each pass:
  1. an episode or render run that hangs far longer than any build is cancelled;
  2. a failed or cancelled run has its failed jobs rerun (the jobs that passed
     keep their artifacts), up to MAX_ATTEMPTS; after that the owner gets an
     email - a comment on the open "run-log" issue that mentions them;
  3. if fewer than `buffer` (config/channel.json) episodes are ahead - built
     and waiting, or scheduled - and nothing is being built, the next episode
     is started (episode.yml; the GPU rules are kept inside it);
  4. an episode waiting in the buffer is uploaded as soon as YT_TOKEN_JSON
     exists (upload.yml), on the nearest free slot;
  5. once a day the state is committed (state/heartbeat.json), so GitHub never
     switches the schedules off for a quiet repository.
Every email is recorded in state/watchdog.json so one failure is reported once.
"""
import json
import os
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from core.logger import get_logger  # noqa: E402

log = get_logger("watchdog")
REPO = os.environ.get("GITHUB_REPOSITORY", "edbackulin-eng/prehistoric-channel")
OWNER = REPO.split("/")[0]
CFG = json.loads((ROOT / "config" / "channel.json").read_text(encoding="utf-8"))
WATCHED = {"episode.yml": 7, "render.yml": 10, "upload.yml": 3,      # hours before a run counts as hung
           "restyle.yml": 3}       # a script being retold: no build of it may start meanwhile
MAX_ATTEMPTS = 3
SETTLE_MIN = 12
STATE = ROOT / "state" / "watchdog.json"
HEARTBEAT = ROOT / "state" / "heartbeat.json"


def now():
    return datetime.now(timezone.utc)


def ts(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def gh(*args, parse=True):
    r = subprocess.run(["gh", *args], capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"gh {' '.join(args)}: {r.stderr.strip()[:300]}")
    return json.loads(r.stdout) if parse and r.stdout.strip() else r.stdout


def runs(workflow, n=15):
    return gh("api", f"repos/{REPO}/actions/workflows/{workflow}/runs?per_page={n}")["workflow_runs"]


def memory():
    return json.loads(STATE.read_text(encoding="utf-8")) if STATE.exists() else {"emailed": []}


def email(key, subject, body, mem):
    if key in mem["emailed"]:
        return
    issues = gh("issue", "list", "-R", REPO, "--label", "run-log", "--state", "open", "--json", "number")
    if issues:
        n = issues[0]["number"]
    else:
        subprocess.run(["gh", "label", "create", "run-log", "-R", REPO, "--color", "b60205"], capture_output=True)
        url = gh("issue", "create", "-R", REPO, "--title", "Run failures", "--label", "run-log",
                 "--body", "A comment here mentioning the owner arrives as an email.", parse=False)
        n = url.strip().rsplit("/", 1)[-1]
    gh("issue", "comment", str(n), "-R", REPO, "--body", f"@{OWNER} **{subject}**\n\n{body}", parse=False)
    mem["emailed"].append(key)
    log.warning("emailed: %s", subject)


def failed_tail(run_id):
    try:
        tail = gh("run", "view", str(run_id), "-R", REPO, "--log-failed", parse=False)
        return "\n".join(l.split("Z ", 1)[-1] for l in tail.strip().splitlines()[-25:])
    except Exception as e:                                           # noqa: BLE001
        return f"(log unavailable: {e})"


def check_runs(mem, actions):
    """Steps 1 and 2. True when something is being built."""
    busy = False
    for wf, hung_h in WATCHED.items():
        rs = runs(wf)
        for r in rs:
            if r["status"] in ("in_progress", "queued", "waiting", "pending", "requested"):
                age = now() - ts(r["run_started_at"] or r["created_at"])
                if age > timedelta(hours=hung_h):
                    gh("run", "cancel", str(r["id"]), "-R", REPO, parse=False)
                    mem.setdefault("cancelled", []).append(r["id"])
                    actions.append(f"cancelled hung {wf} run {r['id']} ({age})")
                elif wf != "upload.yml":
                    busy = True
        # only the newest run of a workflow is recovered: an older failed or
        # cancelled one that a newer run has replaced is history, not work
        latest = rs[0] if rs else None
        if not latest or latest["status"] != "completed" or                 (latest["conclusion"] not in ("failure", "timed_out")
                 and not (latest["conclusion"] == "cancelled" and latest["id"] in mem.get("cancelled", []))):
            continue                       # a run cancelled by hand is left alone
        if now() - ts(latest["updated_at"]) < timedelta(minutes=SETTLE_MIN):
            busy = True
            continue
        rid, attempt = latest["id"], latest.get("run_attempt", 1)
        if attempt < MAX_ATTEMPTS:
            try:
                gh("run", "rerun", str(rid), "-R", REPO, "--failed", parse=False)
            except RuntimeError:
                gh("run", "rerun", str(rid), "-R", REPO, parse=False)
            actions.append(f"reran {wf} {rid} (attempt {attempt + 1})")
            busy = busy or wf != "upload.yml"
        else:
            email(f"{wf}-{rid}", f"Urzeit-Kanal: {wf} run {rid} failed {attempt} times",
                  f"Run: https://github.com/{REPO}/actions/runs/{rid}\n\n```\n{failed_tail(rid)}\n```", mem)
    return busy


def check_buffer(busy, actions):
    from automation import release
    from automation import llm_budget
    have = release.ahead()
    d = llm_budget.deferred_now()
    if d:
        actions.append(f"{d['slug']} waits for the quota day (until {d['until']})")
        return have
    if len(have) < CFG["buffer"] and not busy:
        gh("workflow", "run", "episode.yml", "-R", REPO, parse=False)
        actions.append(f"only {len(have)} ahead; next episode started")
    return have


def check_uploads(actions):
    from automation import release
    from pipeline import youtube
    if not youtube.available():
        return
    for row in release.rows():
        if row["status"] == "built" and not (ROOT / "episodes" / row["slug"] / "uploaded.json").exists():
            gh("workflow", "run", "upload.yml", "-R", REPO, "-f", f"slug={row['slug']}", "-f",
               f"run_id={row['run_id']}", parse=False)
            actions.append(f"upload of {row['slug']} started")
            return                                         # one at a time


def main():
    mem = memory()
    actions = []
    if "--test-email" in sys.argv:
        email(f"test-{now():%Y%m%d%H%M}", "Urzeit-Kanal: test email from the watchdog",
              "If this arrived by email, the failure notices will arrive too.", mem)
        STATE.write_text(json.dumps(mem, indent=1), encoding="utf-8")
        return 0
    try:
        busy = check_runs(mem, actions)
    except Exception as e:                                           # noqa: BLE001
        log.error("run check failed: %s", e)
        busy = True
    try:
        have = check_buffer(busy, actions)
    except Exception as e:                                           # noqa: BLE001
        log.error("buffer check failed: %s", e)
        have = []
    try:
        check_uploads(actions)
    except Exception as e:                                           # noqa: BLE001
        email(f"upload-{now():%Y%m%d}", "Urzeit-Kanal: could not start an upload", str(e)[:500], mem)
    mem["emailed"] = mem["emailed"][-200:]
    STATE.parent.mkdir(exist_ok=True)
    STATE.write_text(json.dumps(mem, indent=1), encoding="utf-8")
    beat = {"date": f"{now():%Y-%m-%d}", "checked_at": f"{now():%Y-%m-%dT%H:%M:%SZ}",
            "ahead": [[r["slug"], r["status"], r.get("publish_at")] for r in have], "actions": actions}
    old = json.loads(HEARTBEAT.read_text(encoding="utf-8")) if HEARTBEAT.exists() else {}
    HEARTBEAT.write_text(json.dumps(beat, indent=1), encoding="utf-8")
    print(json.dumps(beat, indent=1))
    out = os.environ.get("GITHUB_OUTPUT")
    if out:
        with open(out, "a", encoding="utf-8") as f:
            f.write(f"commit={'true' if actions or old.get('date') != beat['date'] else 'false'}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
