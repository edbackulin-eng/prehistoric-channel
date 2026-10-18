"""The hourly watchdog: keeps the channel running with nobody watching.

    python automation/watchdog.py            (from .github/workflows/watchdog.yml)

Runs on GitHub Actions only, and acts only through GitHub Actions and the
YouTube API. Each pass:

  1. a daily run that hangs (in progress far longer than any build takes) is
     cancelled, so the next step can recover it;
  2. a daily run that failed or was cancelled has its failed jobs rerun - the
     artifacts of the jobs that passed are kept, so only the broken part runs
     again. After the third failed attempt resume-episode takes over, and if
     that fails too the owner gets an email with the reason;
  3. if fewer than MIN_AHEAD videos are scheduled and no build is running, a
     build is started (outside the GPU window too: push.py still waits for any
     kernel of either channel, so What If is never disturbed);
  4. if the next publication slot is less than RESCUE_H hours away and holds no
     video, the earliest video scheduled later is moved onto it, so no slot is
     ever empty; the buffer refills behind it;
  5. once a day the state is committed (state/heartbeat.json), so the
     repository never goes 60 days without a commit and loses its schedules.

Everything it decides goes to the job log; every email it sends is recorded in
state/watchdog.json so the same failure is reported once.
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

REPO = os.environ.get("GITHUB_REPOSITORY", "edbackulin-eng/closed-tabs")
OWNER = REPO.split("/")[0]
BUILD = "daily.yml"
RESUME = "resume-episode.yml"
MIN_AHEAD = 2
RESCUE_H = 3
HUNG_H = 9            # longer than voice + stills + render + merge + upload put together
SETTLE_MIN = 12       # leave a just-finished run alone for a moment
MAX_ATTEMPTS = 3
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


def failed_reason(run_id):
    """The failed jobs of a run and the tail of their logs, for an email."""
    jobs = gh("api", f"repos/{REPO}/actions/runs/{run_id}/jobs?per_page=50")["jobs"]
    bad = [j for j in jobs if j.get("conclusion") in ("failure", "cancelled", "timed_out")]
    names = ", ".join(f"{j['name']} ({j['conclusion']})" for j in bad) or "unknown"
    try:
        tail = gh("run", "view", str(run_id), "-R", REPO, "--log-failed", parse=False)
        tail = "\n".join(l.split("Z ", 1)[-1] for l in tail.strip().splitlines()[-25:])
    except Exception as e:                                       # noqa: BLE001
        tail = f"(log unavailable: {e})"
    return names, tail


def memory():
    return json.loads(STATE.read_text(encoding="utf-8")) if STATE.exists() else {"emailed": []}


def email(key, subject, body, mem):
    """An issue comment mentioning the owner: GitHub sends it as an email."""
    if key in mem["emailed"]:
        return
    issues = gh("issue", "list", "-R", REPO, "--label", "run-log", "--state", "open",
                "--json", "number")
    if issues:
        n = issues[0]["number"]
    else:
        subprocess.run(["gh", "label", "create", "run-log", "-R", REPO, "--color", "b60205"],
                       capture_output=True)
        url = gh("issue", "create", "-R", REPO, "--title", "Daily run failures", "--label", "run-log",
                 "--body", "A comment here mentioning the owner arrives as an email.", parse=False)
        n = url.strip().rsplit("/", 1)[-1]
    gh("issue", "comment", str(n), "-R", REPO, "--body",
       f"@{OWNER} **{subject}**\n\n{body}", parse=False)
    mem["emailed"].append(key)
    log.warning("emailed: %s", subject)


# --- the checks ---------------------------------------------------------------

def check_runs(mem, actions):
    """Steps 1 and 2. Returns True when a build is running or being recovered."""
    busy = False
    daily = runs(BUILD)
    resumes = runs(RESUME)
    for r in daily:
        if r["status"] in ("in_progress", "queued", "waiting", "pending", "requested"):
            age = now() - ts(r["run_started_at"] or r["created_at"])
            if age > timedelta(hours=HUNG_H):
                gh("run", "cancel", str(r["id"]), "-R", REPO, parse=False)
                actions.append(f"cancelled hung run {r['id']} ({age})")
            else:
                busy = True
    for r in resumes:
        if r["status"] in ("in_progress", "queued", "waiting", "pending"):
            busy = True
    if busy:
        return True
    latest = next((r for r in daily if r["status"] == "completed"), None)
    if not latest or latest["conclusion"] not in ("failure", "cancelled", "timed_out"):
        return False
    if now() - ts(latest["updated_at"]) < timedelta(minutes=SETTLE_MIN):
        return True
    rid, attempt = latest["id"], latest.get("run_attempt", 1)
    # a run whose pick found nothing to build is not a failure worth chasing
    if attempt < MAX_ATTEMPTS:
        try:
            gh("run", "rerun", str(rid), "-R", REPO, "--failed", parse=False)
            actions.append(f"reran the failed jobs of {rid} (attempt {attempt + 1})")
        except RuntimeError as e:
            # GitHub can refuse a failed-jobs rerun (nothing it counts as failed);
            # the whole run then goes again - the stills job reads the kernels
            # it already ran instead of drawing them twice
            log.warning("failed-jobs rerun refused (%s); rerunning the whole run", e)
            gh("run", "rerun", str(rid), "-R", REPO, parse=False)
            actions.append(f"reran all of {rid} (attempt {attempt + 1})")
        return True
    mine = [x for x in resumes if f"resume {rid}" in (x.get("display_title") or "")]
    if not mine:
        names = [a["name"] for a in gh("api", f"repos/{REPO}/actions/runs/{rid}/artifacts")["artifacts"]]
        if "voice" in names:
            gh("workflow", "run", RESUME, "-R", REPO, "-f", f"run_id={rid}", "-f", "upload=true",
               parse=False)
            actions.append(f"run {rid} failed {attempt} times; resume-episode started")
            return True
    if not mine or mine[0]["conclusion"] != "success":
        jobs, tail = failed_reason(rid)
        email(f"run-{rid}", f"Closed Tabs: build {rid} failed {attempt} times and could not be recovered",
              f"Failed: {jobs}\n\nRun: https://github.com/{REPO}/actions/runs/{rid}\n\n```\n{tail}\n```", mem)
    return False


def scheduled():
    from automation.daily import _state, ahead
    return sorted(ahead(_state()), key=lambda r: r["publish_at"])


def check_buffer(busy, actions):
    """Step 3."""
    have = scheduled()
    if len(have) >= MIN_AHEAD or busy:
        return have
    h = now().hour
    if 3 <= h < 15:
        # What If builds in the morning; a forced Closed Tabs kernel then would
        # starve it. The buffer waits for 15:00 (or the 20:05 run).
        actions.append(f"only {len(have)} scheduled ahead; waiting for 15:00 UTC to build")
        return have
    force = "false" if (h >= 20 or h < 2) else "true"
    gh("workflow", "run", BUILD, "-R", REPO, "-f", "upload=true", "-f", f"force_window={force}",
       parse=False)
    actions.append(f"only {len(have)} scheduled ahead; build started (force_window={force})")
    return have


def check_slot(have, mem, actions):
    """Step 4."""
    from pipeline import youtube
    from automation.daily import _state, _save_state
    slot = youtube.next_slot([], now=now() - youtube.MIN_LEAD)     # the very next slot
    if slot - now() > timedelta(hours=RESCUE_H):
        return
    if any(ts(r["publish_at"]).date() == slot.date() for r in have):
        return
    later = [r for r in have if ts(r["publish_at"]) > slot]
    if not later:
        email(f"empty-{slot:%Y%m%d}", f"Closed Tabs: no video for the {slot:%a %d.%m %H:%M} UTC slot",
              "Nothing is scheduled for it and nothing later could be moved onto it. "
              "A build has been started if none was running.", mem)
        return
    pick = later[0]
    youtube.reschedule(pick["youtube_id"], slot)
    rows = _state()
    for r in rows:
        if r["youtube_id"] == pick["youtube_id"]:
            r["publish_at"] = slot.strftime("%Y-%m-%dT%H:%M:%SZ")
            r.setdefault("moved", []).append(pick["publish_at"])
    _save_state(rows)
    actions.append(f"{pick['youtube_id']} moved from {pick['publish_at']} to the empty slot {slot:%Y-%m-%d %H:%M}")


def main():
    mem = memory()
    actions = []
    if "--test-email" in sys.argv:
        email(f"test-{now():%Y%m%d%H%M}", "Closed Tabs: test email from the watchdog",
              "If this arrived by email, the failure notices will arrive too.", mem)
        STATE.write_text(json.dumps(mem, indent=1), encoding="utf-8")
        return 0
    try:
        busy = check_runs(mem, actions)
    except Exception as e:                                       # noqa: BLE001
        log.error("run check failed: %s", e)
        busy = True                       # do not start builds blind
    have = check_buffer(busy, actions)
    try:
        check_slot(have, mem, actions)
    except Exception as e:                                       # noqa: BLE001
        log.error("slot check failed: %s", e)
        email(f"slotcheck-{now():%Y%m%d}", "Closed Tabs: the watchdog could not check the next slot",
              str(e)[:500], mem)
    have = scheduled()
    mem["emailed"] = mem["emailed"][-200:]
    STATE.parent.mkdir(exist_ok=True)
    STATE.write_text(json.dumps(mem, indent=1), encoding="utf-8")
    beat = {"date": f"{now():%Y-%m-%d}", "checked_at": f"{now():%Y-%m-%dT%H:%M:%SZ}",
            "scheduled_ahead": [[r["publish_at"], r["youtube_id"]] for r in have],
            "actions": actions}
    old = json.loads(HEARTBEAT.read_text(encoding="utf-8")) if HEARTBEAT.exists() else {}
    HEARTBEAT.write_text(json.dumps(beat, indent=1), encoding="utf-8")
    print(json.dumps(beat, indent=1))
    gh_out = os.environ.get("GITHUB_OUTPUT")
    if gh_out:
        # commit once a day, or whenever something was done
        with open(gh_out, "a", encoding="utf-8") as f:
            f.write(f"commit={'true' if actions or old.get('date') != beat['date'] else 'false'}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
