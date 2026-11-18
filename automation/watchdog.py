"""The watchdog: keeps the German channel running with nobody watching.

    python automation/watchdog.py            (from .github/workflows/watchdog.yml)

Acts only through GitHub Actions and the YouTube API. Each pass:
  1. an episode or render run that hangs far longer than any build is cancelled;
  2. a failed or cancelled run has its failed jobs rerun (the jobs that passed
     keep their artifacts) - the same failing step at most DAILY_RERUNS times a
     UTC day, whatever run it is in; then the episode is deferred to the next
     day (state/deferred.json) and the owner gets ONE email with the step and
     its log - a comment on the open "run-log" issue that mentions them
     (01.11-02.11: one step failed 21 runs in a row, a letter on every one);
  3. if fewer than `buffer` (config/channel.json) episodes are ahead - built
     and waiting, or scheduled - and nothing is being built, the next episode
     is started (episode.yml; the GPU rules are kept inside it);
  4. an episode waiting in the buffer is uploaded as soon as YT_TOKEN_JSON
     exists (upload.yml), on the nearest free slot;
  5. GPU work handed over until the free Kaggle window opens (state/gpu_pending.json,
     automation/gpu_pending.py) is started when due, just before the window;
  6. once a day the state is committed (state/heartbeat.json), so GitHub never
     switches the schedules off for a quiet repository.
Every email is recorded in state/watchdog.json so one failure is reported once.

Owner 12.11 - no pause that waits for him:
  - a topic whose build gives up on two days is skipped (topics/queue.json "skipped") and the next
    one is built; one letter says so;
  - an episode whose QA found a real fault is rebuilt (render.yml from its GPU results, the faulty
    sources left out: episodes/<slug>/qa_rejects.json) up to REBUILDS times, then skipped the same way;
  - only when the same step gives up for two different topics is the CODE taken to be broken: the
    builds stop (state/deferred.json with the code's fingerprint) until the code changes, one letter.
"""
import json
import re
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
DAILY_RERUNS = 1          # owner 17.11: one rerun of a failing step a UTC day; then it waits for the next day
NO_RERUN = ("format is being rebuilt", "format_gate")      # a deliberate stop is not a failure to repeat
QUOTA_MARKS = ("429", "RESOURCE_EXHAUSTED", "quota", "Too Many Requests")
TOPIC_GIVE_UPS = 2        # owner 12.11: days a topic may give up before the next one is built
REBUILDS = 2              # renders again after a failed QA before the topic is skipped
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
    print(f"LETTER (mail is off (owner 18.11): the text stays in the Actions log, nobody is written to): {subject}\n{body}")
    return
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


def failed_log(run_id):
    try:
        return gh("run", "view", str(run_id), "-R", REPO, "--log-failed", parse=False)
    except Exception as e:                                           # noqa: BLE001
        return f"(log unavailable: {e})"


def failed_tail(run_id, log_text=None):
    """The failed step's own last 40 lines: its output, not the cleanup steps after it."""
    lines = (log_text if log_text is not None else failed_log(run_id)).strip().splitlines()
    end = next((i for i, l in enumerate(lines) if "##[error]" in l), len(lines) - 1)
    return "\n".join(l.split("Z ", 1)[-1] for l in lines[max(0, end - 39):end + 1])


def failed_step(run_id):
    """'job / step' of the run's first failed step ('' if none is named)."""
    try:
        jobs = gh("api", f"repos/{REPO}/actions/runs/{run_id}/jobs")["jobs"]
    except Exception:                                            # noqa: BLE001
        return ""
    for j in jobs:
        if j.get("conclusion") == "failure":
            st = next((s["name"] for s in j.get("steps", []) if s.get("conclusion") == "failure"), "?")
            return f"{j['name']} / {st}"
    return ""


def run_slug(run_id, log_text):
    """The episode a run worked on: the SLUG every step's env shows in the log."""
    import re
    m = re.search(r"\bSLUG: ([a-z0-9-]+)", log_text or "")
    return m.group(1) if m else None


def give_up_today(wf, rid, sig, mem, actions):
    """The step has had its reruns today: the episode waits for the next day, one email."""
    from automation import llm_budget
    text = failed_log(rid)
    tail = failed_tail(rid, text)
    slug = run_slug(rid, text) or "?"
    why = f"{wf}: '{sig}' failed {DAILY_RERUNS + 1} times today"
    gu = mem.setdefault("give_ups", [])
    gu.append({"slug": slug, "sig": sig, "day": f"{now():%Y-%m-%d}"})
    mem["give_ups"] = gu[-100:]
    others = {g["slug"] for g in gu if g["sig"] == sig and g["slug"] not in (slug, "?")}
    if others:
        # the same step stops two topics: the code, not the topic - builds wait for a code change
        llm_budget.defer(slug, f"code broken: '{sig}' fails for {slug} and {sorted(others)}", code_fp=code_fingerprint())
        actions.append(f"builds paused until the code changes: '{sig}' fails for {slug} and {sorted(others)}")
        email(f"code-{sig}-{code_fingerprint()}", f"Urzeit-Kanal: the code is broken - {sig}",
              f"The step fails for two topics ({slug}, {', '.join(sorted(others))}), so the fault is in the code, "
              f"not in a topic. Builds wait until the code changes (any commit to pipeline/, automation/, core/, "
              f"gpu/ or the workflows lifts the pause).\n\nLast run: https://github.com/{REPO}/actions/runs/{rid}"
              f"\n\n```\n{tail}\n```", mem)
        return
    days = {g["day"] for g in gu if g["slug"] == slug}
    if slug != "?" and len(days) >= TOPIC_GIVE_UPS:
        skip_topic(slug, f"{why} (on {len(days)} days)", mem, actions, rid, tail)
        return
    if not llm_budget.deferred_now():
        llm_budget.defer(slug, why)
        actions.append(f"{slug} deferred to the next day: {why}")
    email(f"stop-{wf}-{sig}-{now():%Y%m%d}", f"Urzeit-Kanal: {slug} tries again tomorrow - {sig}",
          f"The same step failed {DAILY_RERUNS + 1} times today ({DAILY_RERUNS} reruns); the build goes on by "
          f"itself tomorrow, and if it fails then too, the next topic is built instead."
          f"\n\nLast run: https://github.com/{REPO}/actions/runs/{rid}\n\n```\n{tail}\n```", mem)


def code_fingerprint():
    """The code as git has it: pipeline/, automation/, core/, gpu/, the workflows."""
    import hashlib
    try:
        out = subprocess.run(["git", "ls-files", "-s", "pipeline", "automation", "core", "gpu", ".github/workflows"],
                             capture_output=True, text=True, cwd=ROOT).stdout
    except OSError:
        return "?"
    return hashlib.sha1(out.encode()).hexdigest()[:16]


def skip_topic(slug, why, mem, actions, rid=None, tail=""):
    """Owner 12.11: a topic that cannot be built is left; the next one is built. One letter."""
    from automation import llm_budget, release, topics
    topics.done(slug, "skipped")
    r = release.rows()
    row = next((x for x in r if x["slug"] == slug), None)
    if row is not None:
        row.update({"status": "skipped", "skipped_why": why, "skipped_at": f"{now():%Y-%m-%dT%H:%M:%SZ}"})
        release.save(r)
    d = llm_budget.deferred_now()
    if d and d.get("slug") == slug:
        llm_budget.DEFERRED.unlink(missing_ok=True)
    actions.append(f"{slug} skipped, the next topic is built: {why}")
    email(f"skip-{slug}", f"Urzeit-Kanal: {slug} skipped - the next topic is being built",
          f"{why}\n\n" + (f"Last run: https://github.com/{REPO}/actions/runs/{rid}\n\n```\n{tail}\n```" if rid
                           else ""), mem)


def pacific_day(t):
    """The quota day (Gemini and Groq count per Pacific day)."""
    from zoneinfo import ZoneInfo
    return t.astimezone(ZoneInfo("America/Los_Angeles")).date()


def check_waived(actions):
    """15.11: a film the owner let out by exception (config release.qa_waived) is not rebuilt, not
    skipped and not checked again: it is built, and check_uploads sends it up."""
    from automation import release
    r = release.rows()
    for row in r:
        if row.get("status") in ("qa-failed", "qa-waiting", "requalifying") and release.waived(row["slug"])                 and not (ROOT / "episodes" / row["slug"] / "uploaded.json").exists() and row.get("run_id"):
            row.update({"status": "built"})
            release.save(r)
            actions.append(f"{row['slug']}: owner's exception, goes up as built")
            return


def check_requalify(mem, actions, busy):
    """14.11: a film whose QA ran out of Gemini quota (release.quota_only) is checked again on the
    next quota day from its finished render - requalify.yml, which releases it when it passes."""
    from automation import release
    if busy:
        return
    r = release.rows()
    for row in r:
        if row.get("status") == "qa-failed" and release.quota_only(row.get("qa_failed")):
            # a render on older code wrote qa-failed for it
            row.update({"status": "qa-waiting", "qa_waiting_since": row.get("built_at") or f"{now():%Y-%m-%dT%H:%M:%SZ}"})
            release.save(r)
            actions.append(f"{row['slug']}: QA ran out of Gemini quota, not a fault; checked again on the next quota day")
        if row.get("status") == "requalifying" and (now() - ts(row["requalify_at"])).total_seconds() > 4 * 3600:
            # the requalify run ended without a record (it broke): the next quota day tries again
            row.update({"status": "qa-waiting", "qa_waiting_since": f"{now():%Y-%m-%dT%H:%M:%SZ}"})
            release.save(r)
            actions.append(f"{row['slug']}: requalify left no record; tried again on the next quota day")
        if row.get("status") != "qa-waiting" or not row.get("run_id"):
            continue
        since = ts(row.get("qa_waiting_since") or row.get("built_at"))
        if pacific_day(now()) <= pacific_day(since):
            continue
        gh("workflow", "run", "requalify.yml", "-R", REPO, "-f", f"slug={row['slug']}", "-f",
           f"run_id={row['run_id']}", "-f", "thumbs=false", parse=False)
        row.update({"status": "requalifying", "requalify_at": f"{now():%Y-%m-%dT%H:%M:%SZ}"})
        release.save(r)
        actions.append(f"{row['slug']}: QA again on the new quota day (requalify of render {row['run_id']})")
        return


def check_rebuilds(mem, actions, busy):
    """Owner 12.11: an episode whose QA found a real fault is rendered again from its GPU results,
    the faulty sources left out (release.record wrote them to episodes/<slug>/qa_rejects.json);
    after REBUILDS such renders the topic is skipped."""
    from automation import release
    if busy:
        return
    r = release.rows()
    for row in r:
        if row.get("status") != "qa-failed" or release.waived(row["slug"]):
            continue
        n = int(row.get("rebuilds", 0))
        if n >= REBUILDS:
            skip_topic(row["slug"], f"QA found a real fault in {n + 1} renders: {row.get('qa_failed')}", mem, actions)
            return
        if (ROOT / "episodes" / row["slug"] / "pool.json").exists() and not row.get("gpu_run"):
            # owner 15.11, real footage: the plan cut again from the pool without the rejected pictures;
            # the narration of the last render is kept when its artifact is still there
            prev = row.get("layers_run") or ""
            try:
                arts = gh("api", f"repos/{REPO}/actions/runs/{prev}/artifacts?per_page=100")["artifacts"] if prev else []
                if not any(a["name"] == "narration" and not a["expired"] for a in arts):
                    prev = ""
            except Exception:                                    # noqa: BLE001
                prev = ""
            gh("workflow", "run", "render.yml", "-R", REPO, "-f", f"slug={row['slug']}", "-f", "real=true",
               "-f", f"layers_run={prev}", "-f", "release=true", parse=False)
            row.update({"status": "rebuilding", "rebuilds": n + 1, "rebuild_at": f"{now():%Y-%m-%dT%H:%M:%SZ}"})
            release.save(r)
            actions.append(f"{row['slug']}: QA fault, cut again from its real pictures without the rejected ones "
                           f"({n + 1}/{REBUILDS})")
            return
        if not row.get("gpu_run"):
            continue
        layers = row.get("layers_run") or ""
        if layers:
            try:
                arts = gh("api", f"repos/{REPO}/actions/runs/{layers}/artifacts?per_page=100")["artifacts"]
                if not any(a["name"].startswith("layers-") and not a["expired"] for a in arts):
                    layers = ""
            except Exception:                                    # noqa: BLE001
                layers = ""
        gh("workflow", "run", "render.yml", "-R", REPO, "-f", f"slug={row['slug']}", "-f", f"gpu_run={row['gpu_run']}",
           "-f", f"layers_run={layers}", "-f", "release=true", parse=False)
        row.update({"status": "rebuilding", "rebuilds": n + 1, "rebuild_at": f"{now():%Y-%m-%dT%H:%M:%SZ}"})
        release.save(r)
        actions.append(f"{row['slug']}: QA fault, rendered again without its faulty sources ({n + 1}/{REBUILDS})")
        return


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
        # one step, whichever matrix shard failed: "layers (3) / ... shard 3 of 12" and "layers (9)"
        # are the same step (04.11: counted apart, the render was rerun ten times on one bug)
        sig = re.sub(r" \(\d+\)|,? shard \d+ of \d+", "", failed_step(rid) or latest["conclusion"])
        day = f"{now():%Y-%m-%d}"
        reruns = mem.setdefault("reruns", [])
        if f"{rid}/{attempt}" in {r["run"] for r in reruns}:
            continue                       # this failure has had its answer (a rerun or the stop)
        if any(m.lower() in sig.lower() for m in NO_RERUN):
            reruns.append({"run": f"{rid}/{attempt}", "wf": wf, "sig": sig, "day": day, "rerun": False})
            mem["reruns"] = reruns[-300:]
            continue                       # owner 17.11: a deliberate stop waits, nothing is rerun
        logf = subprocess.run(["gh", "run", "view", str(rid), "-R", REPO, "--log-failed"],
                              capture_output=True, text=True).stdout[-20000:]
        if any(m.lower() in logf.lower() for m in QUOTA_MARKS):
            reruns.append({"run": f"{rid}/{attempt}", "wf": wf, "sig": sig, "day": day, "rerun": False})
            mem["reruns"] = reruns[-300:]
            actions.append(f"{wf} {rid} stopped on a spent quota; it waits for the quota to turn (no rerun)")
            continue
        done_today = sum(1 for r in reruns if r["day"] == day and r["wf"] == wf and r["sig"] == sig
                         and r.get("rerun"))
        if done_today < DAILY_RERUNS:
            try:
                gh("run", "rerun", str(rid), "-R", REPO, "--failed", parse=False)
            except RuntimeError:
                gh("run", "rerun", str(rid), "-R", REPO, parse=False)
            reruns.append({"run": f"{rid}/{attempt}", "wf": wf, "sig": sig, "day": day, "rerun": True})
            actions.append(f"reran {wf} {rid} (attempt {attempt + 1}; '{sig}' {done_today + 1}/{DAILY_RERUNS} today)")
            busy = busy or wf != "upload.yml"
        else:
            reruns.append({"run": f"{rid}/{attempt}", "wf": wf, "sig": sig, "day": day, "rerun": False})
            give_up_today(wf, rid, sig, mem, actions)
        mem["reruns"] = reruns[-300:]
    return busy


def check_gpu_pending(actions):
    """GPU work handed over until the free window opens (automation/gpu_pending.py): started
    when due. True while a build waits there, so no second build is started meanwhile."""
    from automation import gpu_pending
    for r in gpu_pending.take_due(now()):
        gh("workflow", "run", r["workflow"], "-R", REPO, *sum((["-f", f"{k}={v}"] for k, v in r["inputs"].items()), []),
           parse=False)
        actions.append(f"started {r['workflow']} {r['inputs'].get('slug', '')} for the GPU window at {r['opens']}")
    return bool(gpu_pending.waiting_for("episode.yml"))


def check_buffer(busy, actions):
    from automation import release
    from automation import llm_budget
    have = release.ahead()
    from automation import format_gate
    if not format_gate.ready():
        actions.append("the new format is being built (owner 16.11): no episode started")
        return have
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
        busy = check_gpu_pending(actions) or busy
    except Exception as e:                                           # noqa: BLE001
        log.error("pending GPU starts failed: %s", e)
    try:
        check_waived(actions)
    except Exception as e:                                           # noqa: BLE001
        log.error("waiver check failed: %s", e)
    try:
        check_requalify(mem, actions, busy)
    except Exception as e:                                           # noqa: BLE001
        log.error("requalify check failed: %s", e)
    try:
        check_rebuilds(mem, actions, busy)
    except Exception as e:                                           # noqa: BLE001
        log.error("rebuild check failed: %s", e)
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
