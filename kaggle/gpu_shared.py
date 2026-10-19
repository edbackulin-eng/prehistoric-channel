"""One GPU, three channels: the shared view and lock for the Kaggle account.

The same file lives in kaggle/ of all three repositories (what-if, closed-tabs,
prehistoric-channel); change it in all three, each through its smoke test.

What it gives every channel before it submits a kernel:
  busy_kernels()   kernels of ANY channel that are queued or running. Only
                   kernels run in the last 12 h are asked about (a status call
                   is ~6 s each), every call has a time-out, and an answer that
                   is not a status counts as busy - a failed call must never
                   look like a free GPU.
  wait_for_free()  block until nothing is busy for `idle_s` without a break
                   (Closed Tabs submits in series ~40 s apart, so one free
                   moment proves nothing).
  after_push()     the race two channels can still lose - both see a free GPU
                   and push within seconds of each other (it happened on
                   18.10, 15 s apart). Priority decides: What If > Closed Tabs
                   > prehistoric, and between equals the later push yields. The
                   yielding channel deletes its own queued kernel and waits
                   again; What If never yields.
  week_hours()     GPU hours of all three channels over the last 7 days:
                   Closed Tabs and prehistoric from their published ledgers,
                   What If measured from its kernels' logs (see closed-tabs
                   kaggle/gpu_budget.py for why it is measured that way).
"""
import json
import os
import subprocess
import sys
import time
import urllib.request
from datetime import datetime, timedelta, timezone

CHANNELS = {"whatif": 0, "closedtabs": 1, "prehistoric": 2}     # lower = higher priority
WEEKLY_LIMIT_H = 25.0
LEDGERS = {
    "closedtabs": "https://raw.githubusercontent.com/edbackulin-eng/closed-tabs/main/state/gpu_ledger.json",
    "prehistoric": "https://raw.githubusercontent.com/edbackulin-eng/prehistoric-channel/main/state/gpu_ledger.json",
}
RECENT_H = 12
CALL_TIMEOUT_S = 120
RACE_WINDOW_S = 300
BUSY_WORDS = ("running", "queued", "pending")


def channel_of(ref):
    slug = ref.split("/")[-1]
    return next((c for c in CHANNELS if slug.startswith(c)), None)


def _kaggle(*args):
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"}
    try:
        r = subprocess.run([sys.executable, "-m", "kaggle", *args], capture_output=True, text=True,
                           encoding="utf-8", errors="replace", env=env, timeout=CALL_TIMEOUT_S)
        return r.stdout + r.stderr
    except subprocess.TimeoutExpired:
        return ""


def kernels():
    """[(ref, lastRunTime)] of every channel's kernels, newest first; None if the call failed."""
    out = _kaggle("kernels", "list", "--mine", "--page-size", "60", "--sort-by", "dateRun", "--csv")
    lines = out.splitlines()
    if not lines or not lines[0].startswith("ref"):
        return None
    header = [h.strip() for h in lines[0].split(",")]
    i_t = header.index("lastRunTime")
    rows = []
    for line in lines[1:]:
        cols = line.split(",")
        if len(cols) <= i_t or not channel_of(cols[0]):
            continue
        try:
            t = datetime.fromisoformat(cols[i_t].strip().replace("Z", "+00:00"))
        except ValueError:
            continue
        rows.append((cols[0].strip(), t if t.tzinfo else t.replace(tzinfo=timezone.utc)))
    return rows


def status(ref):
    s = _kaggle("kernels", "status", ref).lower()
    if "kernelworkerstatus" not in s:
        return "unknown"
    return s.rsplit("kernelworkerstatus.", 1)[1].split('"')[0].strip()


def busy_kernels():
    """[(ref, lastRunTime, status)] queued/running/unknown, any channel."""
    rows = kernels()
    if rows is None:
        return [("kernel list unavailable", datetime.now(timezone.utc), "unknown")]
    recent = datetime.now(timezone.utc) - timedelta(hours=RECENT_H)
    busy = []
    for ref, t in rows:
        if t < recent:
            continue
        st = status(ref)
        if st in BUSY_WORDS or st == "unknown":
            busy.append((ref, t, st))
    return busy


def wait_for_free(idle_s=15 * 60, poll_s=60, max_wait_s=8 * 3600, log=print):
    """True once nothing has been busy for idle_s; False if max_wait_s ran out."""
    started, idle_since = time.time(), None
    while time.time() - started < max_wait_s:
        busy = busy_kernels()
        now = time.time()
        if busy:
            idle_since = None
            log(f"  GPU busy: {', '.join(f'{r} ({s})' for r, _, s in busy)}")
        else:
            idle_since = idle_since or now
            if now - idle_since >= idle_s:
                return True
        time.sleep(poll_s)
    return False


def after_push(ref, log=print):
    """Resolve a race right after `ref` was pushed. True: keep it. False: it
    yielded - it has been deleted and the caller should wait and push again."""
    me = channel_of(ref)
    time.sleep(20)                                   # let the other push land in the listing
    rows = kernels() or []
    mine = next((t for r, t in rows if r == ref), datetime.now(timezone.utc))
    for r, t, st in busy_kernels():
        if r == ref or r == "kernel list unavailable":
            continue
        other = channel_of(r)
        if other is None or abs((t - mine).total_seconds()) > RACE_WINDOW_S and st != "running":
            continue
        outranks = CHANNELS[other] < CHANNELS[me] or (CHANNELS[other] == CHANNELS[me] and t < mine)
        if outranks and CHANNELS[me] > 0:
            log(f"  {r} ({st}) goes first; withdrawing {ref}")
            _kaggle("kernels", "delete", ref, "-y")
            return False
    return True


# -------------------------------------------------------------- the week
def _ledger_hours(url, since):
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "gpu-shared"}),
                                    timeout=30) as r:
            rows = json.loads(r.read().decode())
    except Exception:                                            # noqa: BLE001
        return None
    return sum(x["hours"] for x in rows if datetime.fromisoformat(x["start"]) >= since)


def week_hours(whatif_hours_fn, own=None, now=None):
    """{channel: hours} for the last 7 days. `whatif_hours_fn` is the caller's
    measured What If figure (closed-tabs gpu_budget.whatif_hours); `own`
    overrides a channel's figure with its local, freshest ledger value."""
    now = now or datetime.now(timezone.utc)
    since = now - timedelta(days=7)
    out = {c: _ledger_hours(u, since) for c, u in LEDGERS.items()}
    out["whatif"] = whatif_hours_fn(now)
    out.update(own or {})
    missing = [c for c, h in out.items() if h is None]
    if missing:
        raise RuntimeError(f"GPU hours unknown for {missing}")
    return out
