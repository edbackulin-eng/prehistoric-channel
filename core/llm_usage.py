"""How much of each model's free day this channel has used, and how much is left.

Every request to Gemini or Groq is counted here (core/gemini.py, core/groq.py):
one small file per process under state/llm_usage/<day>/, the day as the
providers count it - midnight Pacific time for both - so processes running at
the same time never write the same file. The workflows commit the folder.

The daily allowances are in config/channel.json "llm_quota"; Google publishes
no free-tier table, so they are the values its own 429 answers state (logged
as quotaValue whenever one arrives) or, until one has, a cautious estimate.
"""
import atexit
import json
import os
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parent.parent
DIR = ROOT / "state" / "llm_usage"
PT = ZoneInfo("America/Los_Angeles")
CFG = json.loads((ROOT / "config" / "channel.json").read_text(encoding="utf-8"))
_RUN = {}


def day():
    return datetime.now(PT).strftime("%Y-%m-%d")


def _file():
    run = os.environ.get("GITHUB_RUN_ID", "local")
    job = os.environ.get("GITHUB_JOB", "job")
    return DIR / day() / f"{run}-{job}-{os.getpid()}.json"


def record(model, inp=0, out=0, step=None):
    r = _RUN.setdefault(day(), {}).setdefault(model, {"requests": 0, "in_tokens": 0, "out_tokens": 0, "steps": {}})
    r["requests"] += 1
    r["in_tokens"] += int(inp or 0)
    r["out_tokens"] += int(out or 0)
    if step:
        r["steps"][step] = r["steps"].get(step, 0) + 1
    _flush()


def _flush():
    d = _RUN.get(day())
    if not d:
        return
    f = _file()
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps(d, indent=1), encoding="utf-8")


atexit.register(_flush)


def used(model, on=None):
    """{"requests", "in_tokens", "out_tokens"} of a model on a day, all processes together."""
    tot = {"requests": 0, "in_tokens": 0, "out_tokens": 0}
    for f in (DIR / (on or day())).glob("*.json"):
        try:
            r = json.loads(f.read_text(encoding="utf-8")).get(model)
        except ValueError:
            continue
        if r:
            for k in tot:
                tot[k] += r.get(k, 0)
    return tot


def left(model):
    """Requests (and tokens, where a day's tokens are capped) still free today,
    after the reserve config keeps back."""
    q = CFG["llm_quota"]
    lim = q["models"].get(model, {})
    u = used(model)
    keep = 1 - q["reserve_share"]
    out = {"requests": int(lim.get("rpd", 0) * keep) - u["requests"]}
    if lim.get("tpd"):
        out["tokens"] = int(lim["tpd"] * keep) - u["in_tokens"] - u["out_tokens"]
    return out


def enough(model, requests, tokens=0):
    l = left(model)
    return l["requests"] >= requests and l.get("tokens", tokens) >= tokens
