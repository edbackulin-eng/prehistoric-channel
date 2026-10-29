"""Does today's free quota still hold this step? If not, the episode waits for tomorrow.

    python automation/llm_budget.py need <step> <slug>     # exit 0: go; 3: deferred to the next day
    python automation/llm_budget.py report                # today's use and what is left, per model

The steps and what each costs come from config/channel.json "llm_quota":
text steps (script, hook check, plan) go to Groq and fall back to Gemini
flash-lite; image steps (the FLUX check, the render's species check, masks,
thumbnails) go to Gemini flash, then flash-lite. A step that would run a model
out of its day is not started: state/deferred.json holds the episode until the
providers' day turns (midnight Pacific), and the watchdog picks it up then -
so the script and the image checks fall on different days when one day cannot
hold both.
"""
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from core import groq, llm_usage  # noqa: E402
from core.logger import get_logger  # noqa: E402

log = get_logger("budget")
Q = llm_usage.CFG["llm_quota"]
GROQ = f"groq:{groq.MODEL}"
FLASH, LITE = "gemini-flash-latest", "gemini-flash-lite-latest"
DEFERRED = ROOT / "state" / "deferred.json"


def image_need(slug):
    """Requests the render's checks will make: picture beats not yet judged, masks, thumbnails."""
    ep = ROOT / "episodes" / slug
    shots = json.loads((ep / "shots.json").read_text(encoding="utf-8"))["lines"]
    known = json.loads((ep / "species_verdicts.json").read_text(encoding="utf-8")) \
        if (ep / "species_verdicts.json").exists() else {}
    beats = {b.get("id") or b.get("need") for l in shots for b in l["beats"] if b["kind"] != "anim"}
    recon_known = sum(1 for b in beats if any(k.startswith(f"{b}__s") for k in known))
    r = Q["per_episode"]["render"]
    return int((len(beats) - recon_known) * r["image_per_picture_beat"] + r["image_fixed"])


def text_ok(requests, tokens):
    if groq.available() and llm_usage.enough(GROQ, requests, tokens):
        return True, f"Groq: {llm_usage.left(GROQ)} left"
    return llm_usage.enough(LITE, requests), f"Gemini flash-lite: {llm_usage.left(LITE)} left"


def image_ok(requests):
    room = max(0, llm_usage.left(FLASH)["requests"]) + max(0, llm_usage.left(LITE)["requests"])
    return room >= requests, f"Gemini images: {room} requests left today"


def defer(slug, why):
    now = datetime.now(llm_usage.PT)
    turn = (now + timedelta(days=1)).replace(hour=0, minute=10, second=0, microsecond=0)
    DEFERRED.parent.mkdir(exist_ok=True)
    DEFERRED.write_text(json.dumps({"slug": slug, "why": why, "until": turn.astimezone(timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ")}, indent=1), encoding="utf-8")
    log.warning("%s deferred until %s: %s", slug, turn.isoformat(), why)


def need(step, slug):
    pe = Q["per_episode"]
    if step in ("script", "hook_check", "plan"):
        ok, why = text_ok(pe[step]["text_requests"], pe[step]["text_tokens"])
    elif step == "flux_check":
        ok, why = image_ok(pe["flux_check"]["image_requests"])
    elif step == "render":
        n = image_need(slug)
        ok, why = image_ok(n)
        why = f"needs ~{n}; {why}"
    else:
        raise SystemExit(f"unknown step {step}")
    log.info("%s for %s: %s (%s)", step, slug, "go" if ok else "not today", why)
    if not ok:
        defer(slug, f"{step}: {why}")
        return 3
    if DEFERRED.exists() and json.loads(DEFERRED.read_text(encoding="utf-8")).get("slug") == slug:
        DEFERRED.unlink()
    return 0


def deferred_now():
    """The deferral in force, or None."""
    if not DEFERRED.exists():
        return None
    d = json.loads(DEFERRED.read_text(encoding="utf-8"))
    until = datetime.fromisoformat(d["until"].replace("Z", "+00:00"))
    return d if until > datetime.now(timezone.utc) else None


def report():
    out = {"day_pt": llm_usage.day()}
    for m, lim in Q["models"].items():
        out[m] = {"used": llm_usage.used(m), "limit": lim, "left_after_reserve": llm_usage.left(m)}
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    if sys.argv[1] == "need":
        sys.exit(need(sys.argv[2], sys.argv[3]))
    sys.exit(report())
