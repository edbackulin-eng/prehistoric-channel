"""Daily check that every key the channel runs on still works.

    python scripts/health.py            (from .github/workflows/health.yml)

YouTube (a real token refresh and a channel read), Claude (a one-line answer),
Kaggle (a kernel listing), Gemini and Groq (a model listing), Pixabay (one
search). A key that fails is emailed at once; a key with a known lifetime is
emailed seven days before it runs out:
  - the Claude token from `claude setup-token` lives a year;
  - a YouTube refresh token lives forever unless Google says otherwise, which it
    does by returning refresh_token_expires_in (an OAuth app left in testing).
Issue dates are in state/tokens.json. Each problem is emailed once until the key
works again.
"""
import json
import os
import subprocess
import sys
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
TOKENS = ROOT / "state" / "tokens.json"
MEMORY = ROOT / "state" / "health.json"
REPO = os.environ.get("GITHUB_REPOSITORY", "edbackulin-eng/closed-tabs")
OWNER = REPO.split("/")[0]
WARN_DAYS = 7
GEMINI_KEY = os.environ.get("GEMINI_KEY_NAME", "GEMINI_API_KEY_CT")


def now():
    return datetime.now(timezone.utc)


def _get(url, headers=None, data=None, timeout=40):
    req = urllib.request.Request(url, data=data, headers={"User-Agent": "closed-tabs-health", **(headers or {})})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8", "replace"))


def youtube():
    d = json.loads(os.environ["YT_TOKEN_JSON"])
    body = urllib.parse.urlencode({"client_id": d["client_id"], "client_secret": d["client_secret"],
                                   "refresh_token": d["refresh_token"],
                                   "grant_type": "refresh_token"}).encode()
    tok = _get("https://oauth2.googleapis.com/token", data=body)
    me = _get("https://www.googleapis.com/youtube/v3/channels?part=snippet&mine=true",
              headers={"Authorization": f"Bearer {tok['access_token']}"})
    name = me["items"][0]["snippet"]["title"]
    left = tok.get("refresh_token_expires_in")
    expires = (now() + timedelta(seconds=int(left))) if left else None
    return f"channel {name}", expires


def claude():
    r = subprocess.run(["claude", "-p", "Reply with the single word OK."], capture_output=True,
                       text=True, timeout=180)
    if r.returncode != 0 or "ok" not in r.stdout.lower():
        raise RuntimeError((r.stderr or r.stdout)[-300:])
    return "answers", None


def kaggle():
    r = subprocess.run([sys.executable, "-m", "kaggle", "kernels", "list", "--mine", "--page-size", "1"],
                       capture_output=True, text=True, timeout=120)
    if r.returncode != 0:
        raise RuntimeError(r.stderr[-300:])
    return "kernels listed", None


def gemini():
    key = os.environ[GEMINI_KEY]
    _get("https://generativelanguage.googleapis.com/v1beta/models?pageSize=1",
         headers={"x-goog-api-key": key})
    return "models listed", None


def groq():
    _get("https://api.groq.com/openai/v1/models",
         headers={"Authorization": f"Bearer {os.environ['GROQ_API_KEY']}"})
    return "models listed", None


def pixabay():
    q = urllib.parse.urlencode({"key": os.environ["PIXABAY_API_KEY"].strip(), "q": "city", "per_page": 3})
    _get(f"https://pixabay.com/api/videos/?{q}")
    return "search works", None


CHECKS = {"YouTube": youtube, "Claude": claude, "Kaggle": kaggle, "Gemini": gemini,
          "Groq": groq, "Pixabay": pixabay}


def email(subject, body):
    r = subprocess.run(["gh", "issue", "list", "-R", REPO, "--label", "run-log", "--state", "open",
                        "--json", "number"], capture_output=True, text=True)
    issues = json.loads(r.stdout or "[]")
    if not issues:
        subprocess.run(["gh", "label", "create", "run-log", "-R", REPO, "--color", "b60205"],
                       capture_output=True)
        out = subprocess.run(["gh", "issue", "create", "-R", REPO, "--title", "Daily run failures",
                              "--label", "run-log", "--body",
                              "A comment here mentioning the owner arrives as an email."],
                             capture_output=True, text=True).stdout
        n = out.strip().rsplit("/", 1)[-1]
    else:
        n = str(issues[0]["number"])
    subprocess.run(["gh", "issue", "comment", n, "-R", REPO, "--body",
                    f"@{OWNER} **{subject}**\n\n{body}"], check=True, capture_output=True)


def main():
    only = [a for a in sys.argv[1:] if a in CHECKS]
    tokens = json.loads(TOKENS.read_text(encoding="utf-8")) if TOKENS.exists() else {}
    mem = json.loads(MEMORY.read_text(encoding="utf-8")) if MEMORY.exists() else {"open": []}
    results, problems = {}, []
    for name, fn in CHECKS.items():
        if only and name not in only:
            continue
        try:
            note, expires = fn()
            results[name] = {"ok": True, "note": note}
        except Exception as e:                                   # noqa: BLE001
            results[name] = {"ok": False, "note": str(e)[:300]}
            problems.append((f"{name}-dead", f"{name} key does not work",
                             f"{str(e)[:300]}\n\nThe secret needs renewing in the repository settings."))
            continue
        issued = tokens.get(name, {}).get("issued")
        life = tokens.get(name, {}).get("lifetime_days")
        if issued and life and not expires:
            expires = datetime.fromisoformat(issued.replace("Z", "+00:00")) + timedelta(days=life)
        if expires:
            results[name]["expires"] = expires.strftime("%Y-%m-%d")
            if expires - now() <= timedelta(days=WARN_DAYS):
                problems.append((f"{name}-expiring-{expires:%Y%m%d}", f"{name} key expires {expires:%Y-%m-%d}",
                                 f"It still works today; renew it before {expires:%Y-%m-%d}."))
    print(json.dumps(results, indent=1))
    still_open = []
    for key, subject, body in problems:
        still_open.append(key)
        if key not in mem["open"]:
            email(f"{REPO.split('/')[1]}: {subject}", body)
    mem["open"] = still_open
    mem["last"] = {"at": now().strftime("%Y-%m-%dT%H:%M:%SZ"), "results": results}
    MEMORY.parent.mkdir(exist_ok=True)
    MEMORY.write_text(json.dumps(mem, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
