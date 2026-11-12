"""Does GEMINI_API_KEY_PH answer, and what does Google say about its quota? (12.11: the key was
moved to a project of its own; the check stopped after 5 requests on 429.)

    python scripts/gemini_quota.py [n]     # n tiny requests a model (default 3)

Prints per model the HTTP status of each request and, on a 429, the quota Google names
(quotaId, quotaValue, retry delay). The key itself is never printed.
"""
import json
import os
import sys
import time
import urllib.error
import urllib.request

MODELS = ("gemini-flash-lite-latest", "gemini-flash-latest")
URL = "https://generativelanguage.googleapis.com/v1beta/models/{m}:generateContent"


def one(key, model):
    body = json.dumps({"contents": [{"parts": [{"text": "Reply with OK."}]}],
                       "generationConfig": {"maxOutputTokens": 5}}).encode()
    req = urllib.request.Request(URL.format(m=model), data=body,
                                 headers={"Content-Type": "application/json", "x-goog-api-key": key})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return r.status, ""
    except urllib.error.HTTPError as e:
        info = ""
        try:
            d = json.loads(e.read())
            for det in d.get("error", {}).get("details", []):
                for v in det.get("violations", []) or []:
                    info += f" quotaId={v.get('quotaId')} value={v.get('quotaValue')}"
                if det.get("retryDelay"):
                    info += f" retry={det['retryDelay']}"
            info = info or d.get("error", {}).get("message", "")[:160]
        except Exception:                                            # noqa: BLE001
            pass
        return e.code, info


def main():
    key = os.environ.get("GEMINI_API_KEY_PH", "").strip()
    if not key:
        print("GEMINI_API_KEY_PH is empty")
        return 1
    print(f"key: {len(key)} characters")
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    bad = 0
    for m in MODELS:
        for i in range(n):
            code, info = one(key, m)
            print(f"{m}: request {i + 1}: HTTP {code}{info}")
            bad += code != 200
            time.sleep(5)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
