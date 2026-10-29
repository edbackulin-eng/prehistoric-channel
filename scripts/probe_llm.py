"""What Gemma 3 27B really allows on this channel's key, and how it writes against flash-lite.

    python scripts/probe_llm.py        (from .github/workflows/probe.yml)

1. the minute: tiny requests as fast as they go until the first 429, whose body
   names the limit it hit (quotaId, quotaValue); then one request larger than
   the token minute is thought to be, to see that limit's answer too;
2. one chapter of the first episode (the 1938 rediscovery of the coelacanth),
   from the same prompt and the same sources, by Gemma and by flash-lite.
Writes work/probe/probe_report.json. Costs some 40 Gemma requests (of its day)
and two flash-lite requests.
"""
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from core import gemini, llm_usage, textllm  # noqa: E402

GEMMA = "gemma-3-27b-it"
OUT = ROOT / "work" / "probe"


def models():
    """The models this key may call (generateContent), Gemma first."""
    out, token = [], ""
    while True:
        url = "https://generativelanguage.googleapis.com/v1beta/models?pageSize=200" + (f"&pageToken={token}" if token else "")
        d = json.loads(urllib.request.urlopen(urllib.request.Request(url, headers={"x-goog-api-key": gemini.key()}),
                                              timeout=60).read())
        for m in d.get("models", []):
            if "generateContent" in m.get("supportedGenerationMethods", []):
                out.append({"name": m["name"].split("/", 1)[1], "in": m.get("inputTokenLimit"),
                            "out": m.get("outputTokenLimit")})
        token = d.get("nextPageToken")
        if not token:
            return out


def biggest_gemma(ms):
    import re
    g = [m for m in ms if m["name"].startswith("gemma") and "-it" in m["name"] and "e2b" not in m["name"]
         and "e4b" not in m["name"]]
    size = lambda n: float((re.search(r"(\d+(?:\.\d+)?)b", n) or [0, 0])[1])
    ver = lambda n: float((re.search(r"gemma-(\d+(?:\.\d+)?)", n) or [0, 0])[1])
    g.sort(key=lambda m: (-ver(m["name"]), -size(m["name"])))
    return g[0]["name"] if g else None


def raw(model, text, max_tokens=256):
    body = json.dumps({"contents": [{"parts": [{"text": text}]}],
                       "generationConfig": {"maxOutputTokens": max_tokens, "temperature": 0}}).encode()
    req = urllib.request.Request(gemini.URL.format(m=model), data=body,
                                 headers={"Content-Type": "application/json", "x-goog-api-key": gemini.key()})
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            d = json.loads(r.read())
        um = d.get("usageMetadata", {})
        llm_usage.record(model, um.get("promptTokenCount"), um.get("candidatesTokenCount"), "probe")
        return 200, um, ""
    except urllib.error.HTTPError as e:
        b = e.read()[:6000].decode("utf-8", "replace")
        return e.code, {}, b


def minute():
    rep = {"requests_before_429": None, "429": None}
    t0 = time.time()
    for i in range(45):
        code, um, body = raw(GEMMA, "Antworte nur mit: OK")
        if code == 429:
            rep["requests_before_429"] = i
            rep["seconds"] = round(time.time() - t0, 1)
            rep["429"] = gemini._quota(body) or body[:500]
            break
        if code != 200:
            rep["error"] = f"HTTP {code}: {body[:300]}"
            break
    else:
        rep["requests_before_429"] = "none in 45"
        rep["seconds"] = round(time.time() - t0, 1)
    time.sleep(65)                                # the minute clears
    big = "Der Quastenflosser lebt in der Tiefe des Indischen Ozeans. " * 1500   # ~18k tokens
    code, um, body = raw(GEMMA, big + "\nAntworte nur mit: OK")
    rep["big_request"] = {"approx_tokens": um.get("promptTokenCount") or len(big) // 4, "http": code,
                          "answer": gemini._quota(body) or body[:400]}
    return rep


def chapter():
    sys.path.insert(0, str(ROOT / "automation"))
    import write_episode as we
    q = json.loads((ROOT / "topics" / "queue.json").read_text(encoding="utf-8"))
    topic = next(t for t in q["topics"] if t["slug"] == "lebende-fossilien")
    src = we.sources(topic)
    c = {"section": "story", "heading": "Die Wiederentdeckung der Latimeria 1938",
         "points": ["der Fischkutter vor East London im Dezember 1938", "Marjorie Courtenay-Latimer und das Museum",
                    "J. L. B. Smith erkennt den Quastenflosser", "warum der Fund die Wissenschaft erschütterte"]}
    q_ = f"{c['heading']} {' '.join(c['points'])}"
    prompt = we.PART.format(title=topic["title"], section=c["section"], heading=c["heading"],
                            points="; ".join(c["points"]), words=300, before="(Anfang des Kapitels)",
                            sources=textllm.relevant(src, q_, 7500))
    out = {}
    for m in (GEMMA,):                            # flash-lite's chapter is in the first probe's report
        t0 = time.time()
        try:
            ans = gemini.ask(prompt, models=(m,), temperature=0.6, max_tokens=1700, step="probe")
            lines = [l for l in gemini.pick(ans, "lines") if isinstance(l, str)]
            out[m] = {"seconds": round(time.time() - t0, 1), "words": len(" ".join(lines).split()), "lines": lines}
        except Exception as e:                                           # noqa: BLE001
            out[m] = {"error": str(e)[:300]}
    return out


def main():
    global GEMMA
    OUT.mkdir(parents=True, exist_ok=True)
    ms = models()
    # which Gemma answers at all (29.10: gemma-4-31b-it gave only 500s); the first that does is probed
    tries = {}
    for m in [x["name"] for x in ms if x["name"].startswith("gemma")]:
        code, um, body = raw(m, "Antworte nur mit: OK")
        if code >= 500:
            time.sleep(20)
            code, um, body = raw(m, "Antworte nur mit: OK")
        tries[m] = {"http": code, "body": body[:200]}
    (OUT / "gemma_tries.json").write_text(json.dumps(tries, indent=1), encoding="utf-8")
    print("gemma tries:", tries, flush=True)
    ok = [m for m, v in tries.items() if v["http"] == 200]
    GEMMA = (sorted(ok, key=lambda n: n != biggest_gemma(ms))[0] if ok else biggest_gemma(ms)) or GEMMA
    print("models:", [m["name"] for m in ms], "| Gemma used:", GEMMA, flush=True)
    rep = {"models": ms, "gemma": GEMMA, "gemma_tries": tries, "minute": minute()}
    rep.update({"chapter": chapter(), "used_today": {m: llm_usage.used(m) for m in (GEMMA, "gemini-flash-lite-latest")}})
    (OUT / "probe_report.json").write_text(json.dumps(rep, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(rep, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
