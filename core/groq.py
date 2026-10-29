"""Groq for the channel's text steps (script, fact check, plan), so Gemini's free
day is left to the image checks. This channel's own key: GROQ_API_KEY_PH.

Model openai/gpt-oss-120b: 30 requests a minute, 1000 a day, 8000 tokens a
minute and 200000 a day (console.groq.com/docs/rate-limits). The minute's token
cap is the tight one - a request is input and answer together - so requests are
kept under it (callers trim their sources) and paced: a request that would pass
the minute's tokens waits for the minute to clear.
"""
import json
import os
import time
import urllib.error
import urllib.request
from collections import deque

from core import llm_usage
from core.logger import get_logger

log = get_logger("groq")
URL = "https://api.groq.com/openai/v1/chat/completions"
MODEL = "openai/gpt-oss-120b"
TPM = 8000
_WINDOW = deque()                      # (time, tokens) of the last minute


def available():
    return bool(os.environ.get("GROQ_API_KEY_PH", "").strip())


def _pace(tokens):
    now = time.time()
    while _WINDOW and now - _WINDOW[0][0] > 60:
        _WINDOW.popleft()
    spent = sum(t for _, t in _WINDOW)
    if spent + tokens > TPM * 0.9 and _WINDOW:
        wait = 61 - (now - _WINDOW[0][0])
        log.info("groq: %d tokens this minute; waiting %.0f s", spent, wait)
        time.sleep(max(1, wait))


def ask(prompt, json_out=True, temperature=0.4, max_tokens=2500, step=None):
    key = os.environ["GROQ_API_KEY_PH"].strip()
    est = len(prompt) // 3 + max_tokens
    if est > TPM:
        raise ValueError(f"request of ~{est} tokens is over Groq's {TPM} a minute; trim it")
    payload = {"model": MODEL, "messages": [{"role": "user", "content": prompt}], "temperature": temperature,
               "max_tokens": max_tokens}
    if json_out:
        payload["response_format"] = {"type": "json_object"}
    last = None
    for attempt in range(6):
        _pace(est)
        try:
            req = urllib.request.Request(URL, data=json.dumps(payload).encode(), headers={
                "Content-Type": "application/json", "Authorization": f"Bearer {key}",
                "User-Agent": "prehistoric-channel/1.0"})       # Cloudflare refuses urllib's default
            with urllib.request.urlopen(req, timeout=180) as r:
                d = json.loads(r.read())
            u = d.get("usage", {})
            _WINDOW.append((time.time(), u.get("total_tokens", est)))
            llm_usage.record(f"groq:{MODEL}", u.get("prompt_tokens"), u.get("completion_tokens"), step)
            text = d["choices"][0]["message"]["content"] or ""
            log.info("groq %s: %s tokens", MODEL, u.get("total_tokens"))
            if not json_out:
                return text
            return json.loads(text[min(i for i in (text.find("{"), text.find("[")) if i >= 0):])
        except urllib.error.HTTPError as e:
            body = e.read()[:400].decode("utf-8", "replace")
            last = f"HTTP {e.code} {body[:200]}"
            if e.code == 429:
                wait = float(e.headers.get("retry-after") or 30)
                if "per day" in body.lower() or wait > 600:
                    raise RuntimeError(f"groq: the day is spent ({body[:160]})")
                time.sleep(min(wait, 120) + 1)
                continue
            if e.code >= 500:
                time.sleep(20 * (attempt + 1))
                continue
            raise RuntimeError(f"groq: {last}")
        except (ValueError, KeyError) as e:
            last = f"{type(e).__name__}: {e}"
            time.sleep(3)
    raise RuntimeError(f"groq gave no answer: {last}")
