"""LLM access: Groq primary, Gemini fallback. Stdlib only."""
import json
import os
import re
import time
import urllib.error
import urllib.request
from pathlib import Path
from threading import Lock

from core.logger import get_logger
from core.retry import retry

log = get_logger("llm")

ROOT = Path(__file__).resolve().parent.parent
ENV_PATH = ROOT / ".env"

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
# llama-3.3-70b-versatile is gone from the free tier: the key answers
# model_not_found for it. gpt-oss-120b is what the account actually has,
# and what the stoicism channel has been running on.
GROQ_MODEL = "openai/gpt-oss-120b"
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
# "gemini-flash-lite" (inherited from yt-factory) is not a served id at all, and
# the models listing is not a reliable guide either: it advertises
# models/gemini-2.5-flash, which then 404s with "no longer available to new
# users". The rolling "-latest" alias is the one that actually answers, and it
# keeps working when Google retires a numbered release.
# Closed Tabs runs on its own Google project and key (GEMINI_API_KEY_CT), so its
# twenty requests a day per model are not shared with What If.
GEMINI_MODEL = "gemini-flash-lite-latest"
GEMINI_KEY_NAME = "GEMINI_API_KEY_PH"   # this channel's own key; never another channel's quota
# Separate free-tier quota bucket from GEMINI_MODEL, for bulk work.
GEMINI_BULK_MODEL = "gemini-flash-lite-latest"

# The free Gemini tier allows far fewer requests than yt-factory assumed at 25:
# at that rate the scene planner drew 17 rate-limit rejections in six minutes and
# got through 10 of 216 scenes. 8/minute keeps it under the cap without backoff.
MAX_RPM = 8
_RATE_WINDOW = 60.0
_MAX_429_WAITS = 5


class LLMError(Exception):
    pass


# --- .env ------------------------------------------------------------------

_env_cache = None


def load_env(path=ENV_PATH):
    """Parse a simple KEY=value .env file. Later keys win; os.environ wins over the file."""
    global _env_cache
    if _env_cache is not None:
        return _env_cache
    values = {}
    path = Path(path)
    if path.exists():
        # utf-8-sig: PowerShell's > and Out-File write a BOM, which would otherwise
        # end up glued to the first key name.
        for raw in path.read_text(encoding="utf-8-sig").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            # "export FOO=bar" is a common shape to paste in; treat it as FOO=bar.
            key = key.strip().removeprefix("export ").strip().lstrip("﻿")
            value = value.strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
                value = value[1:-1]
            if key:
                values[key] = value
    _env_cache = values
    return values


def get_key(name):
    """Environment variable first, then .env. Returns None when unset or blank."""
    value = os.environ.get(name) or load_env().get(name)
    return value.strip() or None if value else None


# --- rate limit ------------------------------------------------------------

_calls = []
_calls_lock = Lock()


def _rate_limit():
    """Block until fewer than MAX_RPM calls sit inside the trailing 60s window."""
    while True:
        with _calls_lock:
            now = time.monotonic()
            _calls[:] = [t for t in _calls if now - t < _RATE_WINDOW]
            if len(_calls) < MAX_RPM:
                _calls.append(now)
                return
            wait = _RATE_WINDOW - (now - _calls[0]) + 0.05
        log.info("rate limit: %d calls in the last minute, waiting %.1fs", MAX_RPM, wait)
        time.sleep(wait)


# --- http ------------------------------------------------------------------

# Cloudflare in front of Groq answers urllib's default User-Agent with a 403
# (its code 1010, a browser-signature ban), so every request carries an
# explicit one. The same curl request succeeds, which is what made this look
# like a bad key rather than a blocked client.
USER_AGENT = "closed-tabs/1.0 (+https://github.com/edbackulin-eng/closed-tabs)"


def _post(url, payload, headers, timeout=180):
    """POST JSON. Returns (status, body_text). HTTP errors come back as values, not exceptions."""
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, method="POST",
                                 headers={"Content-Type": "application/json",
                                          "User-Agent": USER_AGENT, **headers})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, resp.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8", "replace")
    except urllib.error.URLError as e:
        raise LLMError(f"network error calling {url}: {e.reason}") from e


class QuotaExhausted(LLMError):
    """A quota that will not recover within this run - do not retry."""


# Longer than this and the limit is a daily one, not a burst one.
_UNRECOVERABLE_WAIT_S = 300


def _retry_after_seconds(body, attempt):
    """Seconds to wait after a 429. Honours the API's own hint when it gives one.

    Raises QuotaExhausted when the hint is hours away. Gemini's free tier caps
    requests PER DAY per model, and answers an exhausted quota with a retryDelay
    of several hours; backing off 5, 10, 20, 40, 80 seconds against that just
    burns five minutes to arrive at the same failure.
    """
    hint = re.search(r"try again in ([\d.]+)s", body)
    delay = re.search(r'"retryDelay"\s*:\s*"(\d+(?:\.\d+)?)s"', body)
    for match in (hint, delay):
        if match:
            wait = float(match.group(1)) + 1.0
            if wait > _UNRECOVERABLE_WAIT_S:
                quota = re.search(r'"quotaId"\s*:\s*"([^"]+)"', body)
                raise QuotaExhausted(
                    f"quota exhausted, not retryable for {wait / 3600:.1f}h"
                    + (f" ({quota.group(1)})" if quota else ""))
            return min(wait, 120.0)
    return min(5.0 * 2 ** attempt, 120.0)


# --- providers -------------------------------------------------------------

def _call_groq(prompt, system, max_tokens, temperature, json_mode):
    key = get_key("GROQ_API_KEY")
    if not key:
        raise LLMError("GROQ_API_KEY is not set (put it in .env)")
    messages = ([{"role": "system", "content": system}] if system else [])
    messages.append({"role": "user", "content": prompt})
    payload = {"model": GROQ_MODEL, "messages": messages,
               "max_tokens": max_tokens, "temperature": temperature}
    if json_mode:
        payload["response_format"] = {"type": "json_object"}

    for attempt in range(_MAX_429_WAITS):
        _rate_limit()
        started = time.monotonic()
        status, body = _post(GROQ_URL, payload, {"Authorization": f"Bearer {key}"})
        elapsed = time.monotonic() - started
        if status == 429:
            wait = _retry_after_seconds(body, attempt)
            log.warning("groq 429 (attempt %d/%d), waiting %.1fs", attempt + 1, _MAX_429_WAITS, wait)
            time.sleep(wait)
            continue
        if status != 200:
            raise LLMError(f"groq HTTP {status}: {body[:400]}")
        data = json.loads(body)
        usage = data.get("usage") or {}
        log.info("groq %s | %s tokens (%s in, %s out) | %.1fs", GROQ_MODEL,
                 usage.get("total_tokens", "?"), usage.get("prompt_tokens", "?"),
                 usage.get("completion_tokens", "?"), elapsed)
        text = data["choices"][0]["message"]["content"]
        if not (text or "").strip():
            # gpt-oss-120b reasons before answering and the reasoning is
            # charged to max_tokens, so a budget that is too tight comes back
            # with a token count but no content. Returning that empty string
            # would put it straight into a script.
            raise LLMError(
                f"groq returned empty content ({data.get('usage', {})}); "
                f"max_tokens={max_tokens} is likely too tight for a model "
                f"that reasons first")
        return text
    raise LLMError(f"groq still rate-limited after {_MAX_429_WAITS} waits")


def _call_gemini(prompt, system, max_tokens, temperature, json_mode,
                 thinking_budget=0, model=None):
    """Call Gemini.

    `thinking_budget` matters more than it looks. The flash models reason before
    answering and those thinking tokens come out of maxOutputTokens: asking for
    64 tokens spends ~59 on thought, returns one token of answer and a
    finishReason of MAX_TOKENS, which reads downstream as "empty response". The
    default of 0 turns thinking off for the short structured calls this pipeline
    makes; fact-checking passes a real budget (and a matching max_tokens) where
    the reasoning is worth paying for.
    """
    key = get_key(GEMINI_KEY_NAME)
    if not key:
        raise LLMError(f"{GEMINI_KEY_NAME} is not set")
    payload = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {
            "maxOutputTokens": max_tokens,
            "temperature": temperature,
        },
    }
    # A lite model rejects thinkingConfig outright with a 400, so it never
    # goes there. Everywhere else it is sent even when the budget is zero:
    # leaving it out does not mean no thinking, it means the model's own
    # default. gemini-flash-latest spent 383 of a 400-token budget thinking
    # and returned empty text, which read as a parse failure.
    model = model or GEMINI_MODEL
    if "lite" not in model:
        payload["generationConfig"]["thinkingConfig"] = {
            "thinkingBudget": thinking_budget}
    if system:
        payload["systemInstruction"] = {"parts": [{"text": system}]}
    if json_mode:
        payload["generationConfig"]["responseMimeType"] = "application/json"
    url = GEMINI_URL.format(model=model)

    for attempt in range(_MAX_429_WAITS):
        _rate_limit()
        started = time.monotonic()
        status, body = _post(url, payload, {"x-goog-api-key": key})
        elapsed = time.monotonic() - started
        if status == 429:
            wait = _retry_after_seconds(body, attempt)
            log.warning("gemini 429 (attempt %d/%d), waiting %.1fs", attempt + 1, _MAX_429_WAITS, wait)
            time.sleep(wait)
            continue
        if status != 200:
            raise LLMError(f"gemini HTTP {status}: {body[:400]}")
        data = json.loads(body)
        candidates = data.get("candidates") or []
        if not candidates:
            raise LLMError(f"gemini returned no candidates: {body[:400]}")
        usage = data.get("usageMetadata") or {}
        thoughts = usage.get("thoughtsTokenCount", 0)
        log.info("gemini %s | %s tokens (%s thinking) | %.1fs", model,
                 usage.get("totalTokenCount", "?"), thoughts, elapsed)
        candidate = candidates[0]
        parts = candidate.get("content", {}).get("parts") or []
        # Thought parts carry a thoughtSignature and no usable text; skip them.
        text = "".join(p.get("text", "") for p in parts if "text" in p)
        if not text.strip():
            reason = candidate.get("finishReason", "unknown")
            if reason == "MAX_TOKENS":
                raise LLMError(
                    f"gemini hit the {max_tokens}-token cap with {thoughts} spent on "
                    "thinking and none left for the answer; raise max_tokens or "
                    "lower thinking_budget")
            raise LLMError(f"gemini returned no text (finishReason={reason})")
        return text
    raise LLMError(f"gemini still rate-limited after {_MAX_429_WAITS} waits")


# --- public API ------------------------------------------------------------

CLAUDE_TIMEOUT_S = 900
PROVIDERS = ("claude", "gemini", "groq")


def _call_claude(prompt, system, max_tokens, temperature, json_mode):
    """Headless Claude through the CLI, on the subscription.

    In CI it authenticates with CLAUDE_CODE_OAUTH_TOKEN (from `claude setup-token`);
    locally with the logged-in CLI. max_tokens and temperature are not exposed by
    the CLI and are ignored.
    """
    import shutil
    import subprocess
    exe = shutil.which("claude")
    if not exe:
        raise LLMError("claude CLI is not installed")
    full = prompt if not system else f"{system}\n\n{prompt}"
    if json_mode:
        full += "\n\nReply with JSON only, no prose, no code fences."
    started = time.monotonic()
    try:
        # The prompt goes in on stdin, not as an argument: a script prompt with two
        # long sources failed with "the command line is too long", and argument
        # limits differ by platform while stdin has none.
        r = subprocess.run([exe, "-p", "--output-format", "text"], input=full,
                           capture_output=True, text=True, encoding="utf-8",
                           timeout=CLAUDE_TIMEOUT_S, env=os.environ.copy())
    except subprocess.TimeoutExpired as e:
        raise LLMError(f"claude timed out after {CLAUDE_TIMEOUT_S}s") from e
    if r.returncode != 0:
        raise LLMError(f"claude -p exit {r.returncode}: {(r.stderr or r.stdout)[:300]}")
    text = (r.stdout or "").strip()
    if not text:
        raise LLMError("claude -p returned nothing")
    log.info("claude | %d chars | %.1fs", len(text), time.monotonic() - started)
    return text


def complete(prompt, system=None, max_tokens=4096, temperature=0.8, json_mode=False,
             thinking_budget=0, model=None, providers=PROVIDERS):
    """Claude, then Gemini, then Groq. Raises LLMError when none answers.

    Not wrapped in @retry: each provider already retries its own transient
    failures, and retrying the whole chain would run a slow Claude call three
    times before Gemini was ever asked.
    """
    errors = []
    for name in providers:
        try:
            if name == "claude":
                return _call_claude(prompt, system, max_tokens, temperature, json_mode)
            if name == "gemini":
                return _call_gemini(prompt, system, max_tokens, temperature, json_mode,
                                    thinking_budget, model)
            if name == "groq":
                return _call_groq(prompt, system, max_tokens, temperature, json_mode)
        except LLMError as e:
            errors.append(f"{name}: {e}")
            log.warning("%s failed (%s), trying the next provider", name, e)
    raise LLMError("no provider answered: " + " | ".join(errors))


def extract_json(text):
    """Pull a JSON value out of a model reply, including ```json fenced blocks."""
    if not text:
        raise ValueError("empty response")
    fence = re.search(r"```(?:json)?\s*(.+?)```", text, re.S)
    candidate = fence.group(1).strip() if fence else text.strip()
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        pass
    # Fall back to the outermost {...} or [...] span in the text.
    for opener, closer in (("{", "}"), ("[", "]")):
        start, end = candidate.find(opener), candidate.rfind(closer)
        if start != -1 and end > start:
            try:
                return json.loads(candidate[start:end + 1])
            except json.JSONDecodeError:
                continue
    raise ValueError(f"no JSON found in response: {text[:200]}")


def complete_json(prompt, system=None, max_tokens=4096, temperature=0.7,
                  thinking_budget=0, model=None, providers=PROVIDERS):
    """complete() that returns parsed JSON, with one stricter re-ask on a parse failure."""
    text = complete(prompt, system, max_tokens, temperature, json_mode=True,
                    thinking_budget=thinking_budget, model=model, providers=providers)
    try:
        return extract_json(text)
    except ValueError as e:
        log.warning("JSON parse failed (%s), re-asking with a stricter instruction", e)
        strict = (f"{prompt}\n\n"
                  "Your previous answer was not valid JSON. Reply with valid JSON only. "
                  "No prose, no markdown, no code fences.")
        text = complete(strict, system, max_tokens, temperature=0.2,
                        json_mode=True, model=model, providers=providers)
        return extract_json(text)
