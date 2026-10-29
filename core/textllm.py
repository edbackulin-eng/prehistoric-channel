"""The text steps' model: Groq first, Gemini (flash-lite) only when Groq cannot answer.

Script, fact check, plan and topics go through here, so Gemini's free day is
kept for the image checks (pipeline/speciescheck.py, pipeline/maskcheck.py,
the thumbnails). Groq's minute holds 8000 tokens, so callers size their
sources to SOURCE_CHARS (relevant() picks the paragraphs that matter).
"""
import re

from core import gemini, groq, llm_usage
from core.logger import get_logger

log = get_logger("textllm")
SOURCE_CHARS = 11000          # what a Groq request can carry beside its prompt and answer
FALLBACK = ("gemini-flash-lite-latest",)


def ask(prompt, step, json_out=True, temperature=0.4, max_tokens=2200, big=False):
    """big: a request too long for Groq's minute (a whole script) goes to Gemini."""
    if groq.available() and not big and llm_usage.enough(f"groq:{groq.MODEL}", 1, len(prompt) // 3 + max_tokens):
        try:
            return groq.ask(prompt, json_out=json_out, temperature=temperature, max_tokens=max_tokens, step=step)
        except Exception as e:                                           # noqa: BLE001
            log.warning("groq could not answer (%s); Gemini instead", str(e)[:160])
    return gemini.ask(prompt, models=FALLBACK, json_out=json_out, temperature=temperature, step=step)


def relevant(sources, query, limit=SOURCE_CHARS):
    """The source paragraphs that share the most words with query, in their own
    order, up to limit characters, each under its article's title."""
    q = {w for w in re.findall(r"[a-zäöüß]{5,}", query.lower())}
    paras = []
    for si, s in enumerate(sources):
        for pi, p in enumerate(re.split(r"\n\s*\n|\n(?==)", s["text"])):
            p = p.strip()
            if len(p) < 40:
                continue
            words = set(re.findall(r"[a-zäöüß]{5,}", p.lower()))
            paras.append((len(q & words) / (1 + len(words) ** 0.5), si, pi, p, s["title"]))
    paras.sort(key=lambda x: -x[0])
    chosen, n = [], 0
    for score, si, pi, p, title in paras:
        if n + len(p) > limit:
            continue
        chosen.append((si, pi, p, title))
        n += len(p)
    chosen.sort()
    out, last = [], None
    for si, pi, p, title in chosen:
        if title != last:
            out.append(f"### {title}")
            last = title
        out.append(p)
    return "\n\n".join(out)
