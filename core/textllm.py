"""The text steps' model: Gemma 3 27B on this channel's Gemini key, flash-lite in reserve.

Script, fact check, hook, plan and topics go through here. Gemma has its own
quota on the Gemini API, apart from flash and flash-lite, so the flash models'
free day is kept for the image checks (pipeline/speciescheck.py,
pipeline/maskcheck.py, the thumbnails). Its minute holds few tokens, so callers
size their sources to SOURCE_CHARS (relevant() picks the paragraphs that
matter) and core/gemini.py paces the requests. Groq stays in the code, switched
off in config/channel.json ("text_llm").
"""
import re

from core import gemini, groq, llm_usage
from core.logger import get_logger

log = get_logger("textllm")
SOURCE_CHARS = 11000          # what one request can carry beside its prompt and answer
_C = llm_usage.CFG["text_llm"]
PRIMARY, FALLBACK = _C["primary"], _C["fallback"]


def ask(prompt, step, json_out=True, temperature=0.4, max_tokens=2200, big=False):
    """big: a request too long for the primary model's minute (a whole script) goes to the reserve."""
    if _C.get("groq_enabled") and groq.available() and not big and \
            llm_usage.enough(f"groq:{groq.MODEL}", 1, len(prompt) // 3 + max_tokens):
        try:
            return groq.ask(prompt, json_out=json_out, temperature=temperature, max_tokens=max_tokens, step=step)
        except Exception as e:                                           # noqa: BLE001
            log.warning("groq could not answer (%s)", str(e)[:160])
    order = (FALLBACK,) if big or not llm_usage.enough(PRIMARY, 1) else (PRIMARY, FALLBACK)
    return gemini.ask(prompt, models=order, json_out=json_out, temperature=temperature, max_tokens=max_tokens,
                      step=step)


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
