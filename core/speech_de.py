"""German narration text: what each voice engine is actually given to read.

The script is written for the screen (subtitles keep "22. Dezember 1938",
"J. L. B. Smith"); the voices get the spoken form:
  - numbers in words, with years read as years ("neunzehnhundertachtunddreißig"),
    a day of the month as an ordinal, "1990er" as "neunzehnhundertneunziger";
  - names a German front end gets wrong from voice/pronunciations_de.json:
    Piper takes their phonemes ([[ ... ]], spliced past espeak-ng's guess),
    Chatterbox takes text only and gets the respelling.
"""
import json
import re
from pathlib import Path

from num2words import num2words

ROOT = Path(__file__).resolve().parent.parent
MONTHS = ("Januar|Februar|März|April|Mai|Juni|Juli|August|September|Oktober|November|Dezember")
_DATE = re.compile(r"\b(der|die|das|dem|den|am|vom|zum|bis zum)?\s*\b(\d{1,2})\.\s+(" + MONTHS + r")\b", re.I)
_DECADE = re.compile(r"\b(1\d|20)(\d0)er\b")
_YEAR = re.compile(r"\b(1[0-9]{3}|20[0-9]{2})\b")
_INT = re.compile(r"\b\d+\b")


def lexicon():
    data = json.loads((ROOT / "voice" / "pronunciations_de.json").read_text(encoding="utf-8"))
    return {k: v for k, v in data.items() if not k.startswith("_")}


def numbers(text):
    def date(m):
        art, day, month = m.group(1), int(m.group(2)), m.group(3)
        ending = "e" if art and art.lower() in ("der", "die", "das") else "en"
        word = num2words(day, lang="de", to="ordinal")          # "zweiundzwanzigste"
        word = word[:-1] + ending if word.endswith("e") else word + ending
        return f"{art + ' ' if art else ''}{word} {month}"
    text = _DATE.sub(date, text)
    text = _DECADE.sub(lambda m: num2words(int(m.group(1) + m.group(2)), lang="de", to="year") + "er", text)
    text = _YEAR.sub(lambda m: num2words(int(m.group(1)), lang="de", to="year"), text)
    text = _INT.sub(lambda m: num2words(int(m.group(0)), lang="de"), text)
    # a sentence that opened with a number opens with a capital
    return re.sub(r"(^|[.!?]\s+)([a-zäöü])", lambda m: m.group(1) + m.group(2).upper(), text)


def _word_re(lex):
    words = sorted(lex, key=len, reverse=True)
    return re.compile(r"(?<![\w-])(" + "|".join(map(re.escape, words)) + r")(?![\w-])")


def for_piper(text, lex=None):
    lex = lex if lex is not None else lexicon()
    return numbers(_word_re(lex).sub(lambda m: f"[[ {lex[m.group(1)]['ipa']} ]]", text))


def for_text_engine(text, lex=None):
    """Chatterbox and any other engine that reads plain text."""
    lex = lex if lex is not None else lexicon()
    return numbers(_word_re(lex).sub(lambda m: lex[m.group(1)]["respell"], text))
