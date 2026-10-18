"""The sources a script is written from: Wikipedia plus independent open pages.

Every Closed Tabs script is written only from what is gathered here, and every
source used is listed in the video description. A topic with fewer than two
usable sources does not get a script.

"Usable" is checked, not assumed: the page has to download and yield a few
paragraphs of readable text. A link count on a Wikipedia article says nothing
about whether any of those pages still exist.
"""
import html
import time
import json
import re
import urllib.parse
import urllib.request

from core.logger import get_logger

log = get_logger("sources")

UA = {"User-Agent": "closed-tabs/1.0 (+https://github.com/edbackulin-eng/closed-tabs)"}
WIKI_API = "https://en.wikipedia.org/w/api.php"
MIN_CHARS = 1500           # below this a page is a stub, a paywall or an error page
MAX_CHARS = 40000          # enough for a script; keeps prompts bounded
MAX_TRIES = 25             # external links tried per topic before giving up
TIMEOUT_S = 20

# Not independent of Wikipedia, or not a page of prose.
_SKIP = re.compile(r"wikipedia\.org|wikimedia\.org|wikidata\.org|archive\.org/details|"
                   r"\.(pdf|jpg|jpeg|png|gif|mp3|mp4|zip)(\?|$)|youtube\.com|youtu\.be|"
                   r"twitter\.com|x\.com|facebook\.com|instagram\.com|reddit\.com|"
                   r"tiktok\.com|imdb\.com|google\.com|doi\.org|worldcat\.org|"
                   r"jstor\.org|books\.google", re.I)


def _get(url, timeout=TIMEOUT_S, attempts=1):
    last = None
    for attempt in range(attempts):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=timeout) as r:
                raw = r.read(3_000_000)
                charset = r.headers.get_content_charset() or "utf-8"
            return raw.decode(charset, "replace")
        except Exception as e:                                   # noqa: BLE001
            last = e
            time.sleep(3 * (attempt + 1))
    raise last


def wikipedia(title):
    """Plain text and external links of one article. Raises when it is missing."""
    q = urllib.parse.urlencode({"action": "query", "titles": title, "redirects": 1,
                                "prop": "extracts|extlinks|info", "inprop": "url",
                                "explaintext": 1, "ellimit": 500,
                                "format": "json", "formatversion": 2})
    # Wikipedia is the one source every topic needs, so a dropped connection is
    # retried rather than taken as "no article".
    page = json.loads(_get(f"{WIKI_API}?{q}", attempts=4))["query"]["pages"][0]
    if page.get("missing"):
        raise LookupError(f"no Wikipedia article {title!r}")
    return {"title": page["title"], "url": page["fullurl"],
            "text": page.get("extract", "")[:MAX_CHARS],
            "links": [l["url"] for l in page.get("extlinks", [])]}


def readable_text(page_html):
    """Paragraph text out of an HTML page, without a parser dependency."""
    body = re.sub(r"(?is)<(script|style|nav|header|footer|aside|form|noscript)[^>]*>.*?</\1>",
                  " ", page_html)
    paras = re.findall(r"(?is)<p[^>]*>(.*?)</p>", body)
    text = "\n".join(re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", "", p))).strip()
                     for p in paras)
    return "\n".join(line for line in text.splitlines() if len(line) > 60)


def _title(page_html, url):
    m = re.search(r"(?is)<title[^>]*>(.*?)</title>", page_html)
    return re.sub(r"\s+", " ", html.unescape(m.group(1))).strip()[:150] if m else url


def _mentions(text, names):
    """Whether the page is actually about the topic: one of its names appears."""
    flat = re.sub(r"[^a-z0-9]+", " ", text.lower())
    for name in names:
        key = re.sub(r"[^a-z0-9]+", " ", name.lower()).strip()
        key = re.sub(r"^the ", "", key)
        if key and f" {key} " in f" {flat} ":
            return True
    return False


def independent(links, want=1, names=()):
    """The first `want` external pages that yield readable prose about the topic.

    A link on the article is not proof the page is about the topic: an archived
    2015 article about a different puzzle was picked up for a 2017 video before
    this checked for the topic's name in the text.
    """
    found, tried = [], 0
    for url in links:
        if len(found) >= want or tried >= MAX_TRIES:
            break
        if not url.startswith("http") or _SKIP.search(url):
            continue
        tried += 1
        try:
            page = _get(url)
        except Exception as e:                                   # noqa: BLE001
            log.info("  skip %s (%s)", url[:80], str(e)[:60])
            continue
        text = readable_text(page)
        if len(text) < MIN_CHARS:
            log.info("  skip %s (%d chars of prose)", url[:80], len(text))
            continue
        if names and not _mentions(text, names):
            log.info("  skip %s (does not mention the topic)", url[:80])
            continue
        found.append({"title": _title(page, url), "url": url, "text": text[:MAX_CHARS]})
    return found


def gather(topic, extra=1):
    """Wikipedia plus `extra` independent sources, or LookupError."""
    wiki = wikipedia(topic["wikipedia"])
    names = {topic["name"], wiki["title"], re.sub(r"\s*\(.*?\)", "", wiki["title"]),
             *topic.get("aliases", [])}
    others = independent(wiki["links"], want=extra, names=names)
    sources = [{"title": f"Wikipedia: {wiki['title']}", "url": wiki["url"],
                "text": wiki["text"]}] + others
    if len(sources) < 2:
        raise LookupError(f"{topic['name']}: only {len(sources)} usable source(s)")
    return sources
