"""Is it the animal asked for? A first answer on the CPU, before Gemini (owner 08.11).

Gemini's free day is the narrowest point of the species check (pipeline/speciescheck.py):
megalodon needed ~274 image requests on 08.11 with 229 left and lost the Sunday GPU window.
SigLIP (open_clip ViT-B-16-SigLIP, webli; Apache-2.0) scores a stock clip's poster frame
against what the beat asks for - the species' name, or the footage search for a beat with no
animal - on the runner's CPU:

    score <= reject   turned down without Gemini (a verdict "by siglip" in species_verdicts.json):
                      the next alternate is tried
    above it          Gemini decides, as before

The CPU never takes a clip (owner 08.11): the species must be right, and Gemini confirms every
clip that can be shown. An "accept" threshold in the config is ignored (speciescheck.cpu_verdict).

The thresholds are set on Gemini's own answers kept in episodes/*/species_verdicts.json
(scripts/calibrate_prefilter.py) and kept in config/channel.json "prefilter", per kind of beat
("species" and "scene"). Without them (or without the model) nothing is decided locally.

The weights (~800 MB) live in assets/models/open_clip, kept by actions/cache.
"""
import json
import os
import urllib.request
from pathlib import Path

from core.logger import get_logger

log = get_logger("prefilter")
ROOT = Path(__file__).resolve().parent.parent
CFG = json.loads((ROOT / "config" / "channel.json").read_text(encoding="utf-8")).get("prefilter") or {}
ARCH = CFG.get("arch", "ViT-B-16-SigLIP")
PRETRAINED = CFG.get("pretrained", "webli")
CACHE_DIR = ROOT / "assets" / "models" / "open_clip"
os.environ.setdefault("HF_HOME", str(CACHE_DIR / "hf"))      # the tokenizer too, in the cached folder
POSTERS = ROOT / "cache" / "posters"
_m = {}
_img_emb, _txt_emb = {}, {}


def thresholds(kind):
    """(accept, reject) for 'species' or 'scene', or None when they have not been set."""
    t = (CFG.get("thresholds") or {}).get(kind) or {}
    a, r = t.get("accept"), t.get("reject")
    return (None if a is None else float(a), None if r is None else float(r)) if (a is not None or r is not None) \
        else None


def _model():
    if "m" not in _m:
        import open_clip
        import torch
        torch.set_num_threads(max(1, os.cpu_count() or 1))
        model, _, pre = open_clip.create_model_and_transforms(ARCH, pretrained=PRETRAINED, cache_dir=str(CACHE_DIR))
        model.eval()
        _m.update(m=model, pre=pre, tok=open_clip.get_tokenizer(ARCH), torch=torch)
        log.info("prefilter: %s (%s) loaded", ARCH, PRETRAINED)
    return _m


def available():
    if "ok" not in _m:
        try:
            _model()
            _m["ok"] = True
        except Exception as e:                                       # noqa: BLE001
            log.warning("prefilter: no local model (%s); every picture goes to Gemini", e)
            _m["ok"] = False
    return _m["ok"]


def poster(src, cid):
    """The clip's poster frame (Pixabay serves one beside every video), downloaded once; None if none."""
    url = (src or {}).get("thumb") or ""
    u = (src or {}).get("url") or ""
    if not url and "cdn.pixabay.com/video/" in u and u.endswith(".mp4"):
        url = u[:-4] + ".jpg"
    if not url:
        return None
    p = POSTERS / f"{cid.replace('/', '_')}.jpg"
    if p.exists() and p.stat().st_size > 1000:
        return p
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        req = urllib.request.Request(url, headers={"User-Agent": "prehistoric-channel/1.0"})
        with urllib.request.urlopen(req, timeout=20) as r:
            p.write_bytes(r.read())
        return p
    except Exception as e:                                           # noqa: BLE001
        log.info("prefilter: no poster for %s (%s)", cid, e)
        return None


def text_for(species_name=None, query=None):
    if species_name:
        return f"a photo of {species_name}"
    return f"a photo of {query or 'a prehistoric scene'}"


def scores(pairs):
    """pairs: [(text, image path)] -> [cosine similarity]."""
    if not pairs:
        return []
    m = _model()
    from PIL import Image
    texts = [t for t, _ in pairs]
    todo_t = [t for t in dict.fromkeys(texts) if t not in _txt_emb]
    for k in range(0, len(todo_t), 64):
        part = todo_t[k:k + 64]
        with m["torch"].no_grad():
            e = m["m"].encode_text(m["tok"](part))
            e = e / e.norm(dim=-1, keepdim=True)
        _txt_emb.update(zip(part, e))
    todo_i = [str(p) for p in dict.fromkeys(str(p) for _, p in pairs) if str(p) not in _img_emb]
    for k in range(0, len(todo_i), 32):
        part = todo_i[k:k + 32]
        x = m["torch"].stack([m["pre"](Image.open(p).convert("RGB")) for p in part])
        with m["torch"].no_grad():
            e = m["m"].encode_image(x)
            e = e / e.norm(dim=-1, keepdim=True)
        _img_emb.update(zip(part, e))
    return [float(_img_emb[str(p)] @ _txt_emb[t]) for t, p in pairs]


def decide(score, th):
    """'accept', 'reject' or 'ask'."""
    if th is None:
        return "ask"
    a, r = th
    if a is not None and score >= a:
        return "accept"
    if r is not None and score <= r:
        return "reject"
    return "ask"
