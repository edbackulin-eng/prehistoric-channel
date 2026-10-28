"""The 2.5D layers on a GitHub runner's CPU: everything after FLUX.

Kaggle's GPU makes only the FLUX frames (gpu/kernels/recon_job.py, flux-only).
Everything else that turns a frame or a real photograph into a shot runs here,
spread over a matrix of runners (.github/workflows/render.yml, job "layers"):

  1. Real-ESRGAN x4plus      upscale to the 4224x2376 working frame
                             (photos already that large are only resized);
  2. Depth Anything V2 Small depth, kept for the background layers;
  3. BiRefNet + IS-Net        the subject's mask; both must agree (IoU >= 0.5)
                             and the mask must pass the checks below, or the
                             shot gets a plain Ken Burns move;
  4. LaMa (big-lama)          the background behind the subject, filled in.

Same models, same rules as the GPU kernel did them (25-28.10); every stage is
timed per frame, so the report can say what a frame costs on a runner.

    python -m pipeline.layers_cpu <slug> <shard> <shards>
"""
import json
import os
import sys
import time
import urllib.request
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "assets" / "models" / "layers"
W, H = 4224, 2376
SW, SH = 1408, 792                  # working scale for depth, mask and fill
DILATE = 22                         # at SW: covers the largest parallax offset
UA = {"User-Agent": "prehistoric-channel/0.1 (research)"}
WEIGHTS = {
    "esrgan": ("https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
               "RealESRGAN_x4plus.pth"),
    "lama": ("https://github.com/enesmsahin/simple-lama-inpainting/releases/download/v0.1.0/big-lama.pt",
             "big-lama.pt"),
}
_M = {}


def _weight(name):
    url, fn = WEIGHTS[name]
    p = MODELS / fn
    if not p.exists():
        MODELS.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(url, p)
    return p


def cover(im, w=W, h=H):
    s = max(w / im.width, h / im.height)
    im = im.resize((round(im.width * s), round(im.height * s)), Image.LANCZOS)
    left, top = (im.width - w) // 2, (im.height - h) // 2
    return im.crop((left, top, left + w, top + h))


# ------------------------------------------------------------- 1. upscale
def upscale(im, tile=256, pad=16):
    import torch
    if "esrgan" not in _M:
        from spandrel import ModelLoader
        _M["esrgan"] = ModelLoader().load_from_file(str(_weight("esrgan"))).eval()
    sr = _M["esrgan"]
    x = torch.from_numpy(np.asarray(im.convert("RGB"))).permute(2, 0, 1).float().div(255)[None]
    _, _, h, w = x.shape
    out = torch.zeros(1, 3, h * 4, w * 4)
    with torch.inference_mode():
        for y0 in range(0, h, tile):
            for x0 in range(0, w, tile):
                ya, yb = max(0, y0 - pad), min(h, y0 + tile + pad)
                xa, xb = max(0, x0 - pad), min(w, x0 + tile + pad)
                r = sr(x[:, :, ya:yb, xa:xb])
                oy, ox = (y0 - ya) * 4, (x0 - xa) * 4
                th, tw = min(tile, h - y0) * 4, min(tile, w - x0) * 4
                out[:, :, y0 * 4:y0 * 4 + th, x0 * 4:x0 * 4 + tw] = r[:, :, oy:oy + th, ox:ox + tw]
    return Image.fromarray((out[0].clamp(0, 1).permute(1, 2, 0).numpy() * 255).round().astype(np.uint8))


# --------------------------------------------------------------- 2. depth
def depth(small):
    import torch
    if "depth" not in _M:
        from transformers import pipeline as hf_pipeline
        _M["depth"] = hf_pipeline("depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf", device=-1)
    r = _M["depth"](small)
    d = r["predicted_depth"]
    d = d.squeeze().float().numpy() if torch.is_tensor(d) else np.asarray(r["depth"], np.float32)
    d = cv2.resize(d, (SW, SH), interpolation=cv2.INTER_CUBIC)
    return (d - d.min()) / max(1e-6, d.max() - d.min())          # 1 = near


# ---------------------------------------------------------------- 3. masks
def seg_birefnet(img):
    import torch
    if "birefnet" not in _M:
        from transformers import AutoModelForImageSegmentation
        _M["birefnet"] = AutoModelForImageSegmentation.from_pretrained(
            "ZhengPeng7/BiRefNet", trust_remote_code=True).eval()
    x = np.asarray(img.resize((1024, 1024), Image.BILINEAR), np.float32) / 255
    x = (x - np.array([0.485, 0.456, 0.406], np.float32)) / np.array([0.229, 0.224, 0.225], np.float32)
    t = torch.from_numpy(x).permute(2, 0, 1)[None]
    with torch.inference_mode():
        pr = _M["birefnet"](t)[-1].sigmoid().float().numpy()[0, 0]
    return cv2.resize(pr, img.size, interpolation=cv2.INTER_LINEAR)


def seg_isnet(img):
    from rembg import new_session, remove
    if "isnet" not in _M:
        _M["isnet"] = new_session("isnet-general-use", providers=["CPUExecutionProvider"])
    return np.asarray(remove(img, session=_M["isnet"], only_mask=True), np.float32) / 255


def mask_checks(prob):
    """A usable animal mask: a sensible share of the frame, one dominant piece,
    and its centre of mass on (or right next to) the object, not in a gap."""
    b = (prob > 0.5).astype(np.uint8)
    h_, w_ = b.shape
    area = float(b.mean())
    n_, lab_, st_, _ = cv2.connectedComponentsWithStats(b)
    if n_ <= 1:
        return {"ok": False, "why": "empty", "area": 0.0}
    big = 1 + int(np.argmax(st_[1:, cv2.CC_STAT_AREA]))
    share = float(st_[big, cv2.CC_STAT_AREA] / b.sum())
    keep = ((lab_ == big) | np.isin(lab_, [i for i in range(1, n_)
                                           if st_[i, cv2.CC_STAT_AREA] >= 0.15 * st_[big, cv2.CC_STAT_AREA]]))
    keep = keep.astype(np.uint8)
    ys_, xs_ = np.nonzero(keep)
    cx, cy = float(xs_.mean()), float(ys_.mean())
    near = cv2.dilate(keep, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(0.06 * w_) | 1,) * 2))
    com_on = bool(near[int(cy), int(cx)])
    ok = 0.02 <= area <= 0.65 and share >= 0.8 and com_on
    why = [] if ok else [w for w, bad in (("area", not 0.02 <= area <= 0.65), ("pieces", share < 0.8),
                                          ("centre off the object", not com_on)) if bad]
    return {"ok": ok, "why": ", ".join(why), "area": round(area, 3), "main_piece": round(share, 2),
            "centre": [round(cx / w_, 2), round(cy / h_, 2)], "centre_on_object": com_on, "mask": keep}


# ------------------------------------------------------------------ 4. fill
def lama_fill(small, hole):
    import torch
    if "lama" not in _M:
        _M["lama"] = torch.jit.load(str(_weight("lama")), map_location="cpu").eval()
    x = torch.from_numpy(np.asarray(small)).permute(2, 0, 1).float().div(255)[None]
    m = torch.from_numpy(hole).float()[None, None]
    with torch.inference_mode():
        return _M["lama"](x, m)[0].clamp(0, 1).permute(1, 2, 0).numpy()


def frame(fid, src, up_dir, lay_dir, is_photo):
    """One frame or photo, all four stages. Returns its timings and mask info."""
    t = {}
    t0 = time.perf_counter()
    im = Image.open(src).convert("RGB")
    if im.width < 2400:                       # FLUX frames (1360 px) and small photos
        if is_photo:
            im.thumbnail((1400, 1400))
        im = upscale(im)
    t["esrgan_s"] = round(time.perf_counter() - t0, 1)
    im = cover(im)
    im.save(up_dir / f"{fid}.jpg", quality=94)
    small = im.resize((SW, SH), Image.LANCZOS)

    t0 = time.perf_counter()
    d = depth(small)
    cv2.imwrite(str(lay_dir / f"{fid}_depth.png"), (d * 65535).astype(np.uint16))
    t["depth_s"] = round(time.perf_counter() - t0, 1)

    t0 = time.perf_counter()
    probs, errs = {}, []
    for name, fn in (("birefnet", seg_birefnet), ("isnet", seg_isnet)):
        try:
            probs[name] = fn(small)
        except Exception as e:                                       # noqa: BLE001
            errs.append(f"{name}: {type(e).__name__}: {str(e)[:200]}")
    checks = {k: mask_checks(v) for k, v in probs.items()}
    iou = None
    if len(probs) == 2:
        a, b = probs["birefnet"] > 0.5, probs["isnet"] > 0.5
        iou = float((a & b).sum() / max(1, (a | b).sum()))
    chosen = next((k for k in ("birefnet", "isnet") if k in checks and checks[k]["ok"]), None)
    if chosen and iou is not None and iou < 0.5:
        chosen = None                          # the two models disagree about what the subject is
    t["segment_s"] = round(time.perf_counter() - t0, 1)

    t0 = time.perf_counter()
    fg = checks[chosen]["mask"] if chosen else np.zeros((SH, SW), np.uint8)
    prob = probs[chosen] * fg if chosen else fg.astype(np.float32)
    alpha = cv2.GaussianBlur(np.clip(prob, 0, 1).astype(np.float32), (0, 0), 1.2)
    hole = cv2.dilate(fg, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * DILATE + 1, 2 * DILATE + 1)))
    if chosen and fg.mean() > 0.002:
        filled = cv2.resize((lama_fill(small, hole) * 255).astype(np.uint8), (W, H), interpolation=cv2.INTER_CUBIC)
        full = np.asarray(im).copy()
        hb = cv2.resize(cv2.GaussianBlur(hole.astype(np.float32), (0, 0), 2), (W, H))[..., None]
        Image.fromarray((full * (1 - hb) + filled * hb).astype(np.uint8)).save(lay_dir / f"{fid}_bg.jpg", quality=93)
        cv2.imwrite(str(lay_dir / f"{fid}_alpha.png"),
                    (cv2.resize(alpha, (W, H), interpolation=cv2.INTER_LINEAR) * 255).astype(np.uint8))
    t["lama_s"] = round(time.perf_counter() - t0, 1)
    t["total_s"] = round(sum(t.values()), 1)
    return {"id": fid, "photo": is_photo, **t, "fg_area": round(float(fg.mean()), 3),
            "segmentation": chosen or "none - Ken Burns", "iou_birefnet_isnet": None if iou is None else round(iou, 3),
            "checks": {k: {kk: vv for kk, vv in c.items() if kk != "mask"} for k, c in checks.items()},
            "seg_errors": errs}


def items(slug, raw_dir):
    """Every frame and photo of the episode, in a fixed order: (id, source, is_photo)."""
    spec = json.loads((ROOT / "episodes" / slug / "recon.json").read_text(encoding="utf-8"))
    out = [(p.stem, p, False) for p in sorted(Path(raw_dir).glob("*.jpg")) if not p.stem.startswith("thumb_")]
    out += [(pid, ph["url"], True) for pid, ph in spec.get("photos", {}).items()]
    return out


def fetch_photo(url, dest):
    for k in range(5):
        try:
            dest.write_bytes(urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=120).read())
            return dest
        except Exception:                                            # noqa: BLE001
            if k == 4:
                raise
            time.sleep(8 * (k + 1))


def main(slug, shard, shards, raw_dir=None):
    import torch
    torch.set_num_threads(os.cpu_count() or 4)
    raw_dir = Path(raw_dir or ROOT / "gpu" / "results_recon" / "raw")
    if not any(raw_dir.glob("*.jpg")):
        # an output from before the GPU made FLUX only (28.10): its frames were
        # upscaled on Kaggle; brought back to FLUX's size, they go through the
        # whole CPU chain like new ones
        rp = raw_dir.parent / "report.json"
        ids = [i["id"] for i in json.loads(rp.read_text(encoding="utf-8")).get("images", [])] if rp.exists() else []
        raw_dir.mkdir(parents=True, exist_ok=True)
        for i in ids:
            p = raw_dir.parent / "up" / f"{i}.jpg"
            if p.exists():
                Image.open(p).convert("RGB").resize((1360, 768), Image.LANCZOS).save(raw_dir / f"{i}.jpg", quality=97)
    out = ROOT / "work" / slug / "layers_cpu"
    up, lay, tmp = out / "up", out / "layers", out / "photos"
    for d in (up, lay, tmp):
        d.mkdir(parents=True, exist_ok=True)
    mine = [x for i, x in enumerate(items(slug, raw_dir)) if i % shards == shard]
    rep = {"shard": shard, "shards": shards, "cpus": os.cpu_count(), "frames": [], "errors": []}
    t0 = time.perf_counter()
    for fid, src, is_photo in mine:
        try:
            if is_photo:
                src = fetch_photo(src, tmp / f"{fid}.jpg")
            r = frame(fid, Path(src), up, lay, is_photo)
            rep["frames"].append(r)
            print(json.dumps({k: r[k] for k in ("id", "esrgan_s", "depth_s", "segment_s", "lama_s", "total_s",
                                                "segmentation")}), flush=True)
        except Exception as e:                                       # noqa: BLE001
            rep["errors"].append(f"{fid}: {type(e).__name__}: {str(e)[:200]}")
            print("FAILED", fid, e, flush=True)
    rep["wall_s"] = round(time.perf_counter() - t0, 1)
    (out / f"report_{shard}.json").write_text(json.dumps(rep, indent=1), encoding="utf-8")
    return 0 if not rep["errors"] else 1


def merge_reports(slug):
    """One report for the plan and the forecast: per-frame CPU time on a runner."""
    out = ROOT / "work" / slug / "layers_cpu"
    reps = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(out.glob("report_*.json"))]
    frames = [f for r in reps for f in r["frames"]]
    gen = [f for f in frames if not f["photo"]]

    def mean(rows, k):
        return round(sum(f[k] for f in rows) / len(rows), 1) if rows else None
    summary = {"shards": len(reps), "cpus_per_runner": reps[0]["cpus"] if reps else None,
               "frames": len(frames), "flux_frames": len(gen), "photos": len(frames) - len(gen),
               "errors": [e for r in reps for e in r["errors"]],
               "per_flux_frame_s": {k: mean(gen, k) for k in ("esrgan_s", "depth_s", "segment_s", "lama_s", "total_s")},
               "per_frame_all_s": {k: mean(frames, k) for k in ("esrgan_s", "depth_s", "segment_s", "lama_s", "total_s")},
               "slowest_shard_wall_s": max((r["wall_s"] for r in reps), default=None),
               "layers": frames}
    return summary


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4] if len(sys.argv) > 4 else None))
