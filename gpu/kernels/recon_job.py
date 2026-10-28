"""Reconstructions and 2.5D layers: the GPU session's code (run on Modal by gpu/modal_app.py;
written for Kaggle, where it first ran).

    python kaggle/recon_job.py episodes/living-fossils/recon.json

One session, because Kaggle bills by session time:
  1. FLUX.1-schnell (Apache-2.0) from the What If weight mirror, 4-bit, as in
     Closed Tabs' job_template - every reconstruction, two seeds each, and the
     thumbnail bases;
  2. Real-ESRGAN x4plus (BSD-3) through spandrel, tiled in fp16 - 1360x768
     becomes 5440x3072, saved at 4224x2376: 3840x2160 plus a 10 % margin for
     the camera to travel over;
  3. Depth Anything V2 Small (Apache-2.0) - a depth map per frame;
  4. a foreground mask from the depth (Otsu, cleaned, soft edge), and
     LaMa (Apache-2.0, the TorchScript big-lama) filling the background behind
     it, so the parallax move reveals painted background instead of stretched
     pixels.

Real photographs (PD / CC BY, from Commons) skip step 1 and 2 and go through 3-4.
Outputs: <id>.jpg, <id>_bg.jpg, <id>_alpha.png, <id>_depth.png, report.json.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
SLUG = "prehistoric-recon"

SOURCE = r'''

# ---- the session's own deadline (as in the shared kaggle/job_template.py) ----
import threading as _th, os as _os, json as _js, time as _tm, pathlib as _pl, sys as _sys
_OUT = _pl.Path("/kaggle/working"); _OUT.mkdir(parents=True, exist_ok=True)
_BUDGET_S = float(JOB.get("budget_s") or 300 + 105 * (JOB.get("seeds_per_prompt", 1) * len(JOB['prompts']) + len(JOB.get('two_seeds', [])) + 2 * len(JOB['thumbs']) + 2 * len(JOB.get('photos', {}))))
_STAGE = {"name": "starting", "since": _tm.time(), "began": _tm.time()}
def _mark(name):
    _STAGE["name"], _STAGE["since"] = name, _tm.time()
    print(f"-- {name} (at {_tm.time() - _STAGE['began']:.0f}s)", flush=True)
def _give_up():
    ran = _tm.time() - _STAGE["began"]
    rep = {"errors": [f"timeout: {ran:.0f}s over a {2 * _BUDGET_S:.0f}s deadline, stuck in {_STAGE['name']}"],
           "timed_out": {"ran_s": round(ran, 1), "stage": _STAGE["name"]}}
    try:
        (_OUT / "report.json").write_text(_js.dumps(rep, indent=2))
    except Exception:
        pass
    print("==== GAVE UP ====", rep, flush=True)
    _os._exit(0)
_WD = _th.Timer(2 * _BUDGET_S, _give_up); _WD.daemon = True; _WD.start()
def _need_gpu():
    import torch
    if not torch.cuda.is_available():
        (_OUT / "report.json").write_text(_js.dumps({"errors": ["no GPU in this session"]}))
        print("no GPU in this session; stopping before loading anything", flush=True)
        _os._exit(0)
import json, os, sys, time, gc, pathlib, subprocess, urllib.request
OUT = pathlib.Path("/kaggle/working"); OUT.mkdir(parents=True, exist_ok=True)
T0 = time.time()
report = {"images": [], "errors": [], "stages": {}}
def stage(name, t): report["stages"][name] = round(time.time() - t, 1); print(f"== {name}: {report['stages'][name]} s", flush=True)

# 28.10: "-U" plus rembg pulled numpy 2.5 over Kaggle's own build, and numpy,
# scipy and diffusers stopped importing (no image made). The packages Kaggle
# compiled together stay at the versions it shipped: they go in as constraints.
import importlib.metadata as _md
_pins = []
for _pkg in ("numpy", "scipy", "scikit-image", "numba", "pillow", "opencv-python", "opencv-python-headless"):
    try:
        _pins.append(f"{_pkg}=={_md.version(_pkg)}")
    except _md.PackageNotFoundError:
        pass
pathlib.Path("/tmp/pins.txt").write_text("\n".join(_pins) + "\n")
print("pinned:", _pins, flush=True)
subprocess.run("pip install -q -U -c /tmp/pins.txt diffusers torchao transformers accelerate sentencepiece protobuf "
               "huggingface_hub bitsandbytes psutil spandrel timm kornia einops", shell=True)
subprocess.run("pip install -q -c /tmp/pins.txt rembg onnxruntime", shell=True)
_mark("install done"); _need_gpu()
import numpy as np, torch, cv2
from PIL import Image
DTYPE = torch.float16
print(torch.__version__, torch.cuda.get_device_name(0))
GEN = OUT / "gen"; GEN.mkdir(exist_ok=True)

# ---------------------------------------------------------------- 1. FLUX
_mark("flux"); t = time.time()
jobs = []                                  # (id, prompt, seed)
# every reconstruction in several seeds: the species check (pipeline/speciescheck.py)
# takes the first one that is anatomically right, and drops the scene if none is
for k, p in JOB["prompts"].items():
    for s in range(max(JOB.get("seeds_per_prompt", 1), 2 if k in JOB.get("two_seeds", []) else 1)):
        jobs.append((f"{k}__s{s}", f"{p}, {JOB['style']}", 1000 + s * 17 + len(jobs)))
for k, p in JOB["thumbs"].items():
    for s in range(2):
        jobs.append((f"{k}__s{s}", f"{p}, {JOB['style']}", 5000 + s * 31 + len(jobs)))
try:
    from diffusers import FluxPipeline, FluxTransformer2DModel
    from diffusers import BitsAndBytesConfig as DiffBnb
    from transformers import BitsAndBytesConfig as TfBnb, T5EncoderModel
    # 25.10: the mirror was attached (kernel_sources) but not at the path this
    # code expected, the fallback hit Hugging Face's gate with no token, and no
    # image was made. So: show what is mounted, look for flux4bit anywhere under
    # /kaggle/input, then Hugging Face with the token injected at submit time,
    # and otherwise stop with a clear error instead of a quiet fallback.
    import glob as _glob
    _inp = pathlib.Path("/kaggle/input")
    print("== /kaggle/input (2 levels):", flush=True)
    for _p in sorted(_inp.glob("*")) + sorted(_inp.glob("*/*")):
        print("   ", _p, flush=True)
    _found = [pathlib.Path(x) for x in _glob.glob("/kaggle/input/**/flux4bit", recursive=True)
              if pathlib.Path(x, "transformer").exists()]
    report["mirror_candidates"] = [str(x) for x in _found]
    if _found:
        REPO = str(_found[0])
        report["weights_source"] = f"kaggle-mirror:{REPO}"
    elif JOB.get("hf_token"):
        os.environ["HF_TOKEN"] = JOB["hf_token"]
        os.environ["HUGGING_FACE_HUB_TOKEN"] = JOB["hf_token"]
        REPO = "black-forest-labs/FLUX.1-schnell"
        report["weights_source"] = "huggingface (token)"
    else:
        raise RuntimeError("no FLUX weights: mirror flux4bit not found under /kaggle/input and no HF token")
    print("FLUX weights from:", report["weights_source"], flush=True)
    # 28.10: the mirror holds the transformer, VAE, tokenizers and scheduler but
    # no text encoders - when it was made (12.10) Hugging Face refused both with
    # a 401 (its manifest.json says so). A part the mirror lacks comes from
    # Hugging Face: the gated original with the token, else the ungated 4-bit
    # copy of the same Apache-2.0 weights (magespace/FLUX.1-schnell-bnb-nf4).
    from transformers import CLIPTextModel
    if JOB.get("hf_token"):
        os.environ["HF_TOKEN"] = JOB["hf_token"]
        os.environ["HUGGING_FACE_HUB_TOKEN"] = JOB["hf_token"]
    NF4 = "magespace/FLUX.1-schnell-bnb-nf4"

    def _sources(sub):
        here = pathlib.Path(REPO, sub)
        if here.is_dir() and (any(here.glob("*.safetensors")) or any(here.glob("*.bin"))):
            return [REPO]
        return (["black-forest-labs/FLUX.1-schnell"] if JOB.get("hf_token") else []) + [NF4]

    def _load(sub, fn):
        errs = []
        for src in _sources(sub):
            try:
                m = fn(src)
                report.setdefault("parts_from", {})[sub] = src
                print(f"  {sub} from {src}", flush=True)
                return m
            except Exception as _e:
                errs.append(f"{src}: {type(_e).__name__}: {str(_e)[:160]}")
        raise RuntimeError(f"{sub}: " + " | ".join(errs))

    def _t5(src):
        if src == NF4:
            # already 4-bit; saved for bfloat16, which a T4 cannot compute in
            m = T5EncoderModel.from_pretrained(src, subfolder="text_encoder_2", torch_dtype=DTYPE)
            import bitsandbytes as _bnb
            for _l in m.modules():
                if isinstance(_l, _bnb.nn.Linear4bit):
                    _l.compute_dtype = DTYPE
            return m
        return T5EncoderModel.from_pretrained(src, subfolder="text_encoder_2", torch_dtype=DTYPE,
                quantization_config=TfBnb(load_in_4bit=True, bnb_4bit_compute_dtype=DTYPE))

    te2 = _load("text_encoder_2", _t5)
    te1 = _load("text_encoder", lambda src: CLIPTextModel.from_pretrained(src, subfolder="text_encoder",
                                                                          torch_dtype=DTYPE))
    tr = FluxTransformer2DModel.from_pretrained(REPO, subfolder="transformer", torch_dtype=DTYPE,
            quantization_config=DiffBnb(load_in_4bit=True, bnb_4bit_compute_dtype=DTYPE))
    if pathlib.Path(REPO, "model_index.json").exists() or not pathlib.Path(REPO).is_dir():
        pipe = FluxPipeline.from_pretrained(REPO, text_encoder=te1, text_encoder_2=te2, transformer=tr,
                                            torch_dtype=DTYPE)
    else:
        # the mirror has no model_index.json, so the pipeline is put together by hand
        from diffusers import AutoencoderKL, FlowMatchEulerDiscreteScheduler
        from transformers import AutoTokenizer
        pipe = FluxPipeline(
            scheduler=FlowMatchEulerDiscreteScheduler.from_pretrained(REPO, subfolder="scheduler"),
            vae=AutoencoderKL.from_pretrained(REPO, subfolder="vae", torch_dtype=DTYPE),
            text_encoder=te1, tokenizer=AutoTokenizer.from_pretrained(REPO, subfolder="tokenizer"),
            text_encoder_2=te2, tokenizer_2=AutoTokenizer.from_pretrained(REPO, subfolder="tokenizer_2"),
            transformer=tr)
    pipe.enable_model_cpu_offload()
    for i, (name, prompt, seed) in enumerate(jobs):
        t1 = time.time()
        im = pipe(prompt, num_inference_steps=4, guidance_scale=0.0, width=1360, height=768,
                  max_sequence_length=256,
                  generator=torch.Generator("cpu").manual_seed(seed)).images[0]
        im.save(GEN / f"{name}.png")
        report["images"].append({"id": name, "seed": seed, "prompt": prompt, "s": round(time.time() - t1, 1)})
        print(f"  [{i+1}/{len(jobs)}] {name} {time.time()-t1:.1f}s", flush=True)
    del pipe, tr, te2, te1; gc.collect(); torch.cuda.empty_cache()
except Exception as e:
    report["errors"].append(f"flux: {type(e).__name__}: {e}"); print("FLUX FAILED", e)
stage("flux", t)

# ---------------------------------------------------------- photos (real)
PHOTOS = OUT / "photos"; PHOTOS.mkdir(exist_ok=True)
# Commons answers a burst of downloads with 429: back off and retry. A photo
# that still fails is reported apart - the species check simply has one
# candidate fewer - instead of failing the whole run.
for pid, ph in JOB.get("photos", {}).items():
    for _try in range(5):
        try:
            req = urllib.request.Request(ph["url"], headers={"User-Agent": "prehistoric-channel/0.1 (research)"})
            (PHOTOS / f"{pid}.jpg").write_bytes(urllib.request.urlopen(req, timeout=120).read())
            break
        except Exception as e:
            if _try == 4:
                report.setdefault("photo_errors", []).append(f"photo {pid}: {e}")
            time.sleep(8 * (_try + 1))
    time.sleep(1.5)

W, H = 4224, 2376
def cover(im, w, h):
    s = max(w / im.width, h / im.height)
    im = im.resize((round(im.width * s), round(im.height * s)), Image.LANCZOS)
    l, tp = (im.width - w) // 2, (im.height - h) // 2
    return im.crop((l, tp, l + w, tp + h))

# --------------------------------------------------------- 2. Real-ESRGAN
_mark("esrgan"); t = time.time()
UP = OUT / "up"; UP.mkdir(exist_ok=True)
try:
    from spandrel import ModelLoader
    wpath = "/tmp/RealESRGAN_x4plus.pth"   # not in /kaggle/working: it would be served back
    urllib.request.urlretrieve("https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth", wpath)
    sr = ModelLoader().load_from_file(wpath).cuda().eval().half()
    def upscale(im, tile=384, pad=24):
        x = torch.from_numpy(np.asarray(im.convert("RGB"))).permute(2, 0, 1).float().div(255)[None]
        _, _, h, w = x.shape
        out = torch.zeros(1, 3, h * 4, w * 4)
        for y0 in range(0, h, tile):
            for x0 in range(0, w, tile):
                ya, yb = max(0, y0 - pad), min(h, y0 + tile + pad)
                xa, xb = max(0, x0 - pad), min(w, x0 + tile + pad)
                with torch.no_grad():
                    r = sr(x[:, :, ya:yb, xa:xb].cuda().half()).float().cpu()
                oy, ox = (y0 - ya) * 4, (x0 - xa) * 4
                th, tw = min(tile, h - y0) * 4, min(tile, w - x0) * 4
                out[:, :, y0 * 4:y0 * 4 + th, x0 * 4:x0 * 4 + tw] = r[:, :, oy:oy + th, ox:ox + tw]
        return Image.fromarray((out[0].clamp(0, 1).permute(1, 2, 0).numpy() * 255).round().astype(np.uint8))
    for p in sorted(GEN.glob("*.png")):
        big = upscale(Image.open(p))
        if p.stem.startswith("thumb_"):
            big.resize((2560, 1440), Image.LANCZOS).save(UP / f"{p.stem}.jpg", quality=95)
        else:
            cover(big, W, H).save(UP / f"{p.stem}.jpg", quality=94)
        print("  up", p.stem, flush=True)
    # real photos smaller than the frame are upscaled the same way (the live
    # coelacanth photograph is 963 px wide); big ones are only resized
    for p in sorted(PHOTOS.glob("*.jpg")):
        try:
            im = Image.open(p).convert("RGB")
            if im.width < 2400:
                im.thumbnail((1400, 1400))
                im = upscale(im)
            cover(im, W, H).save(UP / f"{p.stem}.jpg", quality=94)
            print("  photo", p.stem, flush=True)
        except Exception as e:
            report.setdefault("photo_errors", []).append(f"photo {p.stem}: {type(e).__name__}: {e}")
    del sr; gc.collect(); torch.cuda.empty_cache()
except Exception as e:
    report["errors"].append(f"esrgan: {type(e).__name__}: {e}"); print("ESRGAN FAILED", e)
stage("esrgan", t)

# ------------------------------------------- segmentation of the subject
_SEG = {}


def seg_birefnet(img):
    if "birefnet" not in _SEG:
        from transformers import AutoModelForImageSegmentation
        _SEG["birefnet"] = AutoModelForImageSegmentation.from_pretrained(
            "ZhengPeng7/BiRefNet", trust_remote_code=True).to("cuda").eval().half()
    x = np.asarray(img.resize((1024, 1024), Image.BILINEAR), np.float32) / 255
    x = (x - np.array([0.485, 0.456, 0.406], np.float32)) / np.array([0.229, 0.224, 0.225], np.float32)
    t_ = torch.from_numpy(x).permute(2, 0, 1)[None].cuda().half()
    with torch.no_grad():
        pr = _SEG["birefnet"](t_)[-1].sigmoid().float().cpu().numpy()[0, 0]
    return cv2.resize(pr, img.size, interpolation=cv2.INTER_LINEAR)


def seg_isnet(img):
    if "isnet" not in _SEG:
        from rembg import new_session
        _SEG["isnet"] = new_session("isnet-general-use", providers=["CPUExecutionProvider"])
    from rembg import remove
    return np.asarray(remove(img, session=_SEG["isnet"], only_mask=True), np.float32) / 255


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


# ------------------------------------------- 3. depth, 4. mask and LaMa
_mark("layers"); t = time.time()
LAY = OUT / "layers"; LAY.mkdir(exist_ok=True)
try:
    from transformers import pipeline as hf_pipeline
    depth_pipe = hf_pipeline("depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf", device=0)
    lama_path = "/tmp/big-lama.pt"
    urllib.request.urlretrieve("https://github.com/enesmsahin/simple-lama-inpainting/releases/download/v0.1.0/big-lama.pt", lama_path)
    lama = torch.jit.load(lama_path, map_location="cuda").eval()
    SW, SH = 1408, 792                     # working scale for depth, mask and fill
    DILATE = JOB.get("dilate_px", 22)      # at SW: covers the largest parallax offset
    for p in sorted(UP.glob("*.jpg")):
        if p.stem.startswith("thumb_"):
            continue
        im = Image.open(p).convert("RGB")
        small = im.resize((SW, SH), Image.LANCZOS)
        r = depth_pipe(small)
        d = r["predicted_depth"]
        d = d.squeeze().float().cpu().numpy() if torch.is_tensor(d) else np.asarray(r["depth"], np.float32)
        d = cv2.resize(d, (SW, SH), interpolation=cv2.INTER_CUBIC)
        d = (d - d.min()) / max(1e-6, d.max() - d.min())          # 1 = near
        cv2.imwrite(str(LAY / f"{p.stem}_depth.png"), (d * 65535).astype(np.uint16))
        # The foreground is the animal itself, from a segmentation model, not a
        # depth threshold: 25.10 the depth cut took the corner of a fossil slab
        # and left the fish on the background. BiRefNet (MIT) runs on the GPU;
        # IS-Net (rembg, Apache-2.0) runs on the CPU beside it as a fallback and
        # a second opinion. A mask has to pass the checks below, or the shot
        # gets no foreground layer and the render gives it a plain Ken Burns.
        probs = {}
        try:
            probs["birefnet"] = seg_birefnet(small)
        except Exception as _e:
            report.setdefault("seg_errors", []).append(f"birefnet: {type(_e).__name__}: {str(_e)[:200]}")
        try:
            probs["isnet"] = seg_isnet(small)
        except Exception as _e:
            report.setdefault("seg_errors", []).append(f"isnet: {type(_e).__name__}: {str(_e)[:200]}")
        checks = {k: mask_checks(v) for k, v in probs.items()}
        if len(probs) == 2:
            _a, _b = (probs["birefnet"] > 0.5), (probs["isnet"] > 0.5)
            iou = float((_a & _b).sum() / max(1, (_a | _b).sum()))
        else:
            iou = None
        chosen = next((k for k in ("birefnet", "isnet") if k in checks and checks[k]["ok"]), None)
        if chosen and iou is not None and iou < 0.5:
            chosen = None          # the two models disagree about what the subject is
        if chosen:
            fg = checks[chosen]["mask"]
            prob = probs[chosen] * fg
        else:
            fg = np.zeros((SH, SW), np.uint8)
            prob = fg.astype(np.float32)
        thr = 0
        alpha = cv2.GaussianBlur(np.clip(prob, 0, 1).astype(np.float32), (0, 0), 1.2)
        hole = cv2.dilate(fg, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * DILATE + 1, 2 * DILATE + 1)))
        info = {"id": p.stem, "fg_area": round(float(fg.mean()), 3), "hole_area": round(float(hole.mean()), 3),
                "segmentation": chosen or "none - Ken Burns", "iou_birefnet_isnet": None if iou is None else round(iou, 3),
                "checks": {k: {kk: vv for kk, vv in c.items() if kk != "mask"} for k, c in checks.items()},
                "dilate_px_at_1408": DILATE}
        if chosen and fg.mean() > 0.002:
            x = torch.from_numpy(np.asarray(small)).permute(2, 0, 1).float().div(255)[None].cuda()
            m = torch.from_numpy(hole).float()[None, None].cuda()
            with torch.no_grad():
                filled = lama(x, m)[0].clamp(0, 1).permute(1, 2, 0).cpu().numpy()
            filled = cv2.resize((filled * 255).astype(np.uint8), (W, H), interpolation=cv2.INTER_CUBIC)
            full = np.asarray(im).copy()
            hole_big = cv2.resize(cv2.GaussianBlur(hole.astype(np.float32), (0, 0), 2), (W, H))[..., None]
            bg = (full * (1 - hole_big) + filled * hole_big).astype(np.uint8)
            Image.fromarray(bg).save(LAY / f"{p.stem}_bg.jpg", quality=93)
        if chosen:
            a_big = cv2.resize(alpha, (W, H), interpolation=cv2.INTER_LINEAR)
            cv2.imwrite(str(LAY / f"{p.stem}_alpha.png"), (a_big * 255).astype(np.uint8))
        report.setdefault("layers", []).append(info)
        print("  layers", info, flush=True)
except Exception as e:
    import traceback; traceback.print_exc()
    report["errors"].append(f"layers: {type(e).__name__}: {e}")
stage("layers", t)
_WD.cancel()
report["total_s"] = round(time.time() - T0, 1)
(OUT / "report.json").write_text(json.dumps(report, indent=1))
for f in GEN.glob("*.png"): f.unlink()          # the upscaled jpgs are what the edit uses
print(json.dumps({k: v for k, v in report.items() if k != "images"}, indent=1))
'''


FLUX_TAIL = r'''
# ----------------------------------------- FLUX only: the frames as they came
RAW = OUT / "raw"; RAW.mkdir(exist_ok=True)
for p in sorted(GEN.glob("*.png")):
    Image.open(p).convert("RGB").save(RAW / f"{p.stem}.jpg", quality=97)
    p.unlink()
_WD.cancel()
report["flux_only"] = True
report["total_s"] = round(time.time() - T0, 1)
(OUT / "report.json").write_text(json.dumps(report, indent=1))
print(json.dumps({k: v for k, v in report.items() if k != "images"}, indent=1))
'''


def flux_source():
    """The kernel cut down to what needs a GPU: FLUX and nothing else (28.10).
    Real-ESRGAN, depth, the masks and LaMa run on GitHub's CPUs afterwards
    (pipeline/layers_cpu.py), so the shared Kaggle hours go to FLUX alone."""
    s = SOURCE[:SOURCE.index("# ---------------------------------------------------------- photos (real)")]
    s = s.replace('subprocess.run("pip install -q -c /tmp/pins.txt rembg onnxruntime", shell=True)\n', "")
    s = s.replace(" spandrel timm kornia einops", "")
    return s + FLUX_TAIL


def main():
    raise SystemExit("This channel no longer uses Kaggle: the GPU work runs on Modal "
                     "(python -m modal run gpu/modal_app.py --task recon|voice).")


if __name__ == "__main__":
    sys.exit(main())
