"""Reconstructions and 2.5D layers on a Kaggle GPU, in one session.

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

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "kaggle"))
BUILD = ROOT / "kaggle" / "build_recon"
RESULTS = ROOT / "kaggle" / "results_recon"
SLUG = "prehistoric-recon"

SOURCE = r'''
import json, os, sys, time, gc, pathlib, subprocess, urllib.request
OUT = pathlib.Path("/kaggle/working"); OUT.mkdir(parents=True, exist_ok=True)
T0 = time.time()
report = {"images": [], "errors": [], "stages": {}}
def stage(name, t): report["stages"][name] = round(time.time() - t, 1); print(f"== {name}: {report['stages'][name]} s", flush=True)

subprocess.run("pip install -q -U diffusers torchao transformers accelerate sentencepiece protobuf "
               "huggingface_hub bitsandbytes psutil spandrel", shell=True)
import numpy as np, torch, cv2
from PIL import Image
DTYPE = torch.float16
print(torch.__version__, torch.cuda.get_device_name(0))
GEN = OUT / "gen"; GEN.mkdir(exist_ok=True)

# ---------------------------------------------------------------- 1. FLUX
t = time.time()
jobs = []                                  # (id, prompt, seed)
for k, p in JOB["prompts"].items():
    for s in range(2 if k in JOB.get("two_seeds", []) else 1):
        jobs.append((f"{k}__s{s}", f"{p}, {JOB['style']}", 1000 + s * 17 + len(jobs)))
for k, p in JOB["thumbs"].items():
    for s in range(2):
        jobs.append((f"{k}__s{s}", f"{p}, {JOB['style']}", 5000 + s * 31 + len(jobs)))
try:
    from diffusers import FluxPipeline, FluxTransformer2DModel
    from diffusers import BitsAndBytesConfig as DiffBnb
    from transformers import BitsAndBytesConfig as TfBnb, T5EncoderModel
    MIRROR = pathlib.Path("/kaggle/input/whatif-flux-weights/flux4bit")
    REPO = str(MIRROR) if MIRROR.exists() else "black-forest-labs/FLUX.1-schnell"
    report["weights_source"] = "kaggle-mirror" if MIRROR.exists() else "huggingface"
    te2 = T5EncoderModel.from_pretrained(REPO, subfolder="text_encoder_2", torch_dtype=DTYPE,
            quantization_config=TfBnb(load_in_4bit=True, bnb_4bit_compute_dtype=DTYPE))
    tr = FluxTransformer2DModel.from_pretrained(REPO, subfolder="transformer", torch_dtype=DTYPE,
            quantization_config=DiffBnb(load_in_4bit=True, bnb_4bit_compute_dtype=DTYPE))
    pipe = FluxPipeline.from_pretrained(REPO, text_encoder_2=te2, transformer=tr, torch_dtype=DTYPE)
    pipe.enable_model_cpu_offload()
    for i, (name, prompt, seed) in enumerate(jobs):
        t1 = time.time()
        im = pipe(prompt, num_inference_steps=4, guidance_scale=0.0, width=1360, height=768,
                  max_sequence_length=256,
                  generator=torch.Generator("cpu").manual_seed(seed)).images[0]
        im.save(GEN / f"{name}.png")
        report["images"].append({"id": name, "seed": seed, "prompt": prompt, "s": round(time.time() - t1, 1)})
        print(f"  [{i+1}/{len(jobs)}] {name} {time.time()-t1:.1f}s", flush=True)
    del pipe, tr, te2; gc.collect(); torch.cuda.empty_cache()
except Exception as e:
    report["errors"].append(f"flux: {type(e).__name__}: {e}"); print("FLUX FAILED", e)
stage("flux", t)

# ---------------------------------------------------------- photos (real)
PHOTOS = OUT / "photos"; PHOTOS.mkdir(exist_ok=True)
for pid, ph in JOB.get("photos", {}).items():
    try:
        req = urllib.request.Request(ph["url"], headers={"User-Agent": "prehistoric-channel/0.1 (research)"})
        (PHOTOS / f"{pid}.jpg").write_bytes(urllib.request.urlopen(req, timeout=120).read())
    except Exception as e:
        report["errors"].append(f"photo {pid}: {e}")

W, H = 4224, 2376
def cover(im, w, h):
    s = max(w / im.width, h / im.height)
    im = im.resize((round(im.width * s), round(im.height * s)), Image.LANCZOS)
    l, tp = (im.width - w) // 2, (im.height - h) // 2
    return im.crop((l, tp, l + w, tp + h))

# --------------------------------------------------------- 2. Real-ESRGAN
t = time.time()
UP = OUT / "up"; UP.mkdir(exist_ok=True)
try:
    from spandrel import ModelLoader
    wpath = "/kaggle/working/RealESRGAN_x4plus.pth"
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
    del sr; gc.collect(); torch.cuda.empty_cache()
except Exception as e:
    report["errors"].append(f"esrgan: {type(e).__name__}: {e}"); print("ESRGAN FAILED", e)
for p in PHOTOS.glob("*.jpg"):
    try:
        cover(Image.open(p).convert("RGB"), W, H).save(UP / f"{p.stem}.jpg", quality=94)
    except Exception as e:
        report["errors"].append(f"photo resize {p.stem}: {e}")
stage("esrgan", t)

# ------------------------------------------- 3. depth, 4. mask and LaMa
t = time.time()
LAY = OUT / "layers"; LAY.mkdir(exist_ok=True)
try:
    from transformers import pipeline as hf_pipeline
    depth_pipe = hf_pipeline("depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf", device=0)
    lama_path = "/kaggle/working/big-lama.pt"
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
        u8 = (d * 255).astype(np.uint8)
        thr, _ = cv2.threshold(cv2.GaussianBlur(u8, (5, 5), 0), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        fg = (u8 > thr).astype(np.uint8)
        area = fg.mean()
        if area > 0.55:                    # mostly near: keep only the nearest part
            fg = (u8 > np.percentile(u8, 60)).astype(np.uint8)
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        fg = cv2.morphologyEx(cv2.morphologyEx(fg, cv2.MORPH_OPEN, k), cv2.MORPH_CLOSE, k)
        n, lab, stats, _ = cv2.connectedComponentsWithStats(fg)
        keep = np.zeros_like(fg)
        for i in range(1, n):
            if stats[i, cv2.CC_STAT_AREA] >= 0.01 * SW * SH:
                keep[lab == i] = 1
        fg = keep
        alpha = cv2.GaussianBlur(fg.astype(np.float32), (0, 0), 1.6)
        hole = cv2.dilate(fg, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * DILATE + 1, 2 * DILATE + 1)))
        info = {"id": p.stem, "fg_area": round(float(fg.mean()), 3), "hole_area": round(float(hole.mean()), 3),
                "otsu": int(thr), "dilate_px_at_1408": DILATE}
        if fg.mean() > 0.002:
            x = torch.from_numpy(np.asarray(small)).permute(2, 0, 1).float().div(255)[None].cuda()
            m = torch.from_numpy(hole).float()[None, None].cuda()
            with torch.no_grad():
                filled = lama(x, m)[0].clamp(0, 1).permute(1, 2, 0).cpu().numpy()
            filled = cv2.resize((filled * 255).astype(np.uint8), (W, H), interpolation=cv2.INTER_CUBIC)
            full = np.asarray(im).copy()
            hole_big = cv2.resize(cv2.GaussianBlur(hole.astype(np.float32), (0, 0), 2), (W, H))[..., None]
            bg = (full * (1 - hole_big) + filled * hole_big).astype(np.uint8)
            Image.fromarray(bg).save(LAY / f"{p.stem}_bg.jpg", quality=93)
        a_big = cv2.resize(alpha, (W, H), interpolation=cv2.INTER_LINEAR)
        cv2.imwrite(str(LAY / f"{p.stem}_alpha.png"), (a_big * 255).astype(np.uint8))
        report.setdefault("layers", []).append(info)
        print("  layers", info, flush=True)
except Exception as e:
    import traceback; traceback.print_exc()
    report["errors"].append(f"layers: {type(e).__name__}: {e}")
stage("layers", t)
report["total_s"] = round(time.time() - T0, 1)
(OUT / "report.json").write_text(json.dumps(report, indent=1))
for f in GEN.glob("*.png"): f.unlink()          # the upscaled jpgs are what the edit uses
print(json.dumps({k: v for k, v in report.items() if k != "images"}, indent=1))
'''


def notebook(job):
    import push
    job = dict(job)
    token = os.environ.get("HF_TOKEN") or push._env_token()
    if token:
        job["hf_token"] = token          # only if the mirror is missing; never written to the repo
    cell = f"JOB = {json.dumps(job, indent=1)}\n{SOURCE}"
    return {"cells": [{"cell_type": "code", "metadata": {}, "execution_count": None,
                       "outputs": [], "source": cell.splitlines(keepends=True)}],
            "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                        "name": "python3"}},
            "nbformat": 4, "nbformat_minor": 5}


def main():
    import push
    spec = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    job = {k: spec[k] for k in ("style", "seeds", "prompts", "thumbs", "photos")}
    job["two_seeds"] = spec.get("two_seeds", [])
    BUILD.mkdir(parents=True, exist_ok=True)
    (BUILD / "notebook.ipynb").write_text(json.dumps(notebook(job), indent=1), encoding="utf-8")
    meta = push._metadata(push.username(), SLUG, SLUG)
    (BUILD / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    ref = push.submit_when_idle(BUILD, job_hours=0.75)
    ok = push.wait(ref, poll_s=60, timeout_s=4 * 3600)
    push.publish_ledger()
    push.fetch(ref, RESULTS)
    (BUILD / "notebook.ipynb").unlink(missing_ok=True)      # carries the token
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
