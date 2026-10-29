"""LTX-Video image-to-video on a Kaggle GPU: the FLUX frames that move.

The frames come from this channel's FLUX kernel (prehistoric-recon), attached
as a kernel source; JOB names which ones (those the species check passed, the
8-10 strongest: automation/build_doc.py) and a motion for each. Every clip is
97 frames at 768x448 (4 s at 24 fps); the runners upscale it afterwards
(pipeline/layers_cpu.py). A clip that comes out black or broken is dropped -
its shot stays a still with a 2.5D move.

Model: LTX-Video 2B (Lightricks, diffusers weights of Lightricks/LTX-Video, the
0.9.x releases under the LTX-Video OpenRAIL-M style licence: outputs may be
used commercially, its use restrictions apply). Recorded in LICENSES.md.
"""

SLUG = "prehistoric-animate"

SOURCE = r'''
import subprocess, sys, json, pathlib, time, glob, importlib.metadata as _md
_pins = []
for _pkg in ("numpy", "scipy", "pillow", "opencv-python", "opencv-python-headless"):
    try:
        _pins.append(f"{_pkg}=={_md.version(_pkg)}")
    except _md.PackageNotFoundError:
        pass
pathlib.Path("/tmp/pins.txt").write_text("\n".join(_pins) + "\n")
subprocess.run("pip install -q -U -c /tmp/pins.txt diffusers transformers accelerate sentencepiece protobuf "
               "imageio imageio-ffmpeg", shell=True)
_mark("install done"); _need_gpu()
import numpy as np, torch
from PIL import Image
from diffusers import LTXImageToVideoPipeline
from diffusers.utils import export_to_video

OUT = pathlib.Path("/kaggle/working"); CLIPS = OUT / "clips"; CLIPS.mkdir(parents=True, exist_ok=True)
report = {"clips": [], "errors": [], "gpu": torch.cuda.get_device_name(0)}
raw = {pathlib.Path(p).stem: p for p in glob.glob("/kaggle/input/**/raw/*.jpg", recursive=True)}
report["frames_found"] = len(raw)
t0 = time.time()
_mark("load ltx")
pipe = LTXImageToVideoPipeline.from_pretrained("Lightricks/LTX-Video", torch_dtype=torch.float16)
pipe.enable_model_cpu_offload()
pipe.vae.enable_tiling()
report["load_s"] = round(time.time() - t0, 1)
NEG = "worst quality, inconsistent motion, blurry, jittery, distorted, deformed anatomy, extra limbs, text, watermark"
W, H, N = 768, 448, 97
for i, f in enumerate(JOB["frames"]):
    fid = f["id"]
    if fid not in raw:
        report["errors"].append(f"{fid}: frame not found among the FLUX outputs"); continue
    _mark(f"clip {fid}")
    t1 = time.time()
    try:
        img = Image.open(raw[fid]).convert("RGB").resize((W, H), Image.LANCZOS)
        prompt = f"{f['motion']}. {f['prompt']}. Cinematic wildlife documentary shot, natural movement, steady camera"
        frames = pipe(image=img, prompt=prompt, negative_prompt=NEG, width=W, height=H, num_frames=N,
                      num_inference_steps=JOB.get("steps", 30), guidance_scale=3.0,
                      generator=torch.Generator("cpu").manual_seed(4000 + i)).frames[0]
        arr = np.stack([np.asarray(x, np.float32) for x in frames])
        bad = (not np.isfinite(arr).all()) or arr.mean() < 3 or arr.std() < 2
        if bad:
            report["clips"].append({"id": fid, "ok": False, "why": "black or broken output", "s": round(time.time() - t1, 1)})
            continue
        export_to_video(frames, str(CLIPS / f"{fid}.mp4"), fps=24)
        report["clips"].append({"id": fid, "ok": True, "s": round(time.time() - t1, 1), "frames": len(frames),
                                "w": W, "h": H, "duration": round(len(frames) / 24, 2)})
    except Exception as e:
        report["clips"].append({"id": fid, "ok": False, "why": f"{type(e).__name__}: {str(e)[:200]}",
                                "s": round(time.time() - t1, 1)})
    print(report["clips"][-1], flush=True)
    torch.cuda.empty_cache()
report["total_s"] = round(time.time() - t0, 1)
(OUT / "report.json").write_text(json.dumps(report, indent=1))
_WD.cancel()
print(json.dumps({k: v for k, v in report.items() if k != "clips"}), flush=True)
'''
