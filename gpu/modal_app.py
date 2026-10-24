"""All of this channel's GPU work, on Modal (Kaggle is no longer used here).

    modal run gpu/modal_app.py --task weights            # once: fetch 4-bit FLUX into the Volume
    modal run gpu/modal_app.py --task recon --slug living-fossils
    modal run gpu/modal_app.py --task voice --slug living-fossils

Run from GitHub Actions (.github/workflows/gpu-jobs.yml) with the repository's
MODAL_TOKEN_ID / MODAL_TOKEN_SECRET; nothing depends on anyone's computer.

Money first - the account starts with $1 and no card:
  - one GPU type, NVIDIA L4 ($0.80/h). A T4 is cheaper by the hour but cannot
    hold 4-bit FLUX without offloading to the CPU, which made it ~3x slower
    per image on Kaggle; the L4 holds it whole and costs less per image.
  - every model lives in the Volume "prehistoric-weights": FLUX, already in
    4-bit, is downloaded once on a CPU-only container (a few cents) and from
    then on only read. Real-ESRGAN, LaMa, Depth Anything,
    Chatterbox and Whisper are cached there on first use.
  - a hard timeout on every function, below what the dollar can pay for.
  - each call's seconds are measured and priced (modal_usage.json), so the
    report can say what the test cost and what an episode will.

The work itself is the same code the Kaggle kernels ran (gpu/kernels/recon_job.py
SOURCE, gpu/kernels/voice_job.py body): only the paths and the install lines change,
so there is one version of it to maintain.
"""
import json
import os
import re
import sys
import time
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parent.parent
GPU = "L4"
# Modal list prices, per second (checked 24.10.2026 on modal.com/pricing)
PRICE = {"L4": 0.000222, "cpu_core": 0.0000131, "mem_gib": 0.00000222}

app = modal.App("prehistoric-channel")
weights = modal.Volume.from_name("prehistoric-weights", create_if_missing=True)
outputs = modal.Volume.from_name("prehistoric-out", create_if_missing=True)
VOLS = {"/w": weights, "/out": outputs}

flux_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgl1", "libglib2.0-0")
    .pip_install("torch==2.5.1", "torchvision==0.20.1", "diffusers==0.32.2", "transformers==4.48.3",
                 "accelerate==1.3.0", "bitsandbytes==0.45.2", "sentencepiece", "protobuf",
                 "huggingface_hub[hf_transfer]==0.28.1", "spandrel==0.4.1", "opencv-python-headless==4.10.0.84",
                 "pillow", "numpy<2.2", "psutil")
    .env({"HF_HOME": "/w/hf", "HF_HUB_ENABLE_HF_TRANSFER": "1"})
)
voice_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg")
    .pip_install("torch==2.6.0", "torchvision==0.21.0", "torchaudio==2.6.0", "chatterbox-tts==0.1.2",
                 "openai-whisper", "pyloudnorm", "scipy", "soundfile")
    .env({"HF_HOME": "/w/hf"})
)

FLUX_4BIT = "/w/flux4bit"
# FLUX.1-schnell (Black Forest Labs, Apache-2.0) already quantised to 4-bit NF4
# for diffusers and published without a gate: no Hugging Face token, and no GPU
# time spent quantising. 13.4 GB, downloaded once into the Volume.
FLUX_REPO = "magespace/FLUX.1-schnell-bnb-nf4"


# ------------------------------------------------------------------ weights
@app.function(image=flux_image, cpu=2.0, memory=4096, timeout=40 * 60, volumes=VOLS)
def fetch_flux():
    """Download the 4-bit FLUX pipeline once, on a CPU-only container."""
    if Path(FLUX_4BIT, "transformer", "diffusion_pytorch_model.safetensors").exists():
        return "cached"
    from huggingface_hub import snapshot_download
    snapshot_download(FLUX_REPO, local_dir=FLUX_4BIT)
    weights.commit()
    return "downloaded"


# --------------------------------------------------------------- the work
@app.function(image=flux_image, gpu=GPU, memory=24576, timeout=45 * 60, volumes=VOLS)
def run_recon(source: str, job: dict, slug: str):
    t0 = time.time()
    out = Path("/out") / slug / "recon"
    out.mkdir(parents=True, exist_ok=True)
    Path("/w/models").mkdir(parents=True, exist_ok=True)
    g = {"JOB": job, "__name__": "__kernel__"}
    try:
        exec(compile(source, "recon_kernel", "exec"), g)
    finally:
        outputs.commit()
        weights.commit()
    return {"seconds": round(time.time() - t0, 1)}


@app.function(image=voice_image, gpu=GPU, memory=16384, timeout=25 * 60, volumes=VOLS)
def run_voice(source: str, job: dict, slug: str):
    t0 = time.time()
    (Path("/out") / slug / "voice").mkdir(parents=True, exist_ok=True)
    Path("/w/whisper").mkdir(parents=True, exist_ok=True)
    g = {"JOB": job, "__name__": "__kernel__"}
    try:
        exec(compile(source, "voice_kernel", "exec"), g)
    finally:
        outputs.commit()
        weights.commit()
    return {"seconds": round(time.time() - t0, 1)}


# ---------------------------------------------------- Kaggle code, re-pathed
def recon_source(slug):
    sys.path.insert(0, str(ROOT / "gpu" / "kernels"))
    import recon_job
    s = recon_job.SOURCE
    out = f"/out/{slug}/recon"
    s = s.replace('"/kaggle/working"', f'"{out}"')
    s = re.sub(r'subprocess\.run\("pip install[^)]*\)\n', "", s, flags=re.S)
    s = s.replace('pathlib.Path("/kaggle/input/whatif-flux-weights/flux4bit")', f'pathlib.Path("{FLUX_4BIT}")')
    s = s.replace('"kaggle-mirror"', '"modal-volume"')
    # an L4 computes in bfloat16, which is what the NF4 weights were saved for
    s = s.replace("DTYPE = torch.float16", "DTYPE = torch.bfloat16")
    s = re.sub(r'wpath = "/tmp/RealESRGAN_x4plus\.pth".*', 'wpath = "/w/models/RealESRGAN_x4plus.pth"', s)
    s = re.sub(r'lama_path = "/tmp/big-lama\.pt".*', 'lama_path = "/w/models/big-lama.pt"', s)
    s = re.sub(r'(\n\s*)urllib\.request\.urlretrieve\(([^,]+), (wpath|lama_path)\)',
               r'\1if not os.path.exists(\3): urllib.request.urlretrieve(\2, \3)', s)
    s = s.replace('for f in GEN.glob("*.png"): f.unlink()', 'pass  # Modal keeps the raw FLUX frames too')
    assert "pip install" not in s and "/kaggle/" not in s, "recon source still has Kaggle paths"
    return s


def voice_source(slug):
    sys.path.insert(0, str(ROOT / "gpu" / "kernels"))
    sys.path.insert(0, str(ROOT))
    import voice_job
    s = voice_job.WATCHDOG + "\n" + voice_job.body()
    out = f"/out/{slug}/voice"
    s = s.replace('"/kaggle/working"', f'"{out}"')
    s = re.sub(r'subprocess\.run\(\[sys\.executable, "-m", "pip", "install".*?check=True\)\n', "", s, flags=re.S)
    s = s.replace('whisper.load_model("base.en", device="cuda")',
                  'whisper.load_model("base.en", device="cuda", download_root="/w/whisper")')
    assert "pip" not in s.split("import numpy")[0] and "/kaggle/" not in s, "voice source still has Kaggle bits"
    return s


def voice_job_dict(slug):
    sys.path.insert(0, str(ROOT))
    from automation import build_doc
    from core import delivery, narrate
    script = (ROOT / "episodes" / slug / "script.txt").read_text(encoding="utf-8")
    timeline = build_doc.narration(slug)[1]["timeline"]
    keep = sum(1 for l in timeline if l["end"] <= 31.0)
    lex = narrate.lexicon()
    return {"lines": [{**l.__dict__, "text": narrate.respell(l.text, lex)}
                      for l in delivery.parse(script)[:keep]], "budget_s": 900}


def _price(seconds, gpu=True, cores=0.125, mem_gib=16):
    return round(seconds * ((PRICE["L4"] if gpu else 0) + cores * PRICE["cpu_core"] + mem_gib * PRICE["mem_gib"]), 4)


@app.local_entrypoint()
def main(task: str, slug: str = "living-fossils"):
    usage_path = ROOT / "state" / "modal_usage.json"
    usage = json.loads(usage_path.read_text()) if usage_path.exists() else []

    def log(name, t0, gpu, mem, cores=0.125, extra=None):
        s = round(time.time() - t0, 1)
        row = {"task": name, "slug": slug, "seconds": s, "gpu": GPU if gpu else None, "mem_gib": mem,
               "usd_est": _price(s, gpu, cores, mem), "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               **(extra or {})}
        usage.append(row)
        usage_path.parent.mkdir(exist_ok=True)
        usage_path.write_text(json.dumps(usage, indent=1))
        print(json.dumps(row), flush=True)

    if task == "weights":
        t0 = time.time()
        print(fetch_flux.remote())
        log("fetch_flux", t0, False, 4, cores=2.0)
    elif task == "recon":
        spec = json.loads((ROOT / "episodes" / slug / "recon.json").read_text(encoding="utf-8"))
        job = {k: spec[k] for k in ("style", "seeds", "prompts", "thumbs", "photos")}
        job["two_seeds"] = spec.get("two_seeds", [])
        t0 = time.time()
        r = run_recon.remote(recon_source(slug), job, slug)
        log("recon", t0, True, 24, extra={"inside_s": r["seconds"]})
    elif task == "voice":
        t0 = time.time()
        r = run_voice.remote(voice_source(slug), voice_job_dict(slug), slug)
        log("voice", t0, True, 16, extra={"inside_s": r["seconds"]})
    else:
        raise SystemExit(f"unknown task {task}")
