"""Mirror the FLUX weights into Kaggle so Hugging Face leaves the critical path.

The weights are Apache-2.0, so copying them is allowed outright. What this buys
is independence: the repo is gated, a token can be revoked, terms can change for
future releases, and any of those would stop every build. After this runs, the
generation kernels read the model from a Kaggle kernel output instead, and no
build touches huggingface.co at all.

Run once. Re-run only to pick up a newer revision deliberately.
"""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SLUG = "whatif-flux-weights"

SOURCE = r'''
import os, time, json, pathlib
OUT = pathlib.Path("/kaggle/working/flux4bit")
OUT.mkdir(parents=True, exist_ok=True)

os.system("pip install -q -U diffusers torchao transformers accelerate "
          "sentencepiece protobuf huggingface_hub bitsandbytes")
import torch
from diffusers import FluxTransformer2DModel
from diffusers import BitsAndBytesConfig as DiffBnb
from transformers import BitsAndBytesConfig as TfBnb
from transformers import T5EncoderModel, CLIPTextModel, AutoTokenizer
from diffusers import AutoencoderKL, FlowMatchEulerDiscreteScheduler

REPO = "black-forest-labs/FLUX.1-schnell"
REV = JOB.get("revision") or "main"
TOK = JOB.get("hf_token")
DTYPE = torch.float16
report = {"repo": REPO, "revision": REV, "parts": {}, "errors": []}

# Mirroring the raw repository failed: it is over 30 GB once the duplicated
# root checkpoint is counted, which is past what a kernel may emit. The 4-bit
# build is what the generation kernels actually load anyway, so that is what
# gets mirrored - roughly a third of the size and already in the right form.
t0 = time.time()
try:
    te2 = T5EncoderModel.from_pretrained(
        REPO, subfolder="text_encoder_2", revision=REV, token=TOK,
        torch_dtype=DTYPE,
        quantization_config=TfBnb(load_in_4bit=True, bnb_4bit_compute_dtype=DTYPE))
    te2.save_pretrained(OUT / "text_encoder_2")
    report["parts"]["text_encoder_2"] = "ok"
    del te2
except Exception as e:
    report["errors"].append(f"text_encoder_2: {e}")

try:
    tr = FluxTransformer2DModel.from_pretrained(
        REPO, subfolder="transformer", revision=REV, token=TOK,
        torch_dtype=DTYPE,
        quantization_config=DiffBnb(load_in_4bit=True, bnb_4bit_compute_dtype=DTYPE))
    tr.save_pretrained(OUT / "transformer")
    report["parts"]["transformer"] = "ok"
    del tr
except Exception as e:
    report["errors"].append(f"transformer: {e}")

# The small components stay unquantised; together they are well under a gigabyte.
for name, loader, kwargs in (
        ("text_encoder", CLIPTextModel, {"torch_dtype": DTYPE}),
        ("vae", AutoencoderKL, {"torch_dtype": DTYPE})):
    try:
        m = loader.from_pretrained(REPO, subfolder=name, revision=REV,
                                   token=TOK, **kwargs)
        m.save_pretrained(OUT / name)
        report["parts"][name] = "ok"
        del m
    except Exception as e:
        report["errors"].append(f"{name}: {e}")

for name in ("tokenizer", "tokenizer_2"):
    try:
        AutoTokenizer.from_pretrained(REPO, subfolder=name, revision=REV,
                                      token=TOK).save_pretrained(OUT / name)
        report["parts"][name] = "ok"
    except Exception as e:
        report["errors"].append(f"{name}: {e}")

try:
    FlowMatchEulerDiscreteScheduler.from_pretrained(
        REPO, subfolder="scheduler", revision=REV,
        token=TOK).save_pretrained(OUT / "scheduler")
    report["parts"]["scheduler"] = "ok"
except Exception as e:
    report["errors"].append(f"scheduler: {e}")

total = sum(f.stat().st_size for f in OUT.rglob("*") if f.is_file())
report["total_gb"] = round(total / 1e9, 2)
report["seconds"] = round(time.time() - t0)
pathlib.Path("/kaggle/working/manifest.json").write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
'''


def main():
    sys.path.insert(0, str(ROOT))
    import push as kpush

    job = {"_comment": "One-off: mirror FLUX.1-schnell into a Kaggle kernel output.",
           "revision": "main",
           "image_prompts": ["__mirror__"]}   # makes push.py inject HF_TOKEN
    token = kpush._env_token()
    if not token:
        raise SystemExit("HF_TOKEN not found; the source repo is gated")

    build = ROOT / "build"
    build.mkdir(parents=True, exist_ok=True)
    user = kpush.username()
    cell = f"JOB = {json.dumps({'revision': job['revision'], 'hf_token': token}, indent=2)}\n\n{SOURCE}"
    nb = {"cells": [{"cell_type": "code", "metadata": {}, "execution_count": None,
                     "outputs": [], "source": cell.splitlines(keepends=True)}],
          "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                      "name": "python3"}},
          "nbformat": 4, "nbformat_minor": 5}
    (build / "notebook.ipynb").write_text(json.dumps(nb, indent=1), encoding="utf-8")
    (build / "kernel-metadata.json").write_text(json.dumps({
        "id": f"{user}/{SLUG}",
        "title": "whatif flux weights",
        "code_file": "notebook.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,        # quantisation needs the GPU present
        "enable_internet": True,
        "dataset_sources": [], "competition_sources": [], "kernel_sources": [],
    }, indent=2), encoding="utf-8")

    r = subprocess.run([sys.executable, "-m", "kaggle", "kernels", "push",
                        "-p", str(build)], capture_output=True, text=True)
    print(r.stdout.strip() or r.stderr.strip())
    print(f"\nmirror kernel: {user}/{SLUG}")
    print("once complete, generation kernels can list it in kernel_sources and")
    print("read the model from /kaggle/input/whatif-flux-weights/flux")
    return 0


if __name__ == "__main__":
    sys.exit(main())
