"""The notebook body that runs on Kaggle's free GPU.

push.py injects a JOB dict above this source and uploads the result as a notebook.
Everything it produces lands in /kaggle/working, which `kaggle kernels output`
pulls back down.

Two jobs share one GPU session because Kaggle bills by session time, not by task:
  - Chatterbox TTS  (MIT licence) for the voice comparison
  - FLUX.1-schnell  (Apache-2.0)  for the imagined shots NASA cannot supply

Both licences permit commercial use. Nothing here clones a real person's voice:
Chatterbox runs on its built-in speaker, with no reference audio.
"""
SOURCE = r'''
import json, os, sys, time, gc, pathlib
OUT = pathlib.Path("/kaggle/working")
OUT.mkdir(parents=True, exist_ok=True)
report = {"tts": [], "images": [], "errors": []}

import torch
print("torch", torch.__version__, "| cuda", torch.cuda.is_available(),
      "|", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no gpu")
# P100 and T4 have no usable bfloat16, so everything here stays in float16.
DTYPE = torch.float16

# ----------------------------------------------------------------- Chatterbox
if JOB.get("tts_sentences"):
    try:
        os.system("pip install -q chatterbox-tts soundfile")
        # NOT torchaudio: installing chatterbox pulls a torch that no longer
        # matches Kaggle's preinstalled torchaudio, and importing it dies with
        # "Could not load this library: libtorchaudio.so". soundfile writes the
        # WAV just as well and has no torch coupling at all.
        import numpy as _np
        import soundfile as _sf
        from chatterbox.tts import ChatterboxTTS
        t0 = time.time()
        model = ChatterboxTTS.from_pretrained(device="cuda")
        load_s = time.time() - t0
        print(f"chatterbox loaded in {load_s:.1f}s")

        chunks, total_audio = [], 0.0
        t0 = time.time()
        for i, sentence in enumerate(JOB["tts_sentences"]):
            wav = model.generate(
                sentence,
                exaggeration=JOB.get("exaggeration", 0.5),
                cfg_weight=JOB.get("cfg_weight", 0.5),
            )
            chunks.append(wav)
            total_audio += wav.shape[-1] / model.sr
            print(f"  [{i+1}/{len(JOB['tts_sentences'])}] {total_audio:.1f}s audio", flush=True)
        gen_s = time.time() - t0

        audio = torch.cat(chunks, dim=-1)
        _sf.write(str(OUT / "chatterbox.wav"),
                  audio.cpu().numpy().squeeze().astype(_np.float32), model.sr)
        report["tts"] = {
            "load_s": round(load_s, 1),
            "generate_s": round(gen_s, 1),
            "audio_s": round(total_audio, 1),
            "realtime_factor": round(total_audio / gen_s, 2),
            "sentences": len(JOB["tts_sentences"]),
            "sample_rate": model.sr,
        }
        print("TTS:", report["tts"])
        del model, chunks, audio
        gc.collect(); torch.cuda.empty_cache()
    except Exception as e:
        report["errors"].append(f"tts: {type(e).__name__}: {e}")
        print("TTS FAILED:", e)

# ------------------------------------------------------------- FLUX.1-schnell
if JOB.get("image_prompts"):
    try:
        # Three things have to line up here, and Kaggle's base image gets each
        # one wrong in a different way:
        #   -U is essential, because a plain install sees the preinstalled older
        #     diffusers as satisfying the requirement and leaves it in place
        #     ("cannot import name 'FluxPipeline'").
        #   torchao must be upgraded with it, because current diffusers imports
        #     FqnToConfig from it via diffusers.loaders.ip_adapter.
        #   and if the pair still will not import, fall back to a diffusers old
        #     enough to predate that dependency but new enough to have Flux.
        # bitsandbytes must go in with this first install, not later: transformers
        # decides whether 4-bit quantisation is available when it is imported and
        # caches that answer, so installing it afterwards leaves the process
        # insisting it needs "the latest version of bitsandbytes".
        os.system("pip install -q -U diffusers torchao transformers accelerate "
                  "sentencepiece protobuf huggingface_hub bitsandbytes psutil")
        try:
            import diffusers
            from diffusers import FluxPipeline
        except Exception as import_error:
            print("current diffusers failed to import, pinning:", import_error)
            os.system("pip install -q 'diffusers==0.31.0'")
            for mod in [m for m in list(sys.modules) if m.startswith("diffusers")]:
                del sys.modules[mod]
            import diffusers
            from diffusers import FluxPipeline
        print("diffusers", diffusers.__version__)

        # FLUX.1-schnell is Apache-2.0, but the HF repo is gated, so the download
        # itself needs an authenticated account that has accepted the terms.
        if JOB.get("hf_token"):
            os.environ["HF_TOKEN"] = JOB["hf_token"]
            os.environ["HUGGING_FACE_HUB_TOKEN"] = JOB["hf_token"]
        import psutil
        print(f"RAM total {psutil.virtual_memory().total/1e9:.1f} GB, "
              f"available {psutil.virtual_memory().available/1e9:.1f} GB")

        # FLUX.1-schnell in fp16 is ~24 GB of weights, and the T5 text encoder is
        # most of that. Loading it straight killed the kernel outright - no
        # traceback, just "Kernel died", which is the OOM reaper rather than an
        # exception. Quantising both heavy components to 4-bit brings the load
        # inside the box; enable_model_cpu_offload then keeps VRAM inside a T4.
        from diffusers import FluxTransformer2DModel
        from diffusers import BitsAndBytesConfig as DiffBnb
        from transformers import BitsAndBytesConfig as TfBnb
        from transformers import T5EncoderModel

        # Prefer the mirror. The weights are Apache-2.0 so copying them is
        # allowed outright; what this buys is that a gated repo, a revoked token
        # or changed terms upstream cannot stop a build.
        MIRROR = pathlib.Path("/kaggle/input/whatif-flux-weights/flux4bit")
        REPO = str(MIRROR) if MIRROR.exists() else "black-forest-labs/FLUX.1-schnell"
        report["weights_source"] = "kaggle-mirror" if MIRROR.exists() else "huggingface"
        print("weights from:", report["weights_source"], REPO)
        t0 = time.time()
        text_encoder_2 = T5EncoderModel.from_pretrained(
            REPO, subfolder="text_encoder_2", torch_dtype=DTYPE,
            quantization_config=TfBnb(load_in_4bit=True,
                                      bnb_4bit_compute_dtype=DTYPE))
        transformer = FluxTransformer2DModel.from_pretrained(
            REPO, subfolder="transformer", torch_dtype=DTYPE,
            quantization_config=DiffBnb(load_in_4bit=True,
                                        bnb_4bit_compute_dtype=DTYPE))
        pipe = FluxPipeline.from_pretrained(
            REPO, text_encoder_2=text_encoder_2, transformer=transformer,
            torch_dtype=DTYPE)
        pipe.enable_model_cpu_offload()
        load_s = time.time() - t0
        print(f"RAM after load: {psutil.virtual_memory().available/1e9:.1f} GB free")
        print(f"flux loaded in {load_s:.1f}s")

        img_dir = OUT / "images"; img_dir.mkdir(exist_ok=True)
        style = JOB.get("style", "")
        times = []
        for i, prompt in enumerate(JOB["image_prompts"]):
            full = f"{prompt}, {style}" if style else prompt
            t1 = time.time()
            image = pipe(
                full,
                num_inference_steps=JOB.get("steps", 4),   # schnell is a 4-step model
                guidance_scale=0.0,                        # schnell ignores CFG
                width=JOB.get("width", 1360),
                height=JOB.get("height", 768),
                generator=torch.Generator("cpu").manual_seed(JOB.get("seed", 0) + i),
            ).images[0]
            dt = time.time() - t1
            times.append(dt)
            name = f"{i:03d}.png"
            image.save(img_dir / name)
            report["images"].append({"file": name, "prompt": full, "seconds": round(dt, 1)})
            print(f"  [{i+1}/{len(JOB['image_prompts'])}] {name} in {dt:.1f}s", flush=True)
        if times:
            report["image_timing"] = {
                "load_s": round(load_s, 1),
                "mean_s": round(sum(times) / len(times), 1),
                "count": len(times),
            }
            print("IMAGES:", report["image_timing"])
    except Exception as e:
        report["errors"].append(f"flux: {type(e).__name__}: {e}")
        print("FLUX FAILED:", e)

(OUT / "report.json").write_text(json.dumps(report, indent=2))
print("\n==== REPORT ====")
print(json.dumps(report, indent=2))
'''
