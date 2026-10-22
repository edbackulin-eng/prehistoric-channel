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
import json, os, sys, time, gc, pathlib, threading
OUT = pathlib.Path("/kaggle/working")
OUT.mkdir(parents=True, exist_ok=True)
report = {"tts": [], "images": [], "errors": []}

# ---- the session's own deadline --------------------------------------------
# 22.10: a nine-image job that should have taken eight minutes sat in RUNNING
# for two hours and had to be killed by hand. It burned two hours of a thirty
# hour week shared by three channels, and Kaggle serves no log at all for an
# unfinished run - not by CLI, not by the stream endpoint - so there was
# nothing to read afterwards either. A session that overruns now ends itself
# and leaves a report saying how far it got.
#
# The budget is worked out here rather than passed in, because every channel
# runs this same file through a different push.py. A caller may still override
# it with JOB["budget_s"]. Five minutes to install and load, a generous minute
# an image, three seconds a sentence - and twice that before giving up, so an
# ordinary slow day never trips it.
BUDGET_S = float(JOB.get("budget_s") or (300
                                         + 60 * len(JOB.get("image_prompts") or ())
                                         + 3 * len(JOB.get("tts_sentences") or ())))
DEADLINE_S = 2 * BUDGET_S
_stage = {"name": "starting", "since": time.time(), "began": time.time()}


def mark(name):
    """Name the stage, so a session that dies says where it was standing."""
    _stage["name"], _stage["since"] = name, time.time()
    print(f"-- {name} (at {time.time() - _stage['began']:.0f}s)", flush=True)


def _give_up():
    ran, now = time.time() - _stage["began"], time.time()
    report["timed_out"] = {
        "ran_s": round(ran, 1), "deadline_s": round(DEADLINE_S, 1),
        "stage": _stage["name"], "stage_s": round(now - _stage["since"], 1),
        "cuda": bool(torch.cuda.is_available()) if "torch" in sys.modules else None,
        "images_done": len(report["images"]),
    }
    report["errors"].append(f"timeout: {ran:.0f}s over a {DEADLINE_S:.0f}s deadline, "
                            f"stuck in {_stage['name']}")
    try:
        (OUT / "report.json").write_text(json.dumps(report, indent=2))
    except Exception as e:                                       # noqa: BLE001
        print("could not write the report:", e)
    print("\n==== GAVE UP ====\n" + json.dumps(report["timed_out"], indent=2), flush=True)
    sys.stdout.flush()
    # os._exit, not sys.exit: the hang that prompted this was inside a pip
    # install, and a main thread blocked in C code never sees an exception
    # raised from a timer thread. Exit 0 so Kaggle still saves /kaggle/working
    # and the caller can pull report.json; callers check the image count, not
    # the exit status, and a short count with "timed_out" in the report says
    # far more than a kernel marked ERROR with no output at all.
    os._exit(0)


_watchdog = threading.Timer(DEADLINE_S, _give_up)
_watchdog.daemon = True
_watchdog.start()
print(f"deadline: {DEADLINE_S:.0f}s for {len(JOB.get('image_prompts') or ())} image(s), "
      f"{len(JOB.get('tts_sentences') or ())} sentence(s)", flush=True)

import torch
print("torch", torch.__version__, "| cuda", torch.cuda.is_available(),
      "|", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no gpu")
# P100 and T4 have no usable bfloat16, so everything here stays in float16.
DTYPE = torch.float16

# A session that came up without a GPU is the likeliest explanation for that
# two-hour run: FLUX on CPU is not slow, it is hours, and nothing in the old
# code would have noticed. Stop now rather than spend the week's quota finding
# out, and say so in the report.
if not torch.cuda.is_available():
    report["errors"].append("no GPU in this session; FLUX or Chatterbox on CPU "
                            "would run for hours, so nothing was attempted")
    (OUT / "report.json").write_text(json.dumps(report, indent=2))
    print("NO GPU - stopping before any model is loaded", flush=True)
    sys.stdout.flush()
    os._exit(0)

# ----------------------------------------------------------------- Chatterbox
if JOB.get("tts_sentences"):
    try:
        mark("chatterbox: pip install")
        os.system("pip install -q chatterbox-tts soundfile")
        # NOT torchaudio: installing chatterbox pulls a torch that no longer
        # matches Kaggle's preinstalled torchaudio, and importing it dies with
        # "Could not load this library: libtorchaudio.so". soundfile writes the
        # WAV just as well and has no torch coupling at all.
        import numpy as _np
        import soundfile as _sf
        from chatterbox.tts import ChatterboxTTS
        mark("chatterbox: load")
        t0 = time.time()
        model = ChatterboxTTS.from_pretrained(device="cuda")
        load_s = time.time() - t0
        print(f"chatterbox loaded in {load_s:.1f}s")

        chunks, total_audio = [], 0.0
        mark("chatterbox: generate")
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
        mark("flux: pip install")
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
        mark("flux: load weights")
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

        mark("flux: generate")
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

_watchdog.cancel()          # the work is done; do not shoot it while it saves
(OUT / "report.json").write_text(json.dumps(report, indent=2))
print("\n==== REPORT ====")
print(json.dumps(report, indent=2))
'''
