"""Voice sample: Chatterbox (MIT) on a Kaggle GPU, built-in voice only.

    python kaggle/voice_job.py episodes/living-fossils/script.txt 30

Takes the lines that fill the first `seconds` of the Kokoro narration, reads
them with Chatterbox's default speaker (no reference audio: nobody's voice is
cloned) and Latin names in their phonetic respelling. Each line is transcribed
back with Whisper and redrawn when words go missing. Copied from Closed Tabs'
chatterbox_job.py, with this channel's documentary delivery.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "kaggle"))
from core import delivery, narrate  # noqa: E402

SLUG = "prehistoric-voice"
BUILD = ROOT / "kaggle" / "build_voice"
RESULTS = ROOT / "kaggle" / "results_voice"

SOURCE_PATH = Path(__file__).with_name("chatterbox_job.py")


def body():
    src = SOURCE_PATH.read_text(encoding="utf-8")
    start = src.index("SOURCE = r'''") + len("SOURCE = r'''")
    code = src[start:src.index("'''", start)]
    # documentary delivery: calmer than Closed Tabs' hooks
    code = code.replace('cfg = {"fast": 0.55, "base": 0.45, "slow": 0.30}[line["kind"]]',
                        'cfg = {"fast": 0.50, "base": 0.42, "slow": 0.32}[line["kind"]]')
    code = code.replace('exag = {"fast": 0.70, "base": 0.65, "slow": 0.60}[line["kind"]]',
                        'exag = {"fast": 0.55, "base": 0.50, "slow": 0.45}[line["kind"]]')
    # every line also on its own, so the sample can be laid on the edit line by line
    code += """
for _i, (_l, _w) in enumerate(matched):
    torchaudio.save(str(OUT / f"line_{_i:02d}.wav"), torch.tensor(_w).unsqueeze(0), SR)
"""
    return code


def main():
    import push
    script = Path(sys.argv[1]).read_text(encoding="utf-8")
    seconds = float(sys.argv[2]) if len(sys.argv) > 2 else 30
    timeline = json.loads((ROOT / "work" / "voice" / "bm_lewis.json").read_text())["timeline"]
    keep = sum(1 for l in timeline if l["end"] <= seconds + 1.0)
    lines = delivery.parse(script)[:keep]
    lex = narrate.lexicon()
    job = {"lines": [{**l.__dict__, "text": narrate.respell(l.text, lex)} for l in lines]}
    BUILD.mkdir(parents=True, exist_ok=True)
    cell = f"JOB = {json.dumps(job, indent=1)}\n{body()}"
    nb = {"cells": [{"cell_type": "code", "metadata": {}, "execution_count": None,
                     "outputs": [], "source": cell.splitlines(keepends=True)}],
          "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
          "nbformat": 4, "nbformat_minor": 5}
    (BUILD / "notebook.ipynb").write_text(json.dumps(nb, indent=1), encoding="utf-8")
    os.environ["FLUX_MIRROR"] = "0"
    meta = push._metadata(push.username(), SLUG, SLUG)
    (BUILD / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    ref = push.submit_when_idle(BUILD, job_hours=0.3)
    ok = push.wait(ref, poll_s=30, timeout_s=2 * 3600)
    push.publish_ledger()
    push.fetch(ref, RESULTS)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
