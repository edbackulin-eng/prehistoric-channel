"""Three thumbnails from the FLUX bases (kaggle/recon_job.py), into test/.

    python automation/make_thumbs.py living-fossils

Seeds come from episodes/<slug>/recon_picks.json (chosen by eye on the contact
sheet); every thumbnail is measured for readability at 336x189 (pipeline/doc_thumb.py).
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from pipeline import doc_thumb  # noqa: E402

VARIANTS = [
    ("thumb_coelacanth", ["THOUGHT", "EXTINCT"], "left", 1),
    ("thumb_hsc", ["OLDER THAN", "DINOSAURS"], "right", 1),
    ("thumb_croc", ["THEY", "SURVIVED"], "left", 1),
]


def main(slug):
    picks = json.loads((ROOT / "episodes" / slug / "recon_picks.json").read_text(encoding="utf-8"))
    up = ROOT / "kaggle" / "results_recon" / "up"
    out = ROOT / "test"
    out.mkdir(exist_ok=True)
    reps = []
    for i, (base, words, side, accent) in enumerate(VARIANTS, 1):
        seed = picks.get(base, 0)
        rep = doc_thumb.compose(up / f"{base}__s{seed}.jpg", words, side, out / f"thumbnail_{i}.jpg", accent)
        rep["base"] = f"{base}__s{seed}"
        reps.append(rep)
        print(rep)
    (out / "thumbnails_report.json").write_text(json.dumps(reps, indent=1), encoding="utf-8")
    return 0 if all(r["readable"] for r in reps) else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
