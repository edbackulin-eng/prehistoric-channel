"""Exit 0 when episodes may be built, 1 while the channel's format is being rebuilt (owner 16.11).

    python automation/format_gate.py [slug]     prints why it stops

Owner 18.11: no pause - an interim episode (config "interim" slugs) passes while the new format is built.
"""
import json
import sys
from pathlib import Path

CFG = json.loads((Path(__file__).resolve().parent.parent / "config" / "channel.json").read_text(encoding="utf-8"))


def ready(slug=None):
    return CFG.get("format") != "then_now" or bool(CFG.get("format_ready")) or interim(slug)


def interim(slug):
    return bool(slug) and slug in ((CFG.get("interim") or {}).get("slugs") or [])


if __name__ == "__main__":
    if ready(sys.argv[1] if len(sys.argv) > 1 else None):
        sys.exit(0)
    print("the new format (then_now) is not ready yet: no episode is built, no quota spent (owner 16.11)")
    sys.exit(1)
