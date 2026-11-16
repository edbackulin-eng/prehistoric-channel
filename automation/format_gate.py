"""Exit 0 when episodes may be built, 1 while the channel's format is being rebuilt (owner 16.11).

    python automation/format_gate.py     prints why it stops
"""
import json
import sys
from pathlib import Path

CFG = json.loads((Path(__file__).resolve().parent.parent / "config" / "channel.json").read_text(encoding="utf-8"))


def ready():
    return CFG.get("format") != "then_now" or bool(CFG.get("format_ready"))


if __name__ == "__main__":
    if ready():
        sys.exit(0)
    print("the new format (then_now) is not ready yet: no episode is built, no quota spent (owner 16.11)")
    sys.exit(1)
