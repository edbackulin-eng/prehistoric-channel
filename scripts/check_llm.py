"""Call every link of the LLM chain once and say which answer.

    python scripts/check_llm.py            # all three
    python scripts/check_llm.py claude     # one

Exit 1 if any link that has a key fails, so the workflow goes red.
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core import llm  # noqa: E402

names = sys.argv[1:] or list(llm.PROVIDERS)
failed = 0
for name in names:
    started = time.monotonic()
    try:
        reply = llm.complete("Reply with exactly the word OK.", max_tokens=300,
                             temperature=0.0, providers=(name,))
        print(f"{name:7s} OK    {time.monotonic() - started:5.1f}s  {reply.strip()[:40]!r}")
    except Exception as e:                                       # noqa: BLE001
        failed += 1
        print(f"{name:7s} FAIL  {time.monotonic() - started:5.1f}s  {str(e)[:200]}")
sys.exit(1 if failed else 0)
