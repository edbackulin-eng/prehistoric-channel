"""Refuse a commit or push that carries a secret or a heavy file.

    python scripts/check_repo.py            # every tracked file (CI)
    python scripts/check_repo.py --staged   # what is about to be committed (pre-commit hook)

This repository is public. The What If channel came within one push of
publishing a Hugging Face token in a Kaggle job file and did publish 311 MB of
video before it was noticed. Both are caught here, in CI on every push and
locally before a commit and before a push (.githooks, core.hooksPath).

Locally the actual values in .env and ~/.kaggle/kaggle.json are searched for
as well, so a key with no recognisable shape is still caught.
"""
import re
import subprocess
import sys
from pathlib import Path

MAX_BYTES = 5 * 1024 * 1024
PATTERNS = {
    "Groq key": r"gsk_[A-Za-z0-9]{20,}",
    "Google API key": r"AIza[0-9A-Za-z_-]{30,}",
    "Google OAuth token": r"ya29\.[0-9A-Za-z_-]{20,}",
    "Google OAuth refresh token": r"1//0[0-9A-Za-z_-]{20,}",
    "Google client secret": r"GOCSPX-[0-9A-Za-z_-]{20,}",
    "Hugging Face token": r"hf_[A-Za-z0-9]{30,}",
    "GitHub token": r"gh[pousr]_[A-Za-z0-9]{30,}",
    "Anthropic / Claude token": r"sk-ant-[A-Za-z0-9_-]{20,}",
    "Kaggle key": r"\"key\"\s*:\s*\"[0-9a-f]{32}\"",
    "private key": r"-----BEGIN [A-Z ]*PRIVATE KEY-----",
    "Pixabay key": r"\d{6,10}-[0-9a-f]{32}",
    "Kaggle API token": r"KGAT_[A-Za-z0-9]{20,}",
}
SKIP = {"scripts/check_repo.py"}


def files(staged):
    args = (["git", "diff", "--cached", "--name-only", "--diff-filter=ACM"] if staged
            else ["git", "ls-files"])
    out = subprocess.run(args, capture_output=True, text=True, check=True).stdout
    return [f for f in out.splitlines() if f and f not in SKIP]


def local_secrets():
    """The real secret values on this machine, to search for verbatim."""
    import json
    values = []
    env = Path(".env")
    if env.exists():
        for line in env.read_text(encoding="utf-8-sig").splitlines():
            _, _, v = line.partition("=")
            v = v.strip().strip(chr(34)).strip(chr(39))
            if len(v) >= 12:
                values.append(v)
    cfg = Path.home() / ".kaggle" / "kaggle.json"
    if cfg.exists():
        try:
            values.append(json.loads(cfg.read_text())["key"])
        except (ValueError, KeyError):
            pass
    return values


def main():
    staged = "--staged" in sys.argv
    problems = []
    secrets = local_secrets()
    for name in files(staged):
        path = Path(name)
        if not path.is_file():
            continue
        size = path.stat().st_size
        if size > MAX_BYTES:
            problems.append(f"{name}: {size / 1e6:.1f} MB, over the 5 MB limit (media belongs in .gitignore)")
            continue
        text = path.read_bytes().decode("utf-8", "ignore")
        for label, pattern in PATTERNS.items():
            if re.search(pattern, text):
                problems.append(f"{name}: looks like it contains a {label}")
        if any(v in text for v in secrets):
            problems.append(f"{name}: contains a value from .env or kaggle.json")
    if problems:
        print("REFUSED:")
        for p in problems:
            print("  " + p)
        return 1
    print(f"ok: {len(files(staged))} file(s), no secrets, nothing over 5 MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
