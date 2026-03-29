# ─────────────────────────────────────────────
#  src/utils/helpers.py  —  Shared utilities
# ─────────────────────────────────────────────
import os
import sys
import re

# ── Make project root importable ─────────────
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def ensure_dirs(*dirs):
    """Create directories if they don't exist."""
    for d in dirs:
        os.makedirs(d, exist_ok=True)


def normalize_branch(branch: str) -> str:
    """Lowercase + strip whitespace for consistent matching."""
    return branch.strip().lower()


def normalize_category(cat: str) -> str:
    """Uppercase + strip whitespace."""
    return cat.strip().upper()


def is_college_row(text: str) -> bool:
    """Return True if the row is a college header (starts with E<digits>)."""
    return bool(re.match(r"^E\d{3}", str(text).strip()))


def safe_rank(value) -> float | None:
    """Convert cutoff rank to float; return None if invalid."""
    try:
        v = str(value).replace(",", "").strip()
        if v in ("--", "nan", "", "-"):
            return None
        return float(v)
    except (ValueError, TypeError):
        return None


def banner(text: str, char: str = "─", width: int = 60) -> str:
    """Return a simple terminal banner string."""
    line = char * width
    return f"\n{line}\n  {text}\n{line}"
