# ─────────────────────────────────────────────
#  app/main.py
#  Terminal-based KCET College Predictor UI
# ─────────────────────────────────────────────
"""
Usage:
    python app/main.py

Interactive terminal interface.
Type 'quit' or 'exit' at any prompt to leave.
"""

import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.model.predictor import KCETPredictor
from src.utils.helpers import banner
from config import CATEGORIES, BRANCHES

try:
    from tabulate import tabulate
    HAS_TABULATE = True
except ImportError:
    HAS_TABULATE = False

# ── Optional colour support ───────────────────
try:
    from colorama import init, Fore, Style
    init(autoreset=True)
    GREEN  = Fore.GREEN
    CYAN   = Fore.CYAN
    YELLOW = Fore.YELLOW
    RED    = Fore.RED
    BOLD   = Style.BRIGHT
    RESET  = Style.RESET_ALL
except ImportError:
    GREEN = CYAN = YELLOW = RED = BOLD = RESET = ""


# ─────────────────────────────────────────────
def ask(prompt: str, validator=None, hint: str = ""):
    """Read input with optional validation loop."""
    while True:
        if hint:
            print(f"   {YELLOW}({hint}){RESET}")
        val = input(f"  {CYAN}{prompt}{RESET}: ").strip()
        if val.lower() in ("quit", "exit", "q"):
            print("\nGoodbye! 👋\n")
            sys.exit(0)
        if validator is None or validator(val):
            return val
        print(f"  {RED}Invalid input. Please try again.{RESET}")


# ─────────────────────────────────────────────
def print_results(results: list[dict]):
    if not results:
        print(f"\n  {RED}No eligible colleges found for the given inputs.{RESET}")
        print("  Try a different category or check your rank.\n")
        return

    print(f"\n  {GREEN}{BOLD}✔  {len(results)} college(s) found:{RESET}\n")

    if HAS_TABULATE:
        rows = []
        for i, r in enumerate(results, 1):
            rows.append([
                i,
                r.get("College_Code", ""),
                str(r.get("College_Name") or "")[:45],
                r.get("Branch", "")[:30],
                r.get("Cutoff_Rank", "N/A"),
                r.get("Gap", "N/A"),
                r.get("Source", ""),
            ])
        print(tabulate(
            rows,
            headers=["#", "Code", "College Name", "Branch", "Cutoff", "Gap", "Source"],
            tablefmt="rounded_outline",
        ))
    else:
        for i, r in enumerate(results, 1):
            print(f"  {i:2}. [{r.get('College_Code','')}] {r.get('College_Name','')}")
            print(f"       Branch   : {r.get('Branch','')}")
            print(f"       Cutoff   : {r.get('Cutoff_Rank','N/A')}   Gap: {r.get('Gap','N/A')}")
            print()


# ─────────────────────────────────────────────
def show_hints():
    print(f"\n  {YELLOW}Common categories :{RESET} " + ", ".join(CATEGORIES[:8]) + " …")
    print(f"  {YELLOW}Common branches   :{RESET}")
    for i, b in enumerate(BRANCHES, 1):
        print(f"    {i:2}. {b}")


# ─────────────────────────────────────────────
def main():
    print(banner("🎓  KCET College Predictor", char="═"))
    print(f"  {CYAN}Type 'quit' at any time to exit.{RESET}\n")

    print("  Loading model …", end=" ", flush=True)
    try:
        predictor = KCETPredictor()
        meta = predictor.metadata
        print(f"{GREEN}OK{RESET}")
        print(f"  Model covers {meta.get('n_colleges','')} colleges"
              f" | Test accuracy: {meta.get('test_accuracy','N/A')}")
    except FileNotFoundError as e:
        print(f"\n{RED}Error:{RESET} {e}")
        sys.exit(1)

    while True:
        print(banner("New Prediction", char="─", width=50))

        # ── Rank ──────────────────────────────
        rank_str = ask(
            "Enter your KCET Rank",
            validator=lambda v: v.isdigit() and 1 <= int(v) <= 200000,
            hint="e.g. 12500  |  range 1–200000",
        )
        rank = int(rank_str)

        # ── Branch ────────────────────────────
        show_hints()
        branch = ask(
            "Enter Branch",
            hint="e.g. Computer Science Engineering  (partial names OK)",
        )

        # ── Category ─────────────────────────
        category = ask(
            "Enter Category",
            validator=lambda v: v.strip().upper() in CATEGORIES,
            hint="e.g. GM | SC | OBC | 1G | 2AG  (must be exact)",
        )

        # ── Top N ────────────────────────────
        top_n_str = ask(
            "How many results? [press Enter for 10]",
            validator=lambda v: v == "" or (v.isdigit() and 1 <= int(v) <= 50),
            hint="1–50",
        )
        top_n = int(top_n_str) if top_n_str else 10

        # ── Predict ──────────────────────────
        print(f"\n  Searching for rank {BOLD}{rank}{RESET}, branch '{branch}', category '{category.upper()}' …")
        results = predictor.predict(rank, branch, category.upper(), top_n)
        print_results(results)

        again = input(f"\n  {YELLOW}Run another prediction? (y/n) :{RESET} ").strip().lower()
        if again not in ("y", "yes"):
            print("\n  Good luck with your admissions! 🎓\n")
            break


# ─────────────────────────────────────────────
if __name__ == "__main__":
    main()
