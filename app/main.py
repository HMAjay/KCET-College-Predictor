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
        print(f"\n  {RED}No colleges cleared the projected 2026 cutoff for the given inputs.{RESET}")
        print("  Try a different branch/category or a broader rank range.\n")
        return

    print(f"\n  {GREEN}{BOLD}✔  {len(results)} college(s) found:{RESET}\n")

    if HAS_TABULATE:
        rows = []
        for i, r in enumerate(results, 1):
            branch_option = r.get("Branch_Option") or r.get("Branch", "")
            rows.append([
                i,
                f"{r.get('College_Code', '')} | {branch_option}",
                str(r.get("College_Name") or ""),
                r.get("Cutoff_Rank", "N/A"),
                r.get("Cutoff_2025", "N/A"),
                r.get("Gap", "N/A"),
            ])
        print(tabulate(
            rows,
            headers=["#", "College / Course", "College Name", "2026 Predicted", "2025 Cutoff", "Gap"],
            tablefmt="rounded_outline",
        ))
    else:
        for i, r in enumerate(results, 1):
            branch_option = r.get("Branch_Option") or r.get("Branch", "")
            print(
                f"  {i:2}. [{r.get('College_Code','')}] "
                f"{branch_option} - {r.get('College_Name','')}"
            )
            print(
                f"       2026 Predicted: {r.get('Cutoff_Rank','N/A')}   "
                f"2025 Cutoff: {r.get('Cutoff_2025','N/A')}   "
                f"Gap: {r.get('Gap','N/A')}"
            )
            print()


# ─────────────────────────────────────────────
def show_hints(predictor: KCETPredictor):
    print(f"\n  {YELLOW}Common categories :{RESET} " + ", ".join(predictor.available_categories[:10]) + " …")
    print(f"  {YELLOW}Common courses    :{RESET}")
    for i, b in enumerate(predictor.common_branch_options, 1):
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
        print(
            f"  Model covers {meta.get('n_colleges','')} colleges"
            f" | Trend years: {meta.get('min_year','?')}-{meta.get('max_year','?')}"
            f" | Target: {meta.get('target_year','2026')}"
        )
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
        show_hints(predictor)
        branch = ask(
            "Enter Branch",
            hint="e.g. IE, RA, CSE, Information Science, Robotics and Automation",
        )

        # ── Category ─────────────────────────
        category = ask(
            "Enter Category",
            validator=lambda v: v.strip().upper() in predictor.available_categories,
            hint="e.g. GM | SCG | STG | 1G | 2AG | 3BK",
        )

        # ── Top N ────────────────────────────
        top_n_str = ask(
            "How many results? [press Enter for 10]",
            validator=lambda v: v == "" or (v.isdigit() and 1 <= int(v) <= 50),
            hint="1–50",
        )
        top_n = int(top_n_str) if top_n_str else 10

        # ── Predict ──────────────────────────
        branch_matches = predictor.resolve_branch_matches(branch)
        if branch_matches:
            branch_labels = [
                predictor.branch_option_map.get(name, name)
                for name in branch_matches[:3]
            ]
            print(f"\n  {YELLOW}Matched branch(es):{RESET} " + ", ".join(branch_labels))
        else:
            print(f"\n  {RED}No similar branch found in the dataset for '{branch}'.{RESET}")

        print(
            f"  Predicting {BOLD}{predictor.target_year}{RESET} cutoff for rank {BOLD}{rank}{RESET}, "
            f"branch '{branch}', category '{category.upper()}' …"
        )
        results = predictor.predict(rank, branch, category.upper(), top_n)
        print_results(results)

        again = input(f"\n  {YELLOW}Run another prediction? (y/n) :{RESET} ").strip().lower()
        if again not in ("y", "yes"):
            print("\n  Good luck with your admissions! 🎓\n")
            break


# ─────────────────────────────────────────────
if __name__ == "__main__":
    main()
