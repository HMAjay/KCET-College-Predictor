# ─────────────────────────────────────────────
#  src/data/merge_data.py
#  Step 3 — Merge all cleaned yearly CSVs into one master dataset
# ─────────────────────────────────────────────
"""
Usage:
    python src/data/merge_data.py

Reads every *_clean.csv from  data/cleaned/
Writes  data/final/kcet_master.csv

It also infers the Year column from filenames like:
    KCET_2022_Cutoff_clean.csv  →  Year = 2022
"""

import os
import sys
import re
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from config import CLEANED_DIR, FINAL_DIR, MASTER_CSV
from src.utils.helpers import ensure_dirs, banner


# ─────────────────────────────────────────────
def infer_year(filename: str) -> int | None:
    """Pull a 4-digit year from the filename, e.g. 'KCET_2022_clean.csv' → 2022."""
    match = re.search(r"(20\d{2})", filename)
    return int(match.group(1)) if match else None


def filter_invalid_rows(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["College_Code"] = df["College_Code"].astype(str).str.strip()
    df["College_Name"] = df["College_Name"].astype(str).str.strip()
    df["Branch"] = df["Branch"].astype(str).str.strip()
    df["Category"] = df["Category"].astype(str).str.strip().str.upper()

    invalid_name_mask = df["College_Name"].str.lower().isin({"", "nan", "none"})
    invalid_code_mask = df["College_Code"].str.upper().isin({"", "UNKNOWN", "NAN", "NONE"})
    invalid_branch_mask = df["Branch"].str.lower().isin({"", "nan", "none", "college:", "generated"})
    invalid_category_mask = df["Category"].str.lower().isin({"", "nan", "none"})

    return df[~(invalid_name_mask | invalid_code_mask | invalid_branch_mask | invalid_category_mask)].copy()


# ─────────────────────────────────────────────
def merge_all():
    ensure_dirs(CLEANED_DIR, FINAL_DIR)

    files = [f for f in os.listdir(CLEANED_DIR) if f.endswith("_clean.csv")]

    if not files:
        print("⚠  No *_clean.csv files found in data/cleaned/")
        print("   Run clean_data.py first.")
        return

    print(banner("Step 3 — Merging Yearly Datasets"))

    frames = []
    for file in sorted(files):
        path = os.path.join(CLEANED_DIR, file)
        df   = pd.read_csv(path)
        year = infer_year(file)

        if year:
            df["Year"] = year
        else:
            print(f"   ⚠  Could not infer year from: {file}  (Year will be NaN)")
            df["Year"] = None

        frames.append(df)
        print(f"   ✔  {file}  ({len(df):,} rows)  →  Year {year}")

    master = pd.concat(frames, ignore_index=True)

    # ── Quality checks ────────────────────────
    before = len(master)
    master.dropna(subset=["College_Code", "Branch", "Category", "Cutoff_Rank"], inplace=True)
    master = filter_invalid_rows(master)
    master["Cutoff_Rank"] = pd.to_numeric(master["Cutoff_Rank"], errors="coerce")
    master.dropna(subset=["Cutoff_Rank"], inplace=True)
    master["Cutoff_Rank"] = master["Cutoff_Rank"].astype(int)
    after = len(master)

    if before != after:
        print(f"\n   ⚠  Dropped {before - after:,} rows with missing/invalid values.")

    master.to_csv(MASTER_CSV, index=False)

    print(banner(f"Master dataset saved  →  {MASTER_CSV}"))
    print(f"   Total rows : {len(master):,}")
    print(f"   Colleges   : {master['College_Code'].nunique()}")
    print(f"   Branches   : {master['Branch'].nunique()}")
    print(f"   Categories : {master['Category'].nunique()}")
    print(f"   Years      : {sorted(master['Year'].dropna().unique().tolist())}\n")


# ─────────────────────────────────────────────
if __name__ == "__main__":
    merge_all()
