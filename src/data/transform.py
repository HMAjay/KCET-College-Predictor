# ─────────────────────────────────────────────
#  src/data/transform.py
#  Master dataset loading and validation helpers
# ─────────────────────────────────────────────
"""
Called internally by train_model.py.
Not meant to be run standalone, but you can:
    python src/data/transform.py   (prints a preview)
"""

import os
import sys
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from config import MASTER_CSV


def _filter_invalid_master_rows(df: pd.DataFrame) -> pd.DataFrame:
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
def load_master() -> pd.DataFrame:
    if not os.path.exists(MASTER_CSV):
        raise FileNotFoundError(
            f"Master CSV not found at {MASTER_CSV}.\n"
            "Run merge_data.py first."
        )
    df = pd.read_csv(MASTER_CSV)
    df["Cutoff_Rank"] = pd.to_numeric(df["Cutoff_Rank"], errors="coerce")
    df.dropna(subset=["Cutoff_Rank", "Branch", "Category", "College_Code"], inplace=True)
    df = _filter_invalid_master_rows(df)
    return df


# ─────────────────────────────────────────────
if __name__ == "__main__":
    df = load_master()
    print(f"Rows       : {len(df):,}")
    print(f"Colleges   : {df['College_Code'].nunique()}")
    print(f"Branches   : {df['Branch'].nunique()}")
    print(f"Categories : {df['Category'].nunique()}")
    print(f"Years      : {sorted(df['Year'].dropna().astype(int).unique().tolist())}")
