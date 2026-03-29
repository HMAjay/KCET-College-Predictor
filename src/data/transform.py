# ─────────────────────────────────────────────
#  src/data/transform.py
#  Feature engineering — convert raw master CSV
#  into ML-ready feature matrix + label encoders
# ─────────────────────────────────────────────
"""
Called internally by train_model.py and predictor.py.
Not meant to be run standalone, but you can:
    python src/data/transform.py   (prints a preview)
"""

import os
import sys
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from config import MASTER_CSV


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
    return df


# ─────────────────────────────────────────────
def build_encoders(df: pd.DataFrame) -> dict:
    """
    Fit LabelEncoders for Branch, Category, College_Code.
    Returns a dict:  {'branch': le, 'category': le, 'college': le}
    """
    encoders = {}
    for col, key in [("Branch", "branch"), ("Category", "category"), ("College_Code", "college")]:
        le = LabelEncoder()
        le.fit(df[col].astype(str))
        encoders[key] = le
    return encoders


# ─────────────────────────────────────────────
def build_features(df: pd.DataFrame, encoders: dict) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns (X, y) arrays ready for sklearn.

    Features (X):
        - Cutoff_Rank  (numeric)
        - Branch       (label-encoded)
        - Category     (label-encoded)
        - Year         (numeric, 0 if missing)

    Target (y):
        - College_Code (label-encoded)
    """
    df = df.copy()

    # Encode categoricals — unseen values fall back to 0
    df["Branch_enc"]   = _safe_transform(encoders["branch"],   df["Branch"])
    df["Category_enc"] = _safe_transform(encoders["category"], df["Category"])
    df["College_enc"]  = _safe_transform(encoders["college"],  df["College_Code"])
    df["Year"]         = pd.to_numeric(df.get("Year", 0), errors="coerce").fillna(0).astype(int)

    X = df[["Cutoff_Rank", "Branch_enc", "Category_enc", "Year"]].values
    y = df["College_enc"].values

    return X, y


# ─────────────────────────────────────────────
def _safe_transform(le: LabelEncoder, series: pd.Series) -> pd.Series:
    """Transform; unknown labels map to 0 instead of crashing."""
    known = set(le.classes_)
    mapped = series.astype(str).map(lambda v: v if v in known else le.classes_[0])
    return le.transform(mapped)


# ─────────────────────────────────────────────
if __name__ == "__main__":
    df = load_master()
    enc = build_encoders(df)
    X, y = build_features(df, enc)
    print(f"Feature matrix : {X.shape}")
    print(f"Target classes : {len(enc['college'].classes_)} colleges")
    print(f"Sample X row   : {X[0]}")
    print(f"Sample y label : {enc['college'].classes_[y[0]]}")
