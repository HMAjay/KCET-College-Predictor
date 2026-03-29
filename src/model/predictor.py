# ─────────────────────────────────────────────
#  src/model/predictor.py
#  Core prediction engine  (used by app/main.py)
# ─────────────────────────────────────────────
"""
Two-stage prediction:
  1. LOOKUP  — directly query the master DataFrame for rows where
               category matches AND cutoff_rank >= student_rank.
               Sort by closest cutoff (most competitive colleges last).
  2. ML      — if lookup returns nothing, use Random Forest top-k
               probabilities to suggest nearby colleges.
"""

import os
import sys
import joblib
import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from config import MODEL_PATH
from src.data.transform import _safe_transform


# ─────────────────────────────────────────────
class KCETPredictor:

    def __init__(self):
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"Model not found at {MODEL_PATH}.\n"
                "Run:  python src/model/train_model.py"
            )
        bundle = joblib.load(MODEL_PATH)

        self.encoders    = bundle["encoders"]
        self.rf          = bundle["rf_model"]
        self.df          = bundle["master_df"].copy()
        self.metadata    = bundle.get("metadata", {})

        # Pre-normalise for fast lookups
        self.df["Branch_lower"]   = self.df["Branch"].str.lower()
        self.df["Category_upper"] = self.df["Category"].str.upper()

    # ─────────────────────────────────────────
    def predict(
        self,
        rank:     int,
        branch:   str,
        category: str,
        top_n:    int = 10,
    ) -> list[dict]:
        """
        Returns a list of dicts with keys:
            College_Code, College_Name, Branch, Category, Cutoff_Rank, Source
        Sorted from safest (lowest cutoff gap) to hardest.
        """
        results = self._lookup(rank, branch, category, top_n)

        if not results:
            results = self._ml_predict(rank, branch, category, top_n)

        return results

    # ── Stage 1: Exact cutoff lookup ─────────
    def _lookup(self, rank, branch, category, top_n):
        df = self.df

        b = branch.strip().lower()
        c = category.strip().upper()

        # ── Primary: exact branch + category ──
        mask = (
            (df["Branch_lower"].str.contains(b, na=False)) &
            (df["Category_upper"] == c) &
            (df["Cutoff_Rank"] >= rank)
        )
        filtered = df[mask].copy()

        # ── Relax: try without branch if nothing ──
        if filtered.empty:
            mask2 = (
                (df["Category_upper"] == c) &
                (df["Cutoff_Rank"] >= rank)
            )
            filtered = df[mask2].copy()

        if filtered.empty:
            return []

        # Keep one row per college (the worst cutoff → most accessible)
        filtered = (
            filtered
            .sort_values("Cutoff_Rank")
            .drop_duplicates(subset=["College_Code", "Branch", "Category"])
        )

        # Closest cutoff to student rank first
        filtered["Gap"] = filtered["Cutoff_Rank"] - rank
        filtered = filtered.sort_values("Gap")

        rows = []
        for _, r in filtered.head(top_n).iterrows():
            rows.append({
                "College_Code": r["College_Code"],
                "College_Name": r.get("College_Name", ""),
                "Branch":       r["Branch"],
                "Category":     r["Category"],
                "Cutoff_Rank":  int(r["Cutoff_Rank"]),
                "Gap":          int(r["Gap"]),
                "Source":       "Exact Lookup",
            })
        return rows

    # ── Stage 2: ML fallback ─────────────────
    def _ml_predict(self, rank, branch, category, top_n):
        enc = self.encoders

        branch_enc   = _safe_transform(enc["branch"],   pd.Series([branch]))[0]
        category_enc = _safe_transform(enc["category"], pd.Series([category]))[0]

        x = np.array([[rank, branch_enc, category_enc, 0]])   # Year=0 (unknown)

        proba = self.rf.predict_proba(x)[0]
        top_indices = np.argsort(proba)[::-1][:top_n]

        rows = []
        for idx in top_indices:
            college_code = enc["college"].classes_[idx]
            prob         = proba[idx]
            if prob < 0.001:
                break

            # Get best display info from master df
            sub = self.df[self.df["College_Code"] == college_code]
            name = sub["College_Name"].iloc[0] if not sub.empty else ""

            rows.append({
                "College_Code": college_code,
                "College_Name": name,
                "Branch":       branch,
                "Category":     category,
                "Cutoff_Rank":  "N/A",
                "Gap":          "N/A",
                "Confidence":   f"{prob*100:.1f}%",
                "Source":       "ML Prediction",
            })
        return rows
