import os
import sys

import joblib

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from config import MODEL_PATH, MODEL_DIR
from src.utils.helpers import ensure_dirs, banner
from src.data.transform import load_master
from src.model.predictor import KCETPredictor


def train():
    print(banner("Step 4 - Model Training"))

    print("\nLoading master dataset ...")
    df = load_master()
    year_values = sorted(df["Year"].dropna().astype(int).unique().tolist())
    target_year = (max(year_values) + 1) if year_values else 2026

    print(f"   {len(df):,} rows  |  {df['College_Code'].nunique()} colleges")
    print(f"   Historical years : {year_values}")
    print(f"   Target year      : {target_year}")
    print("\nBuilding trend-ready model bundle ...")

    trend_builder = object.__new__(KCETPredictor)
    trend_builder.df = df.copy()
    trend_builder.target_year = target_year
    trend_df = KCETPredictor._build_trend_predictions(trend_builder)
    print(f"   Trend rows       : {len(trend_df):,}")

    ensure_dirs(MODEL_DIR)

    bundle = {
        "trend_df": trend_df,
        "metadata": {
            "n_colleges": int(df["College_Code"].nunique()),
            "n_branches": int(df["Branch"].nunique()),
            "n_categories": int(df["Category"].nunique()),
            "min_year": min(year_values) if year_values else None,
            "max_year": max(year_values) if year_values else None,
            "target_year": target_year,
            "model_type": "trend_projection",
        }
    }

    joblib.dump(bundle, MODEL_PATH)
    print(banner(f"Model saved -> {MODEL_PATH}"))
    print(f"   Colleges in model : {bundle['metadata']['n_colleges']}")
    print(f"   Target year       : {bundle['metadata']['target_year']}")
    print("   Prediction mode   : trend projection\n")


if __name__ == "__main__":
    train()
