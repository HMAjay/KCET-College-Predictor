import os
import sys
import joblib
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from config import MODEL_PATH, MODEL_DIR
from src.utils.helpers import ensure_dirs, banner
from src.data.transform import load_master, build_encoders, build_features


def train():
    print(banner("Step 4 - Model Training"))

    print("\nLoading master dataset ...")
    df = load_master()
    print(f"   {len(df):,} rows  |  {df['College_Code'].nunique()} colleges")

    print("\nBuilding feature encoders ...")
    encoders = build_encoders(df)
    X, y     = build_features(df, encoders)
    print(f"   X shape : {X.shape}")
    print(f"   Classes : {len(encoders['college'].classes_)}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42
    )
    print(f"\n   Train : {len(X_train):,}  |  Test : {len(X_test):,}")

    print("\nTraining Random Forest ...")
    rf = RandomForestClassifier(
        n_estimators=50,       # reduced from 300 to save memory
        max_depth=20,          # capped depth to save memory
        min_samples_leaf=2,
        class_weight="balanced",
        n_jobs=1,              # single job to avoid memory duplication
        random_state=42,
    )
    rf.fit(X_train, y_train)

    rf_preds = rf.predict(X_test)
    rf_acc   = accuracy_score(y_test, rf_preds)
    print(f"   Test accuracy : {rf_acc:.4f}  ({rf_acc*100:.1f}%)")
    print("   (Note: predictions use exact cutoff lookup primarily - ML is fallback only)")

    ensure_dirs(MODEL_DIR)

    bundle = {
        "encoders":  encoders,
        "rf_model":  rf,
        "master_df": df,
        "feature_cols": ["Cutoff_Rank", "Branch_enc", "Category_enc", "Year"],
        "metadata": {
            "n_colleges":   len(encoders["college"].classes_),
            "n_branches":   len(encoders["branch"].classes_),
            "n_categories": len(encoders["category"].classes_),
            "test_accuracy": round(rf_acc, 4),
            "cv_mean": "skipped",
        }
    }

    joblib.dump(bundle, MODEL_PATH)
    print(banner(f"Model saved -> {MODEL_PATH}"))
    print(f"   Colleges in model : {bundle['metadata']['n_colleges']}")
    print(f"   Test accuracy     : {bundle['metadata']['test_accuracy']}\n")


if __name__ == "__main__":
    train()