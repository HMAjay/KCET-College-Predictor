#!/usr/bin/env python3
# ─────────────────────────────────────────────
#  run_pipeline.py
#  Run the full data → model pipeline in one go
# ─────────────────────────────────────────────
"""
Usage:
    python run_pipeline.py           # full pipeline
    python run_pipeline.py --skip-extract   # if CSVs already exist
    python run_pipeline.py --predict-only   # jump straight to terminal UI
"""

import sys
import argparse

# ── Arg parsing ───────────────────────────────
parser = argparse.ArgumentParser(description="KCET College Predictor — full pipeline runner")
parser.add_argument("--skip-extract",  action="store_true", help="Skip PDF extraction step")
parser.add_argument("--skip-clean",    action="store_true", help="Skip cleaning step")
parser.add_argument("--skip-merge",    action="store_true", help="Skip merge step")
parser.add_argument("--skip-train",    action="store_true", help="Skip model training step")
parser.add_argument("--predict-only",  action="store_true", help="Launch terminal UI only")
args = parser.parse_args()


# ─────────────────────────────────────────────
def run():
    if args.predict_only:
        from app.main import main
        main()
        return

    if not args.skip_extract:
        print("\n" + "="*60)
        print("  STEP 1 — PDF EXTRACTION")
        print("="*60)
        from src.data.extract_pdf import extract_all_pdfs
        extract_all_pdfs()

    if not args.skip_clean:
        print("\n" + "="*60)
        print("  STEP 2 — DATA CLEANING")
        print("="*60)
        from src.data.clean_data import clean_all_files
        clean_all_files()

    if not args.skip_merge:
        print("\n" + "="*60)
        print("  STEP 3 — MERGING DATASETS")
        print("="*60)
        from src.data.merge_data import merge_all
        merge_all()

    if not args.skip_train:
        print("\n" + "="*60)
        print("  STEP 4 — MODEL TRAINING")
        print("="*60)
        from src.model.train_model import train
        train()

    print("\n" + "="*60)
    print("  ✔  Pipeline complete!  Launching terminal UI …")
    print("="*60 + "\n")

    from app.main import main
    main()


# ─────────────────────────────────────────────
if __name__ == "__main__":
    run()
