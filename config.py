# ─────────────────────────────────────────────
#  config.py  —  Central configuration
# ─────────────────────────────────────────────
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Data paths ────────────────────────────────
RAW_DIR       = os.path.join(BASE_DIR, "data", "raw")
EXTRACTED_DIR = os.path.join(BASE_DIR, "data", "extracted")
CLEANED_DIR   = os.path.join(BASE_DIR, "data", "cleaned")
FINAL_DIR     = os.path.join(BASE_DIR, "data", "final")

MASTER_CSV    = os.path.join(FINAL_DIR, "kcet_master.csv")

# ── Model path ────────────────────────────────
MODEL_DIR     = os.path.join(BASE_DIR, "models")
MODEL_PATH    = os.path.join(MODEL_DIR, "kcet_model.pkl")

# ── Known categories (KCET standard) ─────────
CATEGORIES = [
    "GM", "GMK", "GMR",
    "SC", "SCK", "SCR",
    "ST", "STK", "STR",
    "OBC", "OBCK", "OBCR",
    "1G", "1GK", "1GR",
    "2AG", "2AGK", "2AGR",
    "2BG", "2BGK", "2BGR",
    "3AG", "3AGK", "3AGR",
    "3BG", "3BGK", "3BGR",
]

# ── Common KCET branches ──────────────────────
BRANCHES = [
    "Computer Science Engineering",
    "Information Science Engineering",
    "Electronics and Communication Engineering",
    "Electrical and Electronics Engineering",
    "Mechanical Engineering",
    "Civil Engineering",
    "Chemical Engineering",
    "Biotechnology",
    "Aerospace Engineering",
    "Industrial Engineering",
    "Instrumentation Technology",
    "Medical Electronics",
]
