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

# ── Categories currently present in the merged dataset ──
CATEGORIES = [
    "GM", "GMK", "GMR",
    "SCG", "SCK", "SCR",
    "STG", "STK", "STR",
    "1G", "1K", "1R",
    "2AG", "2AK", "2AR",
    "2BG", "2BK", "2BR",
    "3AG", "3AK", "3AR",
    "3BG", "3BK", "3BR",
]

# ── Common branch labels seen in the merged dataset ──
BRANCHES = [
    "CS Computers",
    "COMPUTER SCIENCE AND ENGINEERING",
    "IE Info.Science",
    "INFORMATION SCIENCE AND ENGINEERING",
    "EC Electronics",
    "ELECTRONICS AND COMMUNICATIO N ENGG",
    "EE Electrical",
    "ELECTRICAL & ELECTRONICS ENGINEERING",
    "CE Civil",
    "CIVIL ENGINEERING",
    "ME Mechanical",
    "AI Artificial Intelligence",
]
