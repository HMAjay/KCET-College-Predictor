import os
import sys
import re
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from config import EXTRACTED_DIR, CLEANED_DIR, RAW_DIR
from src.utils.helpers import ensure_dirs, banner

try:
    import pdfplumber
except ImportError:
    print("pdfplumber not installed.")
    sys.exit(1)

KNOWN_CATEGORIES = {
    "1G","1K","1R","2AG","2AK","2AR","2BG","2BK","2BR",
    "3AG","3AK","3AR","3BG","3BK","3BR","GM","GMK","GMR",
    "SCG","SCK","SCR","STG","STK","STR","OBC","OBCK","OBCR",
    "SC","ST"
}


def is_category_header(cells):
    matches = [c for c in cells if str(c).strip().upper() in KNOWN_CATEGORIES]
    return len(matches) >= 3


def extract_college_list(pdf_path: str) -> list:
    """
    Handles two formats:
    2021-2024: '1 E001 University Visveswariah College of Engineering Bangalore'
    2025:      'College: E001 Univesity of Visvesvaraya College of Engineering ...'
    """
    college_list = []
    seen = set()

    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if not text:
                continue
            for line in text.splitlines():
                line = line.strip()

                # Format 1: '1 E001 College Name'  or  'E001 College Name'
                m = re.match(r"^\d*\s*(E\d{3})\s+(.+)", line)
                if m:
                    code = m.group(1).strip()
                    name = m.group(2).strip()
                    if code not in seen:
                        seen.add(code)
                        college_list.append((code, name))
                    continue

                # Format 2: 'College: E001 College Name'
                m2 = re.match(r"^College:\s*(E\d{3})\s+(.+)", line, re.IGNORECASE)
                if m2:
                    code = m2.group(1).strip()
                    name = m2.group(2).strip()
                    if code not in seen:
                        seen.add(code)
                        college_list.append((code, name))

    return college_list


def clean_file(file_path: str, year: int, pdf_path: str) -> pd.DataFrame:
    college_list = extract_college_list(pdf_path)
    print(f"   Found {len(college_list)} colleges in PDF")
    if college_list:
        print(f"   Sample: {college_list[:3]}")

    df = pd.read_csv(file_path, dtype=str, header=None)

    college_idx = -1
    category_col_map = {}
    cleaned_rows = []
    last_was_cat_header = False

    for _, row in df.iterrows():
        cells = [str(c).strip() if str(c).strip() != "nan" else "" for c in row]

        if not any(cells):
            continue

        first = cells[0]

        # Skip the column index row (0, 1, 2, 3 ...)
        if first == "0" and cells[1] == "1" and cells[2] == "2":
            continue

        # Category header row
        if is_category_header(cells):
            if not last_was_cat_header:
                college_idx += 1
            category_col_map = {}
            for i, c in enumerate(cells):
                cu = c.strip().upper()
                if cu in KNOWN_CATEGORIES:
                    category_col_map[i] = cu
            last_was_cat_header = True
            continue

        last_was_cat_header = False

        if not category_col_map:
            continue
        if first in ("", "--", "-"):
            continue
        # Skip pure small-number rows
        if re.match(r"^\d{1,2}$", first):
            continue

        # Resolve college
        if 0 <= college_idx < len(college_list):
            current_code, current_name = college_list[college_idx]
        else:
            continue

        if not current_code or str(current_code).strip().upper() == "UNKNOWN":
            continue
        if not current_name or str(current_name).strip().lower() == "nan":
            continue

        branch_name = first.replace("\n", " ").strip()

        for col_idx, cat_name in category_col_map.items():
            if col_idx >= len(cells):
                continue
            val = cells[col_idx].replace(",", "").strip()
            if val in ("", "--", "-", "None", "nan"):
                continue
            if not re.match(r"^\d+$", val):
                continue

            cleaned_rows.append({
                "College_Code": current_code,
                "College_Name": current_name,
                "Branch":       branch_name,
                "Category":     cat_name,
                "Cutoff_Rank":  int(val),
                "Year":         year,
            })

    return pd.DataFrame(cleaned_rows)


def infer_year(filename: str):
    m = re.search(r"(20\d{2})", filename)
    return int(m.group(1)) if m else None


def infer_pdf_path(csv_filename: str) -> str:
    name = csv_filename.replace("_raw.csv", "").replace("_", " ")
    return os.path.join(RAW_DIR, name + ".pdf")


def clean_all_files():
    ensure_dirs(EXTRACTED_DIR, CLEANED_DIR)

    files = [f for f in os.listdir(EXTRACTED_DIR) if f.endswith("_raw.csv")]

    if not files:
        print("No *_raw.csv files found in data/extracted/")
        return

    print(banner("Step 2 - Data Cleaning"))

    for file in sorted(files):
        print(f"\nCleaning: {file}")
        year = infer_year(file)
        path = os.path.join(EXTRACTED_DIR, file)
        pdf_path = infer_pdf_path(file)

        if not os.path.exists(pdf_path):
            print(f"   WARNING: PDF not found at {pdf_path}")
            continue

        df_clean = clean_file(path, year, pdf_path)

        if df_clean.empty:
            print("   WARNING: No valid rows found.")
            continue

        out_name  = file.replace("_raw.csv", "_clean.csv")
        save_path = os.path.join(CLEANED_DIR, out_name)
        df_clean.to_csv(save_path, index=False)
        print(f"   Saved: {out_name} ({len(df_clean):,} rows)")
        print(f"   Colleges: {df_clean['College_Code'].nunique()}  Branches: {df_clean['Branch'].nunique()}")
        print(f"\n   Sample:")
        print(df_clean[["College_Code","College_Name","Branch","Category","Cutoff_Rank"]].head(8).to_string(index=False))

    print(banner("Cleaning complete"))


if __name__ == "__main__":
    clean_all_files()
