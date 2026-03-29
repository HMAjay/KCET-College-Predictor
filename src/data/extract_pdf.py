import os
import sys
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from config import RAW_DIR, EXTRACTED_DIR
from src.utils.helpers import ensure_dirs, banner

try:
    import pdfplumber
except ImportError:
    print("pdfplumber not installed. Run:  pip install pdfplumber")
    sys.exit(1)


def extract_pdf(pdf_path: str):
    all_rows = []

    with pdfplumber.open(pdf_path) as pdf:
        total = len(pdf.pages)
        for i, page in enumerate(pdf.pages, 1):

            # Extract raw text lines first to catch college header rows
            text = page.extract_text()
            text_lines = []
            if text:
                for line in text.splitlines():
                    line = line.strip()
                    if line:
                        text_lines.append(line)

            # Extract structured tables
            tables = page.extract_tables()

            if tables:
                # Find college header lines from text (E001, E002 etc.)
                import re
                for line in text_lines:
                    if re.match(r"^E\d{3}\b", line):
                        all_rows.append(["COLLEGE_HEADER", line] + [""] * 23)

                for table in tables:
                    for row in table:
                        if row and any(cell for cell in row):
                            cleaned = [str(c).strip().replace("\n", " ") if c else "" for c in row]
                            all_rows.append(cleaned)
            else:
                import re
                for line in text_lines:
                    if re.match(r"^E\d{3}\b", line):
                        all_rows.append(["COLLEGE_HEADER", line] + [""] * 23)
                    else:
                        parts = line.split()
                        if parts:
                            all_rows.append(parts)

            if i % 20 == 0 or i == total:
                print(f"   ... page {i}/{total}", end="\r")

    print()

    if not all_rows:
        return None

    max_cols = max(len(r) for r in all_rows)
    padded = [r + [""] * (max_cols - len(r)) for r in all_rows]
    return pd.DataFrame(padded)


def extract_all_pdfs():
    ensure_dirs(RAW_DIR, EXTRACTED_DIR)

    pdf_files = [f for f in os.listdir(RAW_DIR) if f.lower().endswith(".pdf")]

    if not pdf_files:
        print("No PDF files found in data/raw/")
        return

    print(banner("Step 1 - PDF Extraction"))

    ok = fail = 0
    for pdf_file in sorted(pdf_files):
        print(f"\nProcessing: {pdf_file}")
        pdf_path = os.path.join(RAW_DIR, pdf_file)

        try:
            df = extract_pdf(pdf_path)
        except Exception as e:
            print(f"   Error: {e}")
            fail += 1
            continue

        if df is None or df.empty:
            print("   No content extracted - skipping.")
            fail += 1
            continue

        csv_name = pdf_file.replace(".pdf", "_raw.csv").replace(" ", "_")
        output_path = os.path.join(EXTRACTED_DIR, csv_name)
        df.to_csv(output_path, index=False)
        print(f"   Saved: {csv_name} ({len(df):,} rows)")
        ok += 1

    print(banner(f"Done - {ok} extracted, {fail} failed"))


if __name__ == "__main__":
    extract_all_pdfs()