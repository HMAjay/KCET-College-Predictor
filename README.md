# KCET College Predictor

This project builds a **2026 cutoff predictor** from KCET cutoff data for
**2021-2025**. Instead of returning mixed historical rows, it projects a single
cutoff for each college, branch, and category combination, then compares your
rank against that projected value.

## How It Works

1. PDFs in `data/raw/` are extracted into raw CSVs.
2. Raw CSVs are cleaned into yearly cutoff tables.
3. Clean yearly files are merged into `data/final/kcet_master.csv`.
4. `src/model/train_model.py` reads the master CSV and builds a trend-based
   2026 model bundle.
5. `app/main.py` loads the saved model and shows projected colleges in the
   terminal.

The saved bundle at `models/kcet_model.pkl` contains:
- `trend_df`: the projected 2026 cutoff table
- `metadata`: year range, target year, and dataset counts

## Project Structure

Note: if the tree below renders oddly in your terminal, use the folder list in
the repository explorer. The main runtime files are `app/main.py`,
`src/model/predictor.py`, `src/model/train_model.py`, `src/data/transform.py`,
`run_pipeline.py`, and `view_model.py`.

```text
KCET-College-Predictor/
├── app/
│   └── main.py
├── data/
│   ├── raw/
│   ├── extracted/
│   ├── cleaned/
│   └── final/
├── models/
├── notebooks/
├── src/
│   ├── data/
│   │   ├── extract_pdf.py
│   │   ├── clean_data.py
│   │   ├── merge_data.py
│   │   └── transform.py
│   └── model/
│       ├── predictor.py
│       └── train_model.py
├── config.py
├── run_pipeline.py
├── view_model.py
└── requirements.txt
```

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Run The Project

Full rebuild from merged data onward:

```bash
python src/model/train_model.py
python app/main.py
```

Full pipeline from PDFs:

```bash
python run_pipeline.py
```

Run only the app with the latest saved model:

```bash
python app/main.py
```

## Inspect The Saved Model

Use the helper script:

```bash
python view_model.py
```

Show more projected rows:

```bash
python view_model.py --rows 20
```

## What The App Shows

Each result row represents a projected 2026 cutoff, not a raw historical row.

- `2026 Predicted`: estimated cutoff for 2026
- `2025 Cutoff`: latest 2025 cutoff when available, otherwise the latest available cutoff
- `Gap`: `predicted cutoff - your rank`
- `College / Course`: college code plus the resolved branch code or label

Smaller positive gaps are closer, more competitive options. Larger gaps are safer.

## Supported Category Codes

`GM`, `GMK`, `GMR`, `SCG`, `SCK`, `SCR`, `STG`, `STK`, `STR`,
`1G`, `1K`, `1R`, `2AG`, `2AK`, `2AR`, `2BG`, `2BK`, `2BR`,
`3AG`, `3AK`, `3AR`, `3BG`, `3BK`, `3BR`

The app also resolves common branch aliases like `CSE`, `CSE-AIML`, `AIML`, `ISE`, `IE`, `ECE`, `ETCE`,
`AI`, and `Civil`.

## Latest Improvements

- Better branch alias resolution for short inputs like `CSE`, `CSE-AIML`, `AIML`, `ISE`, `IE`, `ECE`, and `ETCE`.
- Consistent branch display codes in results, so common branches now show expected labels like `CSE`, `ISE`, `ECE`, `AIML`, and `ETCE`.
- Cleaner terminal prompts and safer exit handling for `quit`, `Ctrl+C`, and `Ctrl+D`.
- The UI reads model metadata from the saved bundle and shows covered colleges and trend years on startup.

## Notes

- If you change the source data, rerun `python src/model/train_model.py`.
- Predictions are only as strong as the available history.
- Combinations with more historical years are generally more reliable than one-year combinations.
- Some rows may fall back to the latest available cutoff when 2025 data is not present for that exact branch-category combination.

## Technologies

- `pdfplumber` for PDF extraction
- `pandas` and `numpy` for cleaning and trend projection
- `joblib` for model bundle storage
- `colorama` and `tabulate` for terminal output
