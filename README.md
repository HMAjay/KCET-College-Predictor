# 🎓 KCET College Predictor

A terminal-based Machine Learning project that predicts eligible colleges
based on **KCET rank, branch, and category** using historical cutoff data.

---

## 📂 Folder Structure

```
KCET-College-Predictor/
│
├── data/
│   ├── raw/              ← Place your KCET PDFs here
│   ├── extracted/        ← Auto-generated raw CSVs
│   ├── cleaned/          ← Auto-generated clean CSVs
│   └── final/            ← kcet_master.csv (merged dataset)
│
├── src/
│   ├── data/
│   │   ├── __init__.py
│   │   ├── extract_pdf.py    ← Step 1: PDF → CSV
│   │   ├── clean_data.py     ← Step 2: clean raw CSVs
│   │   ├── merge_data.py     ← Step 3: combine years
│   │   └── transform.py      ← Feature engineering (used internally)
│   │
│   ├── model/
│   │   ├── __init__.py
│   │   ├── train_model.py    ← Step 4: train + save model
│   │   └── predictor.py      ← Prediction engine (lookup + ML)
│   │
│   ├── utils/
│   │   ├── __init__.py
│   │   └── helpers.py        ← Shared utilities
│   │
│   └── __init__.py
│
├── models/
│   └── kcet_model.pkl        ← Saved model bundle
│
├── app/
│   └── main.py               ← Terminal UI (the app you run)
│
├── notebooks/
│   └── data_exploration.ipynb
│
├── config.py                 ← Central paths + constants
├── run_pipeline.py           ← One-shot pipeline runner
├── requirements.txt
└── README.md
```

---

## ⚙️ Setup

```bash
# 1. Clone
git clone https://github.com/your-username/kcet-college-predictor.git
cd kcet-college-predictor

# 2. Virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

---

## 🚀 Quick Start

### Full pipeline (PDF → prediction)

```bash
python run_pipeline.py
```

### Step by step

```bash
# Step 1 — Extract tables from PDFs in data/raw/
python src/data/extract_pdf.py

# Step 2 — Clean extracted CSVs
python src/data/clean_data.py

# Step 3 — Merge all years into one master CSV
python src/data/merge_data.py

# Step 4 — Train and save model
python src/model/train_model.py

# Step 5 — Launch terminal predictor
python app/main.py
```

### Skip to prediction (if model already trained)

```bash
python run_pipeline.py --predict-only
```

---

## 🧠 How Predictions Work

The predictor uses a **two-stage engine**:

1. **Direct Lookup** — Queries the master cutoff table for colleges where  
   `cutoff_rank ≥ student_rank` in the given branch + category.  
   Results are sorted by the smallest gap (safest pick first).

2. **ML Fallback** — If no exact match is found (e.g. rare category),  
   a Random Forest model trained on all historical data predicts  
   the most likely colleges by probability.

The model bundle (`models/kcet_model.pkl`) stores:
- Label encoders for Branch, Category, College
- Trained `RandomForestClassifier` (300 trees, balanced classes)
- The full master DataFrame for fast lookup

---

## 🖥 Terminal UI Example

```
════════════════════════════════════════════════════════════
  🎓  KCET College Predictor
════════════════════════════════════════════════════════════

  Enter your KCET Rank: 12500
  Enter Branch: Computer Science Engineering
  Enter Category: GM
  How many results? [Enter for 10]: 

  ✔  8 college(s) found:

  ╭────┬──────┬──────────────────────────────┬────────────────┬─────────┬─────╮
  │  # │ Code │ College Name                 │ Branch         │ Cutoff  │ Gap │
  ├────┼──────┼──────────────────────────────┼────────────────┼─────────┼─────┤
  │  1 │ E123 │ ABC Institute of Technology  │ Computer Sci…  │ 12610   │ 110 │
  │  2 │ E045 │ XYZ Engineering College      │ Computer Sci…  │ 13200   │ 700 │
  ...
  ╰────┴──────┴──────────────────────────────┴────────────────┴─────────┴─────╯
```

---

## 📊 Technologies

| Layer        | Tool                          |
|--------------|-------------------------------|
| PDF Extract  | `tabula-py`                   |
| Data Wrangling | `pandas`, `numpy`           |
| ML Model     | `scikit-learn` RandomForest   |
| Model Storage | `joblib`                     |
| Terminal UI  | `colorama`, `tabulate`        |

---

## 🚀 Future Improvements

- Web interface (FastAPI + React)
- Cutoff trend charts per college
- Seat availability integration
- Category-wise separate models
- Docker + cloud deployment
