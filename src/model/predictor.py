# ─────────────────────────────────────────────
#  src/model/predictor.py
#  Core prediction engine  (used by app/main.py)
# ─────────────────────────────────────────────
"""
Trend-based prediction for the next admission cycle.

The predictor builds a single projected cutoff for 2026 for each
college + branch + category combination using its historical trend
from the merged KCET cutoff data.
"""

import os
import re
import sys
from difflib import SequenceMatcher

import joblib
import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from config import MODEL_PATH
from src.data.transform import load_master


BRANCH_ALIASES = {
    "cse": "computer science engineering",
    "cs": "computer science",
    "ie": "information science engineering",
    "ise": "information science engineering",
    "ece": "electronics communication engineering",
    "ec": "electronics communication",
    "eee": "electrical electronics engineering",
    "ee": "electrical electronics",
    "ce": "civil engineering",
    "me": "mechanical engineering",
    "ai": "artificial intelligence",
    "aiml": "artificial intelligence and machine learning",
    "ml": "machine learning",
    "etce": "electronics telecommunication engineering",
    "telecommunic": "telecommunication",
    "instrumentati": "instrumentation",
    "bio": "biotechnology",
    "it": "information technology",
    "engg": "engineering",
    "engg.": "engineering",
    "comp": "computer",
    "comp.": "computer",
    "sc": "science",
    "sc.": "science",
    "comm": "communication",
    "comm.": "communication",
    "omputer": "computer",
    "artificia": "artificial",
    "machi": "machine",
    "learni": "learning",
    "engg(": "engineering",
}

PREFERRED_BRANCH_CODES = {
    "computer science engineering": "CSE",
    "information science engineering": "ISE",
    "computer science engineering artificial intelligence and machine learning": "CSE-AIML",
    "b tech in computer science engineering artificial intelligence and machine learning": "CSE-AIML",
    "artificial intelligence and machine learning": "AIML",
    "electronics and communication engineering": "ECE",
    "electronics and telecommunication engineering": "ETCE",
    "electronics and instrumentation engineering": "EIE",
    "electrical and electronics engineering": "EEE",
    "mechanical engineering": "ME",
    "civil engineering": "CE",
    "artificial intelligence": "AI",
    "biotechnology": "BT",
    "computer science and design": "CSD",
    "computer science and business systems": "CSBS",
    "aeronautical engineering": "AE",
}

STOPWORDS = {
    "and",
    "in",
    "of",
    "the",
    "for",
    "with",
    "tech",
    "b",
}

WORD_JOIN_CANDIDATES = {
    "artificial",
    "computer",
    "engineering",
    "information",
    "electronics",
    "electrical",
    "communication",
    "intelligence",
    "learning",
    "mechanical",
}

INVALID_BRANCH_VALUES = {
    "college:",
    "generated",
}
NON_STANDARD_CODE_PREFIXES = {
    "b tech",
    "b.tech",
    "btech",
    "bachelor",
}


def _repair_broken_tokens(tokens: list[str]) -> list[str]:
    repaired = []
    i = 0
    while i < len(tokens):
        token = tokens[i]
        next_token = tokens[i + 1] if i + 1 < len(tokens) else ""

        if next_token and len(next_token) == 1:
            merged = token + next_token
            if merged in WORD_JOIN_CANDIDATES:
                repaired.append(merged)
                i += 2
                continue

        repaired.append(token)
        i += 1

    return repaired


def _normalize_text(value: str) -> str:
    text = str(value).strip().lower().replace("&", " and ")
    text = re.sub(r"[()/_,-]+", " ", text)
    text = re.sub(r"\s+", " ", text)

    expanded_tokens = []
    for token in _repair_broken_tokens(text.split()):
        expanded_tokens.extend(BRANCH_ALIASES.get(token, token).split())
    return " ".join(expanded_tokens)


def _extract_branch_code(value: str) -> str:
    text = str(value).strip()
    match = re.match(r"^([A-Z]{1,3})\b(?:\s|$)", text)
    if not match:
        return ""

    code = match.group(1)
    if code in {"B"}:
        return ""
    return code


def _strip_branch_code(value: str) -> str:
    text = str(value).strip()
    code = _extract_branch_code(text)
    if not code:
        return text
    return text[len(code):].strip()


def _extract_preferred_branch_code(value: str) -> str:
    text = str(value).strip()
    code = _extract_branch_code(text)
    if not code:
        return ""

    remainder = _strip_branch_code(text).lower()
    if any(remainder.startswith(prefix) for prefix in NON_STANDARD_CODE_PREFIXES):
        return ""
    return code


def _tokenize(value: str) -> set[str]:
    return {
        token
        for token in _normalize_text(value).split()
        if len(token) > 1 and token not in STOPWORDS
    }


def _canonical_branch(value: str) -> str:
    raw_value = str(value).strip()
    stripped_value = _strip_branch_code(raw_value)
    if stripped_value:
        normalized = _normalize_text(stripped_value)
        tokens = _tokenize(stripped_value)
    else:
        normalized = _normalize_text(raw_value)
        tokens = _tokenize(raw_value)

    if raw_value.lower() in INVALID_BRANCH_VALUES or stripped_value.lower() in INVALID_BRANCH_VALUES:
        return ""

    if (
        {"computer", "science"} <= tokens and
        {"artificial", "intelligence"} <= tokens and
        {"machine", "learning"} <= tokens
    ):
        return "Computer Science Engineering Artificial Intelligence and Machine Learning"
    if {"electronics", "telecommunication", "engineering"} <= tokens:
        return "Electronics and Telecommunication Engineering"
    if {"artificial", "intelligence", "machine", "learning"} <= tokens:
        return "Artificial Intelligence and Machine Learning"
    if {"information", "science"} <= tokens:
        return "Information Science Engineering"
    if {"computer", "science", "engineering"} <= tokens:
        return "Computer Science Engineering"
    if {"electronics", "telecommunication", "engineering"} <= tokens:
        return "Electronics and Telecommunication Engineering"
    if {"electronics", "instrumentation", "engineering"} <= tokens:
        return "Electronics and Instrumentation Engineering"
    if {"electronics", "communication"} <= tokens:
        return "Electronics and Communication Engineering"
    if {"electrical", "electronics"} <= tokens:
        return "Electrical and Electronics Engineering"
    if {"civil", "engineering"} <= tokens:
        return "Civil Engineering"
    if {"biotechnology"} <= tokens:
        return "Biotechnology"
    if {"mechanical", "engineering"} <= tokens:
        return "Mechanical Engineering"
    if {"artificial", "intelligence"} <= tokens and "computer" not in tokens:
        return "Artificial Intelligence"

    if normalized == "information science engineering":
        return "Information Science Engineering"
    if normalized == "computer science engineering":
        return "Computer Science Engineering"
    if normalized == "computer science engineering artificial intelligence and machine learning":
        return "B Tech in Computer Science Engineering Artificial Intelligence and Machine Learning"
    if normalized == "electronics telecommunication engineering":
        return "Electronics and Telecommunication Engineering"
    if normalized == "electronics communication engineering":
        return "Electronics and Communication Engineering"
    if normalized == "electrical electronics engineering":
        return "Electrical and Electronics Engineering"
    if normalized == "civil engineering":
        return "Civil Engineering"
    if normalized == "mechanical engineering":
        return "Mechanical Engineering"
    if normalized == "artificial intelligence":
        return "Artificial Intelligence"
    if normalized == "artificial intelligence and machine learning":
        return "Artificial Intelligence and Machine Learning"

    cleaned_value = stripped_value or raw_value
    return cleaned_value.strip()


def _display_college_name(names: pd.Series) -> str:
    clean_names = [
        str(name).strip()
        for name in names.dropna().astype(str)
        if str(name).strip() and str(name).strip().lower() != "nan"
    ]
    if not clean_names:
        return ""
    clean_names = sorted(set(clean_names), key=lambda name: (len(name), name))
    return clean_names[0]


def _preferred_branch_code(branch_name: str, branch_code: str = "") -> str:
    display_code = PREFERRED_BRANCH_CODES.get(branch_name.lower().strip())
    if display_code:
        return display_code
    return str(branch_code).strip()


def _code_candidate_score(raw_branch: str, canonical_branch: str) -> float:
    raw_tokens = _tokenize(raw_branch)
    canonical_tokens = _tokenize(canonical_branch)
    if not canonical_tokens:
        return 0.0

    overlap = len(raw_tokens & canonical_tokens)
    coverage = overlap / len(canonical_tokens)
    ratio = SequenceMatcher(None, _normalize_text(raw_branch), _normalize_text(canonical_branch)).ratio()
    return coverage * 0.7 + ratio * 0.3


# ─────────────────────────────────────────────
class KCETPredictor:

    def __init__(self):
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"Model not found at {MODEL_PATH}.\n"
                "Run:  python src/model/train_model.py"
            )
        bundle = joblib.load(MODEL_PATH)

        self.metadata = bundle.get("metadata", {})
        self.target_year = int(self.metadata.get("target_year", 2026))
        prebuilt_trend_df = bundle.get("trend_df")
        if prebuilt_trend_df is not None:
            self.trend_df = prebuilt_trend_df.copy()
        else:
            self.df = load_master().copy()
            self.df["Category_upper"] = self.df["Category"].str.upper()
            self.df["Branch_norm"] = self.df["Branch"].map(_normalize_text)
            self.df["Canonical_Branch"] = self.df["Branch"].map(_canonical_branch)
            self.trend_df = self._build_trend_predictions()

        self.trend_df = self._stabilize_loaded_trend_df(self.trend_df)
        self._ensure_source_df()
        self.trend_df["Branch"] = self.trend_df["Branch"].map(_canonical_branch)
        self.trend_df["Category_upper"] = self.trend_df["Category"].astype(str).str.upper()
        self.latest_data_year = int(
            self.metadata.get(
                "max_year",
                self.trend_df["Latest_Year"].dropna().max() if not self.trend_df.empty else self.target_year - 1,
            )
        )
        self.active_trend_df = self.trend_df[
            self.trend_df["Latest_Year"] == self.latest_data_year
        ].copy()
        self.active_trend_df = self._attach_branch_catalog(self.active_trend_df)
        self.available_categories = sorted(
            self.active_trend_df["Category_upper"].dropna().unique().tolist()
        )
        branch_counts = self.active_trend_df["Branch"].value_counts()
        self.common_branches = branch_counts.head(15).index.tolist()
        branch_options = (
            self.active_trend_df[["Branch", "Branch_Code", "Branch_Option"]]
            .drop_duplicates()
            .assign(_count=lambda df: df["Branch"].map(branch_counts))
            .sort_values(["_count", "Branch_Option"], ascending=[False, True])
        )
        self.branch_option_map = dict(
            zip(branch_options["Branch"], branch_options["Branch_Option"])
        )
        self.common_branch_options = branch_options["Branch_Option"].head(15).tolist()
        alias_lookup = (
            self.df.groupby("Canonical_Branch", dropna=False)["Branch"]
            .apply(lambda s: sorted({str(v).strip() for v in s if str(v).strip()}))
            .to_dict()
        )
        self._branch_index = []
        for _, row in branch_options.iterrows():
            branch_name = row["Branch"]
            branch_code = row.get("Branch_Code", "")
            branch_option = row.get("Branch_Option", branch_name)
            aliases = alias_lookup.get(branch_name, [])
            core_search_text = " ".join(part for part in [branch_code, branch_name] if part)
            alias_tokens = set()
            alias_codes = set()
            for alias in aliases:
                alias_tokens.update(_tokenize(alias))
                code = _extract_branch_code(alias)
                if code:
                    alias_codes.add(code.lower())
            self._branch_index.append(
                {
                    "name": branch_name,
                    "code": branch_code,
                    "label": branch_option,
                    "normalized": _normalize_text(core_search_text),
                    "tokens": _tokenize(core_search_text),
                    "alias_tokens": alias_tokens,
                    "alias_codes": alias_codes,
                }
            )

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
        return self._predict_from_trends(rank, branch, category, top_n)

    def resolve_branch_matches(self, branch: str, limit: int = 5) -> list[str]:
        query_norm = _normalize_text(branch)
        query_tokens = _tokenize(branch)
        query_text = str(branch).strip()
        is_short_code_query = bool(re.fullmatch(r"[A-Za-z]{1,3}", query_text))
        if not query_norm:
            return []

        scored = []
        for item in self._branch_index:
            available_tokens = item["tokens"] | item.get("alias_tokens", set())
            token_overlap = len(query_tokens & available_tokens)
            query_coverage = token_overlap / len(query_tokens) if query_tokens else 0.0
            branch_coverage = token_overlap / len(item["tokens"]) if item["tokens"] else 0.0
            ratio = SequenceMatcher(None, query_norm, item["normalized"]).ratio()
            extra_tokens = len(item["tokens"] - query_tokens)
            code_bonus = 0.0
            known_codes = set(item.get("alias_codes", set()))
            if item["code"]:
                known_codes.add(item["code"].lower())
            query_code = str(branch).strip().lower()
            exact_code_match = False
            if known_codes:
                if query_code in known_codes:
                    code_bonus = 0.85
                    exact_code_match = True
                elif any(code in str(branch).lower().split() for code in known_codes):
                    code_bonus = 0.30

            bonus = 0.0
            if query_norm == item["normalized"]:
                bonus = 1.0

            score = (
                ratio * 0.30 +
                query_coverage * 0.45 +
                branch_coverage * 0.25 +
                code_bonus +
                bonus -
                (extra_tokens * 0.04)
            )

            if exact_code_match or token_overlap or ratio >= 0.55:
                scored.append((score, item["name"]))

        if not scored:
            return []

        scored.sort(key=lambda pair: (-pair[0], pair[1]))
        best_score = scored[0][0]
        threshold = (
            max(0.35, best_score - 0.10)
            if is_short_code_query
            else max(0.75, best_score - 0.18)
        )
        return [name for score, name in scored if score >= threshold][:limit]

    def _predict_from_trends(self, rank, branch, category, top_n):
        c = category.strip().upper()
        branch_matches = self.resolve_branch_matches(branch)
        if c not in self.available_categories or not branch_matches:
            return []

        filtered = self.active_trend_df[
            (self.active_trend_df["Branch"].isin(branch_matches)) &
            (self.active_trend_df["Category_upper"] == c) &
            (self.active_trend_df["Predicted_Cutoff"] >= rank)
        ].copy()

        if filtered.empty:
            return []

        filtered["Gap"] = filtered["Predicted_Cutoff"] - rank
        filtered = filtered.sort_values(
            ["Gap", "Cutoff_2025"],
            ascending=[True, False],
            na_position="last",
        )

        rows = []
        for _, r in filtered.head(top_n).iterrows():
            rows.append({
                "College_Code": r["College_Code"],
                "College_Name": r["College_Name"],
                "Branch": r["Branch"],
                "Branch_Code": r.get("Branch_Code", ""),
                "Branch_Option": r.get("Branch_Option", r["Branch"]),
                "Category": r["Category"],
                "Cutoff_Rank": int(r["Predicted_Cutoff"]),
                "Gap": int(r["Gap"]),
                "Cutoff_2025": (
                    int(r["Cutoff_2025"])
                    if pd.notna(r["Cutoff_2025"])
                    else int(r["Latest_Cutoff"])
                ),
                "Source": f"{self.target_year} Trend",
            })
        return rows

    def _stabilize_loaded_trend_df(self, trend_df: pd.DataFrame) -> pd.DataFrame:
        if trend_df.empty or "Predicted_Cutoff" not in trend_df.columns:
            return trend_df

        stabilized = trend_df.copy()
        anchor = stabilized["Cutoff_2025"].where(
            stabilized["Cutoff_2025"].notna(),
            stabilized["Latest_Cutoff"],
        )
        lower_bound = (anchor * 0.80).round()
        upper_bound = (anchor * 1.35).round()
        valid_anchor = anchor.notna()

        stabilized.loc[valid_anchor, "Predicted_Cutoff"] = (
            stabilized.loc[valid_anchor, "Predicted_Cutoff"]
            .clip(
                lower=lower_bound[valid_anchor],
                upper=upper_bound[valid_anchor],
            )
            .round()
            .astype(int)
        )
        return stabilized

    def _ensure_source_df(self) -> None:
        if hasattr(self, "df"):
            source_df = self.df.copy()
        else:
            source_df = load_master().copy()

        source_df["Category_upper"] = source_df["Category"].astype(str).str.upper()
        source_df["Canonical_Branch"] = source_df["Branch"].map(_canonical_branch)
        source_df["Branch_Code"] = source_df["Branch"].map(_extract_branch_code)
        source_df["Branch_Code_Pref"] = source_df["Branch"].map(_extract_preferred_branch_code)
        self.df = source_df[source_df["Canonical_Branch"].astype(str).str.strip() != ""].copy()

    def _attach_branch_catalog(self, trend_df: pd.DataFrame) -> pd.DataFrame:
        if trend_df.empty:
            return trend_df

        branch_source_df = self.df[
            self.df["Canonical_Branch"].astype(str).str.strip() != ""
        ].copy()
        if branch_source_df.empty:
            enriched = trend_df.copy()
            enriched["Branch_Code"] = enriched.get("Branch_Code", "").fillna("") if "Branch_Code" in enriched else ""
            enriched["Branch_Option"] = enriched["Branch"]
            return enriched

        code_candidates = branch_source_df[
            branch_source_df["Branch_Code_Pref"].astype(str).str.strip() != ""
        ].copy()
        code_candidates["Branch_Code_Remaining_Len"] = code_candidates["Branch"].map(
            lambda value: len(_strip_branch_code(value))
        )
        code_candidates["Branch_Code_Score"] = code_candidates.apply(
            lambda row: _code_candidate_score(row["Branch"], row["Canonical_Branch"]),
            axis=1,
        )

        if not code_candidates.empty:
            preferred_codes = (
                code_candidates[["Canonical_Branch", "Branch_Code_Pref", "Branch_Code_Remaining_Len", "Branch_Code_Score"]]
                .sort_values(
                    ["Canonical_Branch", "Branch_Code_Score", "Branch_Code_Remaining_Len", "Branch_Code_Pref"],
                    ascending=[True, False, True, True],
                )
                .drop_duplicates(subset=["Canonical_Branch"])
                .rename(columns={"Branch_Code_Pref": "Preferred_Branch_Code"})
            )
        else:
            preferred_codes = pd.DataFrame(columns=["Canonical_Branch", "Preferred_Branch_Code"])

        fallback_codes = (
            branch_source_df.groupby("Canonical_Branch", dropna=False)
            .agg(
                Branch_Code=(
                    "Branch_Code",
                    lambda s: (
                        pd.Series([str(v).strip() for v in s if str(v).strip()])
                        .value_counts()
                        .index[0]
                        if any(str(v).strip() for v in s)
                        else ""
                    ),
                ),
            )
            .reset_index()
        )

        branch_meta = fallback_codes.merge(
            preferred_codes,
            on="Canonical_Branch",
            how="left",
        ).rename(columns={"Canonical_Branch": "Branch"})
        enriched = trend_df.merge(branch_meta, on="Branch", how="left", suffixes=("", "_meta"))
        enriched["Branch_Code"] = (
            enriched["Preferred_Branch_Code"]
            .where(enriched["Preferred_Branch_Code"].astype(str).str.strip() != "", enriched["Branch_Code"])
            .fillna("")
        )
        enriched.drop(columns=["Preferred_Branch_Code"], inplace=True)
        enriched["Branch_Code"] = enriched.apply(
            lambda row: _preferred_branch_code(row["Branch"], row["Branch_Code"]),
            axis=1,
        )
        enriched["Branch_Option"] = enriched.apply(
            lambda row: f"{row['Branch_Code']} - {row['Branch']}"
            if str(row["Branch_Code"]).strip()
            else row["Branch"],
            axis=1,
        )
        return enriched

    def _build_trend_predictions(self) -> pd.DataFrame:
        if "Canonical_Branch" not in self.df.columns:
            self.df["Canonical_Branch"] = self.df["Branch"].map(_canonical_branch)
        delta_lookup = self._build_delta_lookup()
        rows = []

        group_cols = ["College_Code", "Canonical_Branch", "Category"]
        for keys, group in self.df.groupby(group_cols, dropna=False):
            college_code, branch, category = keys
            college_name = _display_college_name(group["College_Name"])
            clean_group = (
                group[["Year", "Cutoff_Rank"]]
                .dropna()
                .sort_values("Year")
                .groupby("Year", as_index=False)["Cutoff_Rank"]
                .max()
            )
            if clean_group.empty:
                continue

            years = clean_group["Year"].astype(int).to_numpy()
            cutoffs = clean_group["Cutoff_Rank"].astype(float).to_numpy()
            cutoff_2025 = clean_group.loc[clean_group["Year"] == 2025, "Cutoff_Rank"]
            predicted_cutoff = self._estimate_2026_cutoff(
                branch=branch,
                category=category,
                years=years,
                cutoffs=cutoffs,
                delta_lookup=delta_lookup,
            )

            rows.append({
                "College_Code": college_code,
                "College_Name": college_name,
                "Branch": branch,
                "Category": category,
                "Category_upper": str(category).upper(),
                "Predicted_Cutoff": int(round(predicted_cutoff)),
                "Cutoff_2025": (
                    float(cutoff_2025.iloc[0])
                    if not cutoff_2025.empty else np.nan
                ),
                "Latest_Cutoff": int(round(cutoffs[-1])),
                "Latest_Year": int(years[-1]),
            })

        return pd.DataFrame(rows)

    def _build_delta_lookup(self) -> dict:
        if "Canonical_Branch" not in self.df.columns:
            self.df["Canonical_Branch"] = self.df["Branch"].map(_canonical_branch)
        branch_category_deltas = {}
        branch_deltas = {}
        category_deltas = {}
        global_deltas = []
        branch_category_pct = {}
        branch_pct = {}
        category_pct = {}
        global_pct = []

        group_cols = ["College_Code", "Canonical_Branch", "Category"]
        for _, group in self.df.groupby(group_cols, dropna=False):
            clean_group = (
                group[["Year", "Cutoff_Rank", "Canonical_Branch", "Category"]]
                .dropna()
                .sort_values("Year")
                .groupby(["Year", "Canonical_Branch", "Category"], as_index=False)["Cutoff_Rank"]
                .max()
            )
            if len(clean_group) < 2:
                continue

            previous_cutoff = float(clean_group["Cutoff_Rank"].iloc[-2])
            current_cutoff = float(clean_group["Cutoff_Rank"].iloc[-1])
            delta = current_cutoff - previous_cutoff
            pct_change = np.clip(
                delta / max(previous_cutoff, 1.0),
                -0.35,
                0.35,
            )
            branch = clean_group["Canonical_Branch"].iloc[-1]
            category = clean_group["Category"].iloc[-1]

            branch_category_deltas.setdefault((branch, category), []).append(delta)
            branch_deltas.setdefault(branch, []).append(delta)
            category_deltas.setdefault(category, []).append(delta)
            global_deltas.append(delta)
            branch_category_pct.setdefault((branch, category), []).append(float(pct_change))
            branch_pct.setdefault(branch, []).append(float(pct_change))
            category_pct.setdefault(category, []).append(float(pct_change))
            global_pct.append(float(pct_change))

        return {
            "branch_category": {key: float(np.median(values)) for key, values in branch_category_deltas.items()},
            "branch": {key: float(np.median(values)) for key, values in branch_deltas.items()},
            "category": {key: float(np.median(values)) for key, values in category_deltas.items()},
            "global": float(np.median(global_deltas)) if global_deltas else 0.0,
            "branch_category_pct": {key: float(np.median(values)) for key, values in branch_category_pct.items()},
            "branch_pct": {key: float(np.median(values)) for key, values in branch_pct.items()},
            "category_pct": {key: float(np.median(values)) for key, values in category_pct.items()},
            "global_pct": float(np.median(global_pct)) if global_pct else 0.0,
        }

    def _default_delta(self, branch: str, category: str, delta_lookup: dict) -> float:
        return (
            delta_lookup["branch_category"].get((branch, category))
            or delta_lookup["branch"].get(branch)
            or delta_lookup["category"].get(category)
            or delta_lookup["global"]
        )

    def _default_pct(self, branch: str, category: str, delta_lookup: dict) -> float:
        return (
            delta_lookup["branch_category_pct"].get((branch, category))
            or delta_lookup["branch_pct"].get(branch)
            or delta_lookup["category_pct"].get(category)
            or delta_lookup["global_pct"]
        )

    def _estimate_2026_cutoff(self, branch, category, years, cutoffs, delta_lookup) -> float:
        last_cutoff = float(cutoffs[-1])
        fallback_pct = self._default_pct(branch, category, delta_lookup)

        if len(years) == 1:
            return max(1.0, last_cutoff * (1.0 + fallback_pct))

        recent_years = years[-3:]
        recent_cutoffs = cutoffs[-3:]

        consecutive_start = len(years) - 1
        while (
            consecutive_start > 0 and
            int(years[consecutive_start]) - int(years[consecutive_start - 1]) == 1
        ):
            consecutive_start -= 1
        consecutive_cutoffs = cutoffs[consecutive_start:]
        consecutive_len = len(consecutive_cutoffs)

        recent_year_gap = max(1, int(years[-1] - years[-2]))
        recent_delta = float(cutoffs[-1] - cutoffs[-2]) / recent_year_gap
        max_recent_step = max(
            last_cutoff * 0.12,
            abs(last_cutoff * fallback_pct),
            1500.0,
        )
        capped_recent_delta = float(np.clip(recent_delta, -max_recent_step, max_recent_step))
        naive_projection = last_cutoff + capped_recent_delta

        if consecutive_len >= 2:
            anchor_weights = np.linspace(1.5, 3.0, consecutive_len)
            recent_anchor = float(np.average(consecutive_cutoffs, weights=anchor_weights))
        else:
            recent_anchor = last_cutoff

        if len(recent_years) >= 2:
            centered_recent_years = recent_years - recent_years.mean()
            recent_weights = np.linspace(2.0, 4.0, len(recent_years))
            slope, intercept = np.polyfit(centered_recent_years, recent_cutoffs, deg=1, w=recent_weights)
            recent_trend_projection = intercept + slope * (self.target_year - recent_years.mean())
        else:
            recent_trend_projection = naive_projection

        base_projection = last_cutoff * (1.0 + fallback_pct)
        if len(years) >= 4:
            all_weights = np.linspace(0.3, 1.0, len(years))
            centered_years = years - years.mean()
            slope, intercept = np.polyfit(centered_years, cutoffs, deg=1, w=all_weights)
            long_trend_projection = intercept + slope * (self.target_year - years.mean())
        elif len(years) == 3:
            long_trend_projection = None
        else:
            long_trend_projection = None

        if consecutive_len >= 3:
            predicted = (
                recent_anchor * 0.45 +
                recent_trend_projection * 0.20 +
                naive_projection * 0.15 +
                base_projection * 0.15 +
                (long_trend_projection * 0.05 if long_trend_projection is not None else 0.05 * recent_anchor)
            )
        elif consecutive_len == 2:
            predicted = (
                recent_anchor * 0.55 +
                recent_trend_projection * 0.15 +
                naive_projection * 0.10 +
                base_projection * 0.15 +
                (long_trend_projection * 0.05 if long_trend_projection is not None else 0.05 * last_cutoff)
            )
        else:
            predicted = (
                last_cutoff * 0.70 +
                recent_trend_projection * 0.10 +
                naive_projection * 0.05 +
                base_projection * 0.10 +
                (long_trend_projection * 0.05 if long_trend_projection is not None else 0.05 * last_cutoff)
            )

        recent_floor_source = float(np.min(consecutive_cutoffs[-2:])) if consecutive_len >= 2 else last_cutoff
        lower_bound = recent_floor_source * 0.95
        upper_bound = last_cutoff * 1.35
        return max(1.0, float(np.clip(predicted, lower_bound, upper_bound)))
