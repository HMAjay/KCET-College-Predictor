"""Streamlit web interface for the KCET College Predictor."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src.model.predictor import KCETPredictor


st.set_page_config(
    page_title="KCET College Predictor",
    page_icon="🎓",
    layout="wide",
)


@st.cache_resource
def load_predictor() -> KCETPredictor:
    return KCETPredictor()


def build_results_frame(results: list[dict]) -> pd.DataFrame:
    if not results:
        return pd.DataFrame()

    frame = pd.DataFrame(results).copy()
    preferred_columns = [
        "College_Code",
        "College_Name",
        "Branch_Code",
        "Branch",
        "Category",
        "Cutoff_Rank",
        "Cutoff_2025",
        "Gap",
    ]
    available_columns = [column for column in preferred_columns if column in frame.columns]
    return frame[available_columns]


def main() -> None:
    predictor = load_predictor()
    metadata = predictor.metadata

    st.title("KCET College Predictor")
    st.caption(
        f"Trend-based prediction for {metadata.get('target_year', predictor.target_year)} "
        f"using historical cutoff data from {metadata.get('min_year', '?')} to {metadata.get('max_year', '?')}."
    )

    with st.sidebar:
        st.header("Model Info")
        st.write(f"Target Year: `{metadata.get('target_year', predictor.target_year)}`")
        st.write(f"Colleges Covered: `{metadata.get('n_colleges', 'N/A')}`")
        st.write(f"Categories: `{len(predictor.available_categories)}`")
        st.write("All categories:")
        st.code(", ".join(predictor.available_categories), language=None)
        st.write("All branches:")
        all_branch_options = sorted(predictor.branch_option_map.values())
        st.code("\n".join(all_branch_options), language=None)

    left, right = st.columns([1, 2], gap="large")

    with left:
        st.subheader("Enter Your Details")

        rank = st.number_input(
            "KCET Rank",
            min_value=1,
            max_value=200000,
            value=10000,
            step=100,
        )

        branch = st.text_input(
            "Preferred Branch",
            value="CSE",
            help="Examples: CSE, ISE, ECE, AIML, ETCE, Information Science, Robotics and Automation",
        )

        category = st.selectbox(
            "Category",
            options=predictor.available_categories,
            index=predictor.available_categories.index("GM") if "GM" in predictor.available_categories else 0,
        )

        top_n = st.slider("Number of Results", min_value=1, max_value=50, value=10)

        submitted = st.button("Predict Colleges", type="primary", use_container_width=True)

    with right:
        st.subheader("Prediction Results")

        if submitted:
            branch_matches = predictor.resolve_branch_matches(branch)
            if branch_matches:
                matched_labels = [predictor.branch_option_map.get(name, name) for name in branch_matches[:5]]
                st.info("Matched branches: " + ", ".join(matched_labels))
            else:
                st.warning(f"No similar branch was found in the dataset for '{branch}'.")

            results = predictor.predict(int(rank), branch, category, top_n)
            results_frame = build_results_frame(results)

            if results_frame.empty:
                st.error("No colleges matched the projected cutoff for the inputs you entered.")
            else:
                st.success(f"Found {len(results_frame)} projected option(s).")
                st.dataframe(results_frame, use_container_width=True, hide_index=True)
                st.download_button(
                    "Download Results as CSV",
                    data=results_frame.to_csv(index=False).encode("utf-8"),
                    file_name="kcet_prediction_results.csv",
                    mime="text/csv",
                    use_container_width=True,
                )
        else:
            st.write(
                "Use the form on the left to enter rank, branch, category, and result count, "
                "then click **Predict Colleges**."
            )


if __name__ == "__main__":
    main()
