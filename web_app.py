"""Streamlit web interface for the KCET College Predictor."""

from __future__ import annotations

from pathlib import Path

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


@st.cache_data
def load_file_bytes(path: str) -> bytes:
    return Path(path).read_bytes()


def inject_styles() -> None:
    st.markdown(
        """
        <style>
            .stApp {
                background:
                    radial-gradient(circle at top left, rgba(255, 209, 102, 0.22), transparent 24%),
                    radial-gradient(circle at top right, rgba(66, 153, 225, 0.18), transparent 26%),
                    linear-gradient(180deg, #f8fbff 0%, #eef4f8 48%, #f7f7f2 100%);
            }
            .block-container {
                padding-top: 2rem;
                padding-bottom: 3rem;
                max-width: 1200px;
            }
            .hero-card, .info-card, .section-card {
                background: rgba(255, 255, 255, 0.82);
                border: 1px solid rgba(15, 23, 42, 0.08);
                border-radius: 22px;
                box-shadow: 0 16px 40px rgba(15, 23, 42, 0.08);
                backdrop-filter: blur(10px);
                padding: 1.35rem 1.4rem;
            }
            .hero-title {
                font-size: 2.4rem;
                font-weight: 800;
                line-height: 1.05;
                color: #102a43;
                margin-bottom: 0.45rem;
            }
            .hero-subtitle {
                font-size: 1rem;
                line-height: 1.75;
                color: #334e68;
                margin-bottom: 0;
            }
            .pill-row {
                display: flex;
                flex-wrap: wrap;
                gap: 0.65rem;
                margin-top: 1rem;
            }
            .pill {
                background: #fff7d6;
                color: #7c5d00;
                border: 1px solid rgba(124, 93, 0, 0.12);
                border-radius: 999px;
                padding: 0.45rem 0.85rem;
                font-size: 0.92rem;
                font-weight: 600;
            }
            .section-title {
                font-size: 1.35rem;
                font-weight: 800;
                color: #102a43;
                margin-bottom: 0.35rem;
            }
            .section-copy {
                color: #486581;
                line-height: 1.7;
                margin-bottom: 0;
            }
            .about-copy {
                color: #334e68;
                line-height: 1.85;
                font-size: 1rem;
            }
            .stButton > button, .stDownloadButton > button {
                border-radius: 14px;
                min-height: 2.8rem;
                font-weight: 700;
            }
            .stDownloadButton > button {
                background: #15803d !important;
                color: #ffffff !important;
                border: 1px solid #166534 !important;
                font-weight: 800 !important;
            }
            .stDownloadButton > button *,
            .stDownloadButton > button p,
            .stDownloadButton > button span,
            .stDownloadButton > button div {
                color: #ffffff !important;
                fill: #ffffff !important;
                font-weight: 800 !important;
            }
            .stDownloadButton > button:hover {
                background: #166534 !important;
                color: #ffffff !important;
                border: 1px solid #14532d !important;
                font-weight: 800 !important;
            }
            .stDownloadButton > button:hover *,
            .stDownloadButton > button:hover p,
            .stDownloadButton > button:hover span,
            .stDownloadButton > button:hover div {
                color: #ffffff !important;
                fill: #ffffff !important;
                font-weight: 800 !important;
            }
            label,
            .stNumberInput label,
            .stTextInput label,
            .stSelectbox label,
            .stSlider label,
            div[data-testid="stWidgetLabel"] p,
            div[data-testid="stWidgetLabel"] label,
            div[data-testid="stMarkdownContainer"] p {
                color: #102a43 !important;
            }
            .streamlit-expanderHeader,
            .streamlit-expanderHeader p,
            details summary,
            details summary p,
            div[data-testid="stExpander"] summary,
            div[data-testid="stExpander"] summary p,
            div[data-testid="stExpander"] summary span {
                color: #ffffff !important;
                fill: #ffffff !important;
            }
            div[data-testid="stMetric"] {
                background: rgba(255, 255, 255, 0.78);
                border: 1px solid rgba(15, 23, 42, 0.08);
                border-radius: 18px;
                padding: 0.7rem 0.8rem;
            }
            div[data-testid="stMetric"] label,
            div[data-testid="stMetric"] p,
            div[data-testid="stMetric"] div {
                color: #111111 !important;
            }
            .popup-card {
                background: linear-gradient(135deg, #fff5cf 0%, #fffdf7 100%);
                border: 1px solid rgba(124, 93, 0, 0.15);
                border-radius: 20px;
                box-shadow: 0 18px 40px rgba(124, 93, 0, 0.12);
                padding: 1.2rem 1.25rem;
                margin-bottom: 1rem;
            }
            .scroll-chip-row {
                overflow-x: auto;
                overflow-y: hidden;
                white-space: nowrap;
                padding-bottom: 0.35rem;
                margin-bottom: 0.75rem;
                scrollbar-width: thin;
            }
            .scroll-chip-row::-webkit-scrollbar {
                height: 8px;
            }
            .scroll-chip-row::-webkit-scrollbar-thumb {
                background: rgba(16, 42, 67, 0.25);
                border-radius: 999px;
            }
            .scroll-chip {
                display: inline-block;
                background: #ffffff;
                color: #102a43;
                border: 1px solid rgba(15, 23, 42, 0.12);
                border-radius: 999px;
                padding: 0.35rem 0.7rem;
                margin-right: 0.45rem;
                font-size: 0.84rem;
                font-weight: 600;
            }
            .branch-list {
                background: rgba(255, 255, 255, 0.72);
                border: 1px solid rgba(15, 23, 42, 0.08);
                border-radius: 16px;
                padding: 0.8rem 0.9rem;
                color: #102a43;
                line-height: 1.55;
                font-size: 0.88rem;
                white-space: pre-wrap;
                max-height: 230px;
                overflow-y: auto;
                overflow-x: auto;
            }
            .results-shell div[data-testid="stDataFrame"] {
                border-radius: 16px;
                overflow: hidden;
                border: 1px solid rgba(15, 23, 42, 0.08);
            }
            @media (max-width: 768px) {
                .block-container {
                    padding-top: 1rem;
                    padding-bottom: 2rem;
                    padding-left: 0.9rem;
                    padding-right: 0.9rem;
                }
                .hero-card, .info-card, .section-card, .popup-card {
                    border-radius: 18px;
                    padding: 1rem;
                }
                .hero-title {
                    font-size: 1.8rem;
                    line-height: 1.12;
                }
                .hero-subtitle,
                .section-copy,
                .about-copy {
                    font-size: 0.95rem;
                    line-height: 1.65;
                }
                .pill-row {
                    gap: 0.5rem;
                }
                .pill {
                    font-size: 0.8rem;
                    padding: 0.4rem 0.7rem;
                }
                .section-title {
                    font-size: 1.15rem;
                }
                .branch-list {
                    font-size: 0.83rem;
                    max-height: 180px;
                }
                .stButton > button, .stDownloadButton > button {
                    min-height: 3rem;
                    font-size: 0.98rem;
                }
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_branch_list(branches: list[str]) -> None:
    st.markdown(
        f'<div class="branch-list">{"<br>".join(branches)}</div>',
        unsafe_allow_html=True,
    )


def render_disclaimer_popup() -> None:
    if "disclaimer_accepted" not in st.session_state:
        st.session_state.disclaimer_accepted = False

    if st.session_state.disclaimer_accepted:
        return

    st.markdown(
        """
        <div class="hero-card" style="max-width: 760px; margin: 5rem auto 0 auto; text-align: center;">
            <div class="hero-title">Disclaimer</div>
            <p class="hero-subtitle">
                This predictor works on previous cutoff trends and should be used for planning only.
                It may be inaccurate in some cases because real allotments can change every year.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    button_left, button_center, button_right = st.columns([2, 1.2, 2])
    with button_center:
        if st.button("I understand", key="accept_disclaimer", use_container_width=True, type="primary"):
            st.session_state.disclaimer_accepted = True
            st.rerun()
    st.stop()


def render_overview(metadata: dict, predictor: KCETPredictor) -> None:
    st.markdown(
        f"""
        <div class="hero-card">
            <div class="hero-title">KCET 2026 College Predictor</div>
            <p class="hero-subtitle">
                Plan colleges using your KCET rank, preferred branch, and category with a cleaner,
                trend-based view of likely options using historical cutoff patterns from
                <strong>{metadata.get('min_year', '?')}</strong> to <strong>{metadata.get('max_year', '?')}</strong>.
            </p>
            <div class="pill-row">
                <div class="pill">Prediction target: {metadata.get('target_year', predictor.target_year)}</div>
                <div class="pill">Colleges covered: {metadata.get('n_colleges', 'N/A')}</div>
                <div class="pill">Based on previous cutoff trends</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar(metadata: dict, predictor: KCETPredictor) -> None:
    with st.sidebar:
        st.header("What's Inside")
        st.metric("Target Year", metadata.get("target_year", predictor.target_year))
        st.metric("Colleges Covered", metadata.get("n_colleges", "N/A"))
        st.metric("Categories", len(predictor.available_categories))
        st.caption("Available categories")
        category_chips = "".join(
            f'<span class="scroll-chip">{category}</span>'
            for category in predictor.available_categories
        )
        st.markdown(
            f'<div class="scroll-chip-row">{category_chips}</div>',
            unsafe_allow_html=True,
        )
        st.caption("Available Branches")
        prioritized_branches = list(dict.fromkeys(predictor.common_branch_options))
        remaining_branches = [
            branch
            for branch in sorted(predictor.branch_option_map.values())
            if branch not in prioritized_branches
        ]
        all_branch_labels = prioritized_branches + remaining_branches
        render_branch_list(prioritized_branches)
        with st.expander("View all"):
            render_branch_list(all_branch_labels)

        st.markdown("---")
        st.caption("Download Cutoffs (Official KEA 2nd Round Extended PDFs)")
        cutoff_2024_pdf = Path("data/raw/KCET 2024 Cutoff.pdf")
        cutoff_2025_pdf = Path("data/raw/KCET 2025 Cutoff.pdf")

        if cutoff_2024_pdf.exists():
            st.download_button(
                "Download 2024 KCET Cutoff PDF",
                data=load_file_bytes(str(cutoff_2024_pdf)),
                file_name="KCET 2024 Official Cutoff.pdf",
                mime="application/pdf",
                key="sidebar_download_2024_raw_pdf",
                use_container_width=True,
            )
        else:
            st.caption("KCET 2024 raw PDF not found.")

        if cutoff_2025_pdf.exists():
            st.download_button(
                "Download 2025 KCET Official Cutoff PDF",
                data=load_file_bytes(str(cutoff_2025_pdf)),
                file_name="KCET 2025 Official Cutoff.pdf",
                mime="application/pdf",
                key="sidebar_download_2025_raw_pdf",
                use_container_width=True,
            )
        else:
            st.caption("KCET 2025 raw PDF not found.")


def render_predictor(predictor: KCETPredictor) -> None:
    st.markdown(
        """
        <div class="section-card">
            <div class="section-title">Predict Your Options</div>
            <p class="section-copy">
                Enter your rank, choose your category, and search by your preferred branch name or code such as
                CSE, ISE, ECE, AIML, ETCE, or Information Science.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.write("")

    left, right = st.columns([1, 1.35], gap="large")

    with left:
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
        top_n = st.slider("Number of Results", min_value=5, max_value=50, value=15, step=5)
        submitted = st.button("Predict Colleges", type="primary", use_container_width=True)

    with right:
        st.markdown("#### Prediction Results")
        st.caption("Nearest matches are shown by projected cutoff gap from your entered rank.")

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
                st.markdown('<div class="results-shell">', unsafe_allow_html=True)
                st.dataframe(results_frame, use_container_width=True, hide_index=True, height=420)
                st.markdown("</div>", unsafe_allow_html=True)
                st.download_button(
                    "Download Results as CSV",
                    data=results_frame.to_csv(index=False).encode("utf-8"),
                    file_name="kcet_prediction_results.csv",
                    mime="text/csv",
                    use_container_width=True,
                )
        else:
            st.empty()


def render_about() -> None:
    st.markdown(
        """
        <div class="section-card">
            <div class="section-title">About</div>
            <p class="about-copy">
                Hi, I’m <strong>H M Ajay</strong>, a 3rd year Information Science Engineering student from Bangalore,
                currently studying at RV Institute of Technology and Management (RVITM). My area of interest is
                web development, and I enjoy building practical tools that make student decisions easier, clearer,
                and more confident.
            </p>
            <p class="about-copy">
                When I was applying for KCET colleges myself, I kept wishing there was a website like this one,
                something simple, reliable, and focused on helping students plan colleges based on their KCET rank.
                A lot of the information available online felt scattered, hard to compare, or not immediately useful
                when trying to make real counselling decisions. That experience stayed with me and pushed me to build
                a platform that brings cutoff trends, planning, and clarity into one place.
            </p>
            <p class="about-copy">
                This project is my attempt to give students a better starting point while they shortlist colleges,
                explore branches, and understand where they stand in the counselling process. I believe student tools
                should be accessible, straightforward, and genuinely helpful, especially during important moments like
                admissions. Building and improving products like this keeps me motivated to keep learning, keep
                designing better experiences, and keep solving real problems through technology.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    predictor = load_predictor()
    metadata = predictor.metadata

    inject_styles()
    render_disclaimer_popup()
    render_sidebar(metadata, predictor)
    render_overview(metadata, predictor)
    st.write("")
    render_predictor(predictor)
    st.write("")
    render_about()


if __name__ == "__main__":
    main()
