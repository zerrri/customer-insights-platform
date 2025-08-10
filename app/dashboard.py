import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
import pandas as pd
from datetime import date

from src.etl import load_and_enrich
from src.segmentation import segment_rfm
from src.modeling import train_churn_model, predict_churn_probability
from src.analytics import monthly_cohort_retention

from app.tabs import overview, churn, retention, cohort, profile, trends, chatbot

st.set_page_config(page_title="Customer Insights Platform", layout="wide")

# === Data file path (use a specific CSV, not the folder) ===
DATA_FILE = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', 'data', 'telco_churn.csv')
)
# To use the other dataset instead, switch to:
# DATA_FILE = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'simulated_customers.csv'))

@st.cache_data(show_spinner=False)
def get_data():
    """Load data, run feature engineering, and apply RFM segmentation."""
    df = load_and_enrich(DATA_FILE)  # <-- pass a file path, not the directory
    df, _, _ = segment_rfm(df, k=4, random_state=42)
    return df


def _safe_date_range(col: pd.Series) -> tuple[date, date]:
    if col.empty:
        today = pd.Timestamp.today().date()
        return today, today
    lo = pd.to_datetime(col.min()).date() if pd.notnull(col.min()) else pd.Timestamp.today().date()
    hi = pd.to_datetime(col.max()).date() if pd.notnull(col.max()) else pd.Timestamp.today().date()
    if lo > hi:
        lo, hi = hi, lo
    return lo, hi


def _apply_filters(df: pd.DataFrame) -> pd.DataFrame:
    st.sidebar.header("🔎 Filters")
    work = df.copy()

    # Segment filter (if exists)
    if "Segment" in work.columns:
        segs = sorted(work["Segment"].dropna().unique().tolist())
        selected_segs = st.sidebar.multiselect("Segments", segs, default=segs)
        if selected_segs:
            work = work[work["Segment"].isin(selected_segs)]

    # Location filter (if exists)
    if "Location" in work.columns:
        locs = sorted(work["Location"].dropna().unique().tolist())
        selected_locs = st.sidebar.multiselect("Locations", locs, default=locs)
        if selected_locs:
            work = work[work["Location"].isin(selected_locs)]

    # Gender filter (if exists)
    if "Gender" in work.columns:
        genders = sorted(work["Gender"].dropna().unique().tolist())
        selected_genders = st.sidebar.multiselect("Gender", genders, default=genders)
        if selected_genders:
            work = work[work["Gender"].isin(selected_genders)]

    # Signup date range (if exists)
    if "SignupDate" in work.columns:
        lo, hi = _safe_date_range(work["SignupDate"])
        dr = st.sidebar.date_input("Signup Date Range", value=(lo, hi))
        if isinstance(dr, tuple) and len(dr) == 2:
            start, end = dr
            work = work[work["SignupDate"].between(pd.to_datetime(start), pd.to_datetime(end))]

    # Threshold (used by several tabs)
    churn_threshold = st.sidebar.slider("High Churn Risk Threshold", min_value=0.0, max_value=1.0, value=0.6, step=0.01)
    st.session_state["churn_threshold"] = churn_threshold

    # === Model selection ===
    st.sidebar.header("⚙️ Model Settings")
    model_type = st.sidebar.selectbox(
        "Choose Churn Model Type",
        options=["auto", "logistic", "xgboost"],
        index=0,
        help="auto = XGBoost if available, else Logistic Regression"
    )
    st.session_state["model_type"] = model_type

    return work


@st.cache_data(show_spinner=False)
def get_models(df: pd.DataFrame):
    """
    Train churn model on the filtered dataset (fallback to full dataset if too small).
    Returns training artifacts + df with predicted Churn_Probability.
    """
    min_rows = 200  # heuristic to avoid instability
    base = df.copy()
    chosen_model_type = st.session_state.get("model_type", "auto")

    if len(base) < min_rows:
        # If filtered data is too small, train on the full dataset but return preds for filtered
        full = get_data()
        train = train_churn_model(full, model_type=chosen_model_type)
        df_pred = predict_churn_probability(train["model"], base)
        return train, df_pred
    else:
        train = train_churn_model(base, model_type=chosen_model_type)
        df_pred = predict_churn_probability(train["model"], base)
        return train, df_pred


def main():
    st.title("Customer Insights Platform")

    df = get_data()
    df_f = _apply_filters(df)

    # Train/predict with the filtered view (fallback handled inside)
    train, df_pred = get_models(df_f)

    tabs = st.tabs(["Overview", "Churn", "Retention", "Cohort", "Profile", "Trends", "Chatbot"])

    with tabs[0]:
        overview.render(df_pred, train)   # <<< pass train so we can show metrics

    with tabs[1]:
        churn.render(df_pred, train)

    with tabs[2]:
        retention.render(df_pred)

    with tabs[3]:
        cohort.render(df_pred)

    with tabs[4]:
        profile.render(df_pred, train)

    with tabs[5]:
        trends.render(df_pred)

    with tabs[6]:
        chatbot.render(df_pred)


if __name__ == "__main__":
    main()
