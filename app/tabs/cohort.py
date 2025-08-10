import streamlit as st
import pandas as pd
import plotly.express as px
from src.analytics import monthly_cohort_retention

def render(df: pd.DataFrame):
    st.subheader("Cohort Retention")

    try:
        retention = monthly_cohort_retention(df)
    except Exception as e:
        st.warning(f"Cohort computation failed: {e}")
        return

    st.dataframe((retention * 100).round(1), use_container_width=True)

    st.write("Heatmap")
    heatmap_df = retention.reset_index().melt(id_vars="cohort_month", var_name="Period", value_name="Retention")
    fig = px.density_heatmap(
        heatmap_df,
        x="Period",
        y="cohort_month",
        z="Retention",
        nbinsx=len(retention.columns),
        nbinsy=len(retention.index),
        histfunc="avg",
        text_auto=True,
        template="plotly_white",
    )
    st.plotly_chart(fig, use_container_width=True)
