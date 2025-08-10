import streamlit as st
import pandas as pd
import plotly.express as px

def render(df: pd.DataFrame):
    st.subheader("Trends")

    if "SignupDate" not in df.columns:
        st.info("SignupDate column not found for trend analysis.")
        return

    work = df.copy()
    work["Month"] = pd.to_datetime(work["SignupDate"]).dt.to_period("M").dt.to_timestamp()

    aggs = {}
    if "CLTV" in work.columns:
        aggs["CLTV"] = "mean"
    if "Churn_Probability" in work.columns:
        aggs["Churn_Probability"] = "mean"

    if not aggs:
        st.info("No numeric metrics available for trends.")
        return

    trend = work.groupby("Month").agg(aggs).reset_index()

    # Combined line chart (one trace per metric)
    melted = trend.melt(id_vars="Month", var_name="Metric", value_name="Value")
    fig = px.line(
        melted,
        x="Month",
        y="Value",
        color="Metric",
        markers=True,
        template="plotly_white",
        title="Monthly Averages",
    )
    st.plotly_chart(fig, use_container_width=True)

    # Optional: segment-sliced trend if Segment exists
    if "Segment" in work.columns and "Churn_Probability" in work.columns:
        st.write("Churn Probability Trend by Segment")
        seg_trend = work.groupby(["Month", "Segment"])["Churn_Probability"].mean().reset_index()
        fig2 = px.line(
            seg_trend,
            x="Month",
            y="Churn_Probability",
            color="Segment",
            markers=True,
            template="plotly_white",
        )
        st.plotly_chart(fig2, use_container_width=True)
