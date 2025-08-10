import streamlit as st
import pandas as pd
import plotly.express as px

SEGMENT_COLOR_MAP = {
    "Champions": "#F6C667",
    "Loyal": "#3CB371",
    "At Risk": "#FF6B6B",
    "Hibernating": "#A0AEC0",
}

def render(df: pd.DataFrame, train: dict):
    st.subheader("Overview")

    threshold = st.session_state.get("churn_threshold", 0.6)

    kpi1 = df["CLTV"].mean() if "CLTV" in df.columns else float("nan")
    kpi2 = ((df["Churn_Probability"] >= threshold).mean() * 100) if "Churn_Probability" in df.columns else float("nan")
    kpi3 = df["CustomerID"].nunique() if "CustomerID" in df.columns else len(df)

    c1, c2, c3 = st.columns(3)
    c1.metric("Avg CLTV", f"${kpi1:,.2f}" if pd.notnull(kpi1) else "N/A")
    c2.metric("% High Churn Risk", f"{kpi2:.1f}%" if pd.notnull(kpi2) else "N/A")
    c3.metric("Total Customers", f"{kpi3:,}")

    # Segment distribution
    if "Segment" in df.columns:
        seg_counts = df["Segment"].value_counts().reset_index()
        seg_counts.columns = ["Segment", "Count"]
        fig_bar = px.bar(
            seg_counts,
            x="Segment",
            y="Count",
            color="Segment",
            color_discrete_map=SEGMENT_COLOR_MAP,
            template="plotly_white",
            title="Segment Distribution",
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    # Churn vs CLTV scatter
    if set(["Churn_Probability", "CLTV", "Segment"]).issubset(df.columns):
        fig_scatter = px.scatter(
            df,
            x="Churn_Probability",
            y="CLTV",
            color="Segment",
            color_discrete_map=SEGMENT_COLOR_MAP,
            hover_data=["CustomerID"] if "CustomerID" in df.columns else None,
            template="plotly_white",
            title="Churn Probability vs CLTV",
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

    # Model metrics summary
    st.subheader("Model Performance")
    metrics = (train or {}).get("metrics", {})
    if metrics:
        m1, m2, m3 = st.columns(3)
        m1.metric("Model", metrics.get("model_type", "N/A").title())
        m2.metric("AUC", f"{metrics.get('auc', 0):.3f}")
        m3.metric("Accuracy", f"{metrics.get('accuracy', 0):.3f}")
    else:
        st.info("No metrics available.")

    # Sample data
    st.caption("Data preview")
    st.dataframe(df.head(50), use_container_width=True)
