import streamlit as st
import pandas as pd
import plotly.express as px

SEGMENT_COLOR_MAP = {
    "Champions": "#F6C667",
    "Loyal": "#3CB371",
    "At Risk": "#FF6B6B",
    "Hibernating": "#A0AEC0",
}

def render(df: pd.DataFrame):
    st.subheader("Retention Priorities")

    threshold = st.session_state.get("churn_threshold", 0.6)

    if set(["Churn_Probability", "CLTV"]).issubset(df.columns):
        top_risk = df.sort_values(["Churn_Probability", "CLTV"], ascending=[False, False]).head(50)
        st.write("Top 50 high-value, high-risk customers")
        show_cols = [c for c in ["CustomerID", "Segment", "CLTV", "Churn_Probability"] if c in top_risk.columns]
        st.dataframe(top_risk[show_cols], use_container_width=True)

        st.download_button(
            "Download CSV",
            data=top_risk[show_cols].to_csv(index=False),
            file_name="retention_targets.csv",
            mime="text/csv",
        )

        st.subheader("CLTV vs Churn Probability")
        if "Segment" in df.columns:
            fig = px.scatter(
                df,
                x="Churn_Probability",
                y="CLTV",
                color="Segment",
                color_discrete_map=SEGMENT_COLOR_MAP,
                hover_data=["CustomerID"] if "CustomerID" in df.columns else None,
                template="plotly_white",
            )
        else:
            fig = px.scatter(
                df,
                x="Churn_Probability",
                y="CLTV",
                template="plotly_white",
            )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("CLTV or Churn_Probability not found in data.")
