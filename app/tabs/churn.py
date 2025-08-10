import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import shap
import matplotlib.pyplot as plt

def render(df: pd.DataFrame, train: dict):
    st.subheader("Churn Model Performance")

    metrics = train.get("metrics", {})
    model_type = metrics.get("model_type", "Unknown")
    auc = metrics.get("auc", float("nan"))
    acc = metrics.get("accuracy", float("nan"))

    st.write(f"**Model:** {model_type}  |  **AUC:** {auc:.3f}  |  **Accuracy:** {acc:.3f}")

    cm = metrics.get("confusion_matrix")
    if cm is not None:
        st.write("Confusion Matrix (test):")
        st.write(pd.DataFrame(cm, index=["Actual 0", "Actual 1"], columns=["Pred 0", "Pred 1"]))

    # Probability histogram
    if "Churn_Probability" in df.columns:
        st.subheader("Churn Probability Distribution")
        fig_hist = px.histogram(
            df,
            x="Churn_Probability",
            nbins=30,
            template="plotly_white",
        )
        st.plotly_chart(fig_hist, use_container_width=True)

    # SHAP global importance
    st.subheader("Top Features Impacting Churn (SHAP)")
    try:
        explainer = train.get("explainer", None)
        shap_values = train.get("shap_values", None)
        X_test = train.get("X_test", None)
        if explainer is not None and shap_values is not None and X_test is not None:
            fig = plt.figure()
            shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
            st.pyplot(fig, bbox_inches="tight")
            plt.close(fig)
        else:
            st.info("SHAP summary not available in this environment.")
    except Exception as e:
        st.warning(f"SHAP plotting skipped: {e}")
