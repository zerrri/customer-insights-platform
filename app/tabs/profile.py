import streamlit as st
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt

def render(df: pd.DataFrame, train: dict):
    st.subheader("Customer Profile")

    if "CustomerID" not in df.columns:
        st.info("CustomerID column not found.")
        return

    ids = df["CustomerID"].astype(str).unique().tolist()
    if not ids:
        st.info("No customers in current filter.")
        return

    cid = st.selectbox("Choose a CustomerID", ids)
    row = df[df["CustomerID"].astype(str) == str(cid)]
    if row.empty:
        st.info("No data for selected customer.")
        return

    st.write(row.head(1).T)

    # Local SHAP (best effort)
    try:
        explainer = train.get("explainer")
        model = train.get("model")
        features = train.get("metrics", {}).get("features", [])
        if explainer is not None and features:
            X = row[features].fillna(0.0)
            shap_val = explainer.shap_values(X)
            st.write("SHAP Waterfall (local explanation)")
            fig = plt.figure()
            shap.plots.waterfall(
                shap.Explanation(values=shap_val[0], base_values=0, data=X.values[0], feature_names=features),
                show=False
            )
            st.pyplot(fig, bbox_inches="tight")
            plt.close(fig)
        else:
            st.info("Local SHAP explanation unavailable.")
    except Exception as e:
        st.info(f"Local SHAP explanation unavailable: {e}")
