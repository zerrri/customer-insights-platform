from __future__ import annotations
import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
import shap

try:
    import xgboost as xgb
    _HAS_XGB = True
except Exception:
    xgb = None
    _HAS_XGB = False

FEATURES_DEFAULT = [
    "Recency", "Frequency", "Monetary", "Tenure", "ActivityGap", "ARPU", "CLTV"
]

def _prepare_supervised(df: pd.DataFrame, label_col: str = "Churn") -> Tuple[pd.DataFrame, pd.Series]:
    work = df.copy()
    if label_col not in work.columns:
        # weak label: churn if no purchase for 180+ days
        work[label_col] = (work["Recency"] >= 180).astype(int)
    X = work[FEATURES_DEFAULT].fillna(0.0)
    y = work[label_col].astype(int)
    return X, y

def train_churn_model(
    df: pd.DataFrame,
    model_type: str = "auto",           # <--- now supports "auto"
    label_col: str = "Churn",
    random_state: int = 42,
) -> Dict[str, Any]:
    X, y = _prepare_supervised(df, label_col)

    # Must have both classes for classification
    if y.nunique() < 2:
        raise ValueError(f"Need at least 2 classes for classification. Found: {y.unique()}")

    # Robust split: try stratify, fall back if it fails (e.g., small minority class)
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=random_state, stratify=y
        )
    except ValueError:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=random_state
        )

    # Choose model
    chosen = model_type
    if model_type == "auto":
        chosen = "xgboost" if _HAS_XGB else "logistic"

    if chosen == "xgboost":
        model = xgb.XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            random_state=random_state,
            eval_metric="logloss",
        )
    else:
        model = LogisticRegression(max_iter=200)

    model.fit(X_train, y_train)
    proba = model.predict_proba(X_test)[:, 1]
    preds = (proba >= 0.5).astype(int)

    metrics = {
        "auc": float(roc_auc_score(y_test, proba)),
        "accuracy": float(accuracy_score(y_test, preds)),
        "confusion_matrix": confusion_matrix(y_test, preds).tolist(),
        "features": FEATURES_DEFAULT,
        "model_type": chosen,
    }

    # ---------- SHAP (robust) ----------
    explainer = None
    shap_values = None
    try:
        # Keep only numeric columns for SHAP background/test (safety)
        X_bg = X_train.select_dtypes(include=["number"]).copy()
        X_te = X_test.select_dtypes(include=["number"]).copy()

        # Limit background size for performance on laptops
        if len(X_bg) > 500:
            X_bg = X_bg.sample(500, random_state=random_state)

        # Let SHAP choose best explainer for the fitted model
        explainer = shap.Explainer(model, X_bg)
        shap_values = explainer(X_te)   # returns shap.Explanation
    except Exception as e:
        print(f"[train_churn_model] SHAP creation skipped: {e}")
        explainer = None
        shap_values = None
    # -----------------------------------

    return {
        "model": model,
        "metrics": metrics,
        "X_test": X_test,
        "y_test": y_test,
        "proba_test": proba,
        "explainer": explainer,
        "shap_values": shap_values,
    }

def predict_churn_probability(model, df: pd.DataFrame) -> pd.DataFrame:
    X = df[FEATURES_DEFAULT].fillna(0.0)
    df_out = df.copy()
    df_out["Churn_Probability"] = model.predict_proba(X)[:, 1]
    return df_out
