from __future__ import annotations
import pandas as pd
import numpy as np
from datetime import datetime
from dateutil.relativedelta import relativedelta

# Minimal schema we try to ensure exists before feature engineering
REQUIRED_BASE_COLS = [
    "CustomerID",
    "SignupDate",
    "LastPurchaseDate",
    "NumTransactions",
    "TotalSpend",
    "LastLoginDate",
    "AvgTransactionValue",
]

def _coerce_dates(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")
    return df

def _safe_div(a, b):
    """
    Safe division that returns:
      - a pandas Series (aligned to a's index) if `a` is a Series
      - otherwise a numpy ndarray
    Zeros in denominator -> 0.0
    """
    a_arr = np.asarray(a)
    b_arr = np.asarray(b)
    # Suppress the warning for this specific operation
    with np.errstate(divide='ignore', invalid='ignore'):
        out = np.where(b_arr == 0, 0.0, a_arr / b_arr)
    if isinstance(a, pd.Series):
        return pd.Series(out, index=a.index)
    return out

def load_and_enrich(path: str, today: datetime | None = None) -> pd.DataFrame:
    """
    Loads a customer CSV and computes standardized features for the rest of the app:
      - Recency (days since last purchase)
      - Frequency (NumTransactions)
      - Monetary (TotalSpend)
      - Tenure (days since signup)
      - ActivityGap (days since last login)
      - ARPU (TotalSpend / months of tenure)
      - CLTV (simple 12-month heuristic based on expected txn rate * ATV)
    Also includes an adapter to automatically support the IBM Telco Customer Churn dataset
    when file contains columns like 'customerID', 'tenure', 'MonthlyCharges', 'TotalCharges', 'Churn'.
    """
    today = today or datetime.utcnow()
    df = pd.read_csv(path)

    # ------------------------------------------------------------------------------
    # Telco adapter (auto-detect by column names)
    # ------------------------------------------------------------------------------
    if {"customerID", "tenure"}.issubset(df.columns):
        # Make numeric where needed
        if "TotalCharges" in df.columns:
            df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0)
        if "MonthlyCharges" in df.columns:
            df["MonthlyCharges"] = pd.to_numeric(df["MonthlyCharges"], errors="coerce").fillna(0)

        # Map churn Yes/No -> 1/0 if present
        if "Churn" in df.columns:
            df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0}).fillna(0).astype(int)

        # Rename to our internal schema
        df = df.rename(columns={
            "customerID": "CustomerID",
            "tenure": "Tenure",                  # months
            "TotalCharges": "TotalSpend"         # lifetime spend proxy
        })

        # Synthesize dates so Recency/Tenure features make sense.
        anchor = pd.to_datetime(today.date())

        # SignupDate is Tenure months ago (approx 30 days per month)
        df["SignupDate"] = anchor - pd.to_timedelta((df["Tenure"].fillna(0) * 30).astype(int), unit="D")

        # LastPurchaseDate:
        # - if churned, assume last activity ~1 month before the end of tenure
        # - if active (or no churn col), assume activity today (anchor)
        if "Churn" in df.columns:
            df["LastPurchaseDate"] = np.where(
                df["Churn"] == 1,
                df["SignupDate"] + pd.to_timedelta(np.maximum(df["Tenure"].fillna(0) - 1, 0) * 30, unit="D"),
                anchor
            )
        else:
            df["LastPurchaseDate"] = anchor

        df["LastPurchaseDate"] = pd.to_datetime(df["LastPurchaseDate"])
        df["LastLoginDate"] = df["LastPurchaseDate"]

        # Transactions proxy: use tenure months as a count-ish measure
        df["NumTransactions"] = df["Tenure"].fillna(0).clip(lower=0)

        # AvgTransactionValue proxy: use MonthlyCharges if available
        if "MonthlyCharges" in df.columns:
            df["AvgTransactionValue"] = df["MonthlyCharges"].fillna(0)
        else:
            df["AvgTransactionValue"] = _safe_div(df.get("TotalSpend", 0), df["NumTransactions"]).astype(float)

    # ------------------------------------------------------------------------------
    # Generic pipeline (works for Telco after adapter and for your original CSV)
    # ------------------------------------------------------------------------------

    # Harmonize a couple common alt names
    if "customer_id" in df.columns and "CustomerID" not in df.columns:
        df = df.rename(columns={"customer_id": "CustomerID"})
    if "avg_txn_value" in df.columns and "AvgTransactionValue" not in df.columns:
        df = df.rename(columns={"avg_txn_value": "AvgTransactionValue"})

    # Coerce date columns if present
    df = _coerce_dates(df, ["SignupDate", "LastPurchaseDate", "LastLoginDate"])

    # Ensure required columns exist (fill with safe defaults)
    for c in REQUIRED_BASE_COLS:
        if c not in df.columns:
            if c in ("NumTransactions", "TotalSpend", "AvgTransactionValue"):
                df[c] = 0.0
            else:
                df[c] = pd.NaT if "Date" in c else np.nan

    # ------ Feature engineering ------
    # Recency in days (if LastPurchaseDate missing, treat as very old)
    df["Recency"] = (today - df["LastPurchaseDate"]).dt.days.replace({np.nan: np.inf})
    # Frequency & Monetary
    df["Frequency"] = df["NumTransactions"].fillna(0).astype(float)
    df["Monetary"] = df["TotalSpend"].fillna(0.0).astype(float)
    # Tenure in days (if missing, 0)
    df["Tenure"] = (today - df["SignupDate"]).dt.days.clip(lower=0).fillna(0).astype(float)

    # Activity gap (days since last login)
    df["ActivityGap"] = (today - df["LastLoginDate"]).dt.days
    df["ActivityGap"] = df["ActivityGap"].fillna(df["ActivityGap"].max()).fillna(0).astype(float)

    # Derived KPIs
    months = np.maximum(_safe_div(df["Tenure"], 30.0), 1.0)  # avoid div by 0
    df["ARPU"] = _safe_div(df["Monetary"], months)

    # Simple CLTV heuristic: expected next-12-month revenue
    # expected_txn_rate = (Frequency / months); next 12 months => *12
    expected_txn_rate = _safe_div(df["Frequency"], months)

    # Fallback for ATV if missing: TotalSpend / Frequency (as a Series, not ndarray)
    fallback_atv = pd.Series(_safe_div(df["Monetary"], df["Frequency"]), index=df.index)

    df["CLTV"] = (expected_txn_rate * 12.0) * df["AvgTransactionValue"].fillna(fallback_atv)
    df["CLTV"] = df["CLTV"].replace([np.inf, -np.inf, np.nan], 0)

    # Guard rails
    df["CLTV"] = df["CLTV"].fillna(0).clip(lower=0)
    df["Recency"] = df["Recency"].replace({np.inf: 9999}).clip(lower=0)
    df["CustomerID"] = df["CustomerID"].astype(str)

    # Order important columns up front, keep everything else too
    cols_order = [
        "CustomerID", "SignupDate", "LastPurchaseDate", "LastLoginDate",
        "Tenure", "Recency", "ActivityGap", "Frequency", "Monetary",
        "AvgTransactionValue", "ARPU", "CLTV", "NumTransactions", "TotalSpend"
    ]
    front = [c for c in cols_order if c in df.columns]
    rest = [c for c in df.columns if c not in front]
    df = df[front + rest]

    return df
