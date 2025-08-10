from __future__ import annotations
import pandas as pd

def monthly_cohort_retention(df: pd.DataFrame) -> pd.DataFrame:
    """
    Expect df with columns: CustomerID, SignupDate, LastPurchaseDate
    Output: retention matrix with cohort_month as rows and period 0..N as columns (% retained)
    """
    work = df[["CustomerID", "SignupDate", "LastPurchaseDate"]].copy()
    work["cohort_month"] = work["SignupDate"].dt.to_period("M").dt.to_timestamp()
    work["active_month"] = work["LastPurchaseDate"].dt.to_period("M").dt.to_timestamp()

    cohort_data = (
        work.groupby(["cohort_month", "active_month"])["CustomerID"]
        .nunique()
        .reset_index()
    )

    # count cohort size at period 0
    cohort_size = cohort_data[cohort_data["cohort_month"] == cohort_data["active_month"]] \
        .groupby("cohort_month")["CustomerID"].sum()

    # period index
    cohort_data["period"] = (
        (cohort_data["active_month"].dt.year - cohort_data["cohort_month"].dt.year) * 12
        + (cohort_data["active_month"].dt.month - cohort_data["cohort_month"].dt.month)
    )

    pivot = cohort_data.pivot_table(
        index="cohort_month", columns="period", values="CustomerID", aggfunc="sum"
    ).fillna(0)

    retention = pivot.div(cohort_size, axis=0).fillna(0)
    retention.columns = [f"Month{int(c)}" for c in retention.columns]
    return retention.round(3)
