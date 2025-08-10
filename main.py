from src.etl import load_and_enrich
from src.segmentation import segment_rfm
from src.modeling import train_churn_model, predict_churn_probability
from src.analytics import monthly_cohort_retention

if __name__ == "__main__":
    df = load_and_enrich("data/telco_churn.csv")
    df_seg, scaler, km = segment_rfm(df)
    result = train_churn_model(df_seg, model_type="xgboost")
    df_pred = predict_churn_probability(result["model"], df_seg)

    retention = monthly_cohort_retention(df_pred)

    print("Rows:", len(df_pred))
    print("Sample:")
    print(df_pred.head(5))
    print("\nCohort retention (head):")
    print(retention.head(3))
