from src.etl import load_and_enrich

def test_etl_runs():
    df = load_and_enrich("data/simulated_customers.csv")
    assert "CustomerID" in df.columns
    assert "CLTV" in df.columns
    assert len(df) > 0
