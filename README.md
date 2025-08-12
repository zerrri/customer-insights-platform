# Customer Insights Platform

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-Enabled-brightgreen)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

A realistic end-to-end analytics stack for customer lifecycle analysis and churn prediction.  
It’s designed to work out of the box with sample data, or with your own CSV, giving you actionable insights without spending weeks wiring things together.

> **Why this exists**  
> Analytics projects often stall because the code is messy, the data prep is unclear, or the outputs aren’t actionable.  
> This project solves that with small, testable modules, reproducible dependencies, and a usable Streamlit UI.

---


## 📂 Table of Contents
- [Architecture](#architecture)
- [Quickstart](#quickstart)
- [Data Requirements](#data-requirements)
- [Usage Guide](#usage-guide)
- [Project Layout](#project-layout)
- [Configuration](#configuration)
- [Modeling Details](#modeling-details)
- [Cohort & Retention](#cohort--retention)
- [Testing](#testing)
- [Troubleshooting](#troubleshooting)
- [Roadmap](#roadmap)
- [Contributing](#contributing)

---


## 🏗 Architecture

```plaintext
CSV/Parquet
    │
    ▼
[ETL] → Feature Engineering (R, F, M, Tenure, ARPU, CLTV proxy)
    │
    ▼
[RFM Segmentation] (Scaler + KMeans) → Segment Labels
    │
    ▼
[Churn Modeling] (Logistic | XGBoost) → Probabilities + SHAP
    │
    └────────────→ [Cohort Analytics] (Monthly retention matrix)
    │
    ▼
[Streamlit UI] Tabs: Overview · Churn · Retention · Cohort · Profile · Trends · Chatbot
```
---


**Design choices that matter**
- **Modular code** — Each step is in its own file under `src/`.
- **Transparent defaults** — Missing `Churn` is auto-derived (`Recency ≥ 180 days`).
- **Reproducible** — `requirements.txt` is pinned for consistent installs.
- **Usable UI** — Tabs for quick KPIs, churn risk, retention, and drill-downs.

---

## 🚀 Quickstart

```bash
# 1. Create and activate a virtual environment (Python 3.10 or 3.11)
python -m venv .venv

# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the demo
python main.py

# 4. Or launch the dashboard
streamlit run app/dashboard.py

Default dataset → data/telco_churn.csv
```
---
## 📊 Data Requirements

| Key                   | Description |
|-----------------------|-------------|
| `CustomerID`          | Unique id. Aliases: `customer_id`, `customerID` |
| `SignupDate`          | Customer signup date |
| `LastPurchaseDate`    | Used to compute `Recency` and cohorts |
| `NumTransactions`     | Count of purchases |
| `AvgTransactionValue` | Either average purchase value or total monetary |
| `LastLoginDate`       | *(Optional)* For engagement/Activity Gap |
| `Tenure`              | In days/months; computed if missing |
| `Churn`               | *(Optional)* If missing, derived as `Recency ≥ 180` |

| Derived Field         | Description |
|-----------------------|-------------|
| `Recency`             | Days since last purchase |
| `Frequency`           | Number of transactions |
| `ARPU`                | Average revenue per user |
| `CLTV`                | Simple lifetime value proxy |
| `Cohort`              | Signup month grouping |
| `Activity Gap`        | Days since last login |
| `Churn (derived)`     | 1 if `Recency ≥ 180` |

---

## 📖 Usage Guide

### Streamlit App
```bash
streamlit run app/dashboard.py
```

---

### Tabs include:

- **Overview** — KPIs, churn rate, ARPU, segment distribution
- **Churn** — Train & evaluate Logistic/XGBoost, see SHAP impact
- **Retention** — RFM clusters + suggested actions
- **Cohort** — Monthly retention heatmap
- **Profile** — Drill into single customer
- **Trends** — Time-series by engagement/purchases
- **Chatbot** — Inline helper (no API key needed)

---

### CLI Pipeline

```bash
python main.py
```
---

## 📂 Project Layout

```plaintext
customer-insights-platform/
│  .gitignore
│  enhance_dataset.py
│  main.py
│  README.md
│  requirements.txt
│
├── app/
│   ├── dashboard.py
│   ├── __init__.py
│   └── tabs/
│       ├── chatbot.py
│       ├── churn.py
│       ├── cohort.py
│       ├── overview.py
│       ├── profile.py
│       ├── retention.py
│       ├── trends.py
│       └── __init__.py
│
├── data/
│   ├── simulated_customers.csv
│   └── telco_churn.csv
│
├── docs/
│   └── screenshots/
│
├── src/
│   ├── analytics.py
│   ├── etl.py
│   ├── modeling.py
│   ├── segmentation.py
│   └── __init__.py
│
└── tests/
    └── test_etl.py
## ⚙️ Configuration
```

- **Dataset path** — Set in `app/dashboard.py` under `DATA_FILE`.
- **Model choice** — Choose `logistic` or `xgboost` in the UI.
- **Segmentation** — Adjust scaler/`n_clusters` in `src/segmentation.py`.

---

## 📈 Modeling Details

- **Labels** → If missing, churn = `Recency ≥ 180 days`.
- **Models** → Logistic (baseline), XGBoost (non-linear boost).
- **Metrics** → Accuracy/AUC, confusion matrix, SHAP plots.
- **Imbalance** → Class weights (Logistic) or `scale_pos_weight` (XGBoost).

---

## 📅 Cohort & Retention

`src/analytics.py` → Monthly cohort retention matrix, useful for onboarding & engagement analysis.

---

## 🧪 Testing

```bash
pip install pytest
pytest -q
```

## 🛠 Troubleshooting

- **Streamlit fails** → Activate venv & install deps.
- **Import errors** → Run from project root.
- **SHAP slow** → Use Logistic model or sample.
- **XGBoost missing** → `pip install xgboost`.

---

## 🛤 Roadmap

- File upload in UI
- Model persistence (joblib)
- Segment-aware retention actions
- Dockerfile/devcontainer
- CI with lint/format/test

---

## 🤝 Contributing

PRs welcome! Please:

1. State the problem clearly  
2. Include repro steps or sample data  
3. Add tests & short demo (GIF/screenshot)  



