# 🔧 Predictive Maintenance Pipeline

A reusable, config-driven machine learning pipeline for manufacturing failure prediction — validated across two structurally different datasets, with an interactive Streamlit dashboard for training and evaluation.

**🔗 Live demo:** https://predictive-maintenance-pipeline-kgktkxwtfn4sajrg8dnszo.streamlit.app/

## What this is

Most predictive maintenance projects are built around a single fixed dataset. This project instead builds a **reusable pipeline**: the same code (data cleaning, feature encoding, model training, evaluation) works across different manufacturing datasets — you just provide a config mapping your dataset's columns to the pipeline's expected schema.

This is not a single model that predicts failures for "any" factory — each dataset gets its own trained model. What generalizes is the **pipeline itself**.

## Validated on

| Dataset | Structure | Recall (failure) | Precision (failure) |
|---|---|---|---|
| AI4I 2020 Predictive Maintenance | Single-row machine snapshots | 0.71–0.78 | 0.70–0.79 |
| NASA CMAPSS Turbofan Engine Degradation | Time-series sensor readings per engine | 0.92 | 0.78 |

Both used the same pipeline code — only the config (column mapping) changed.

## Features

- Upload any CSV with a labeled failure column
- Interactive column mapping (target, numeric features, categorical features, ID columns)
- XGBoost model with class-imbalance handling (`scale_pos_weight`)
- Adjustable decision threshold (recall/precision tradeoff)
- Auto-generated feature importance chart
- Business-readable insight summary

## Tech stack

Python, pandas, scikit-learn, XGBoost, Streamlit, matplotlib

## Run it locally

```bash
git clone https://github.com/EMANFATIMA06/predictive-maintenance-pipeline.git
cd predictive-maintenance-pipeline
python -m venv venv
venv\Scripts\activate   # Windows
pip install -r requirements.txt
streamlit run app.py
```

## Project structure
├── app.py                 # Streamlit dashboard
├── pipeline/
│   ├── data_loader.py     # Config-driven data loading & encoding
│   ├── model_trainer.py   # XGBoost training with imbalance handling
│   └── evaluator.py       # Threshold-based evaluation
├── data/                  # Sample dataset (AI4I 2020)
└── requirements.txt
## Known limitations

- Requires the uploaded dataset to already have a 0/1 failure label column (datasets needing custom label engineering, like raw time-to-event data, need preprocessing first)
- Each dataset trains its own independent model — the pipeline generalizes, not a single trained model

## Author

Eman Fatima
