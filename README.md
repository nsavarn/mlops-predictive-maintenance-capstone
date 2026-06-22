# Industrial Predictive Maintenance

## An End-to-End MLOps Governance Framework for Detecting Machine Failure Risk, Monitoring Statistical Drift, and Triggering Evidence-Based Retraining Decisions

This repository contains a production-style MLOps governance framework for industrial predictive maintenance. The project treats the capstone as a live production crisis: preventing unplanned factory downtime by proving that structurally valid data can still become statistically dangerous.

## Business Context

Unplanned factory downtime can cost approximately Rs. 8-15 lakh per hour. The goal is to build a closed-loop governance pipeline that detects when the operating environment has moved outside the model's familiar statistical territory, even when the raw data still passes standard schema validation.

## Architecture

```text
Raw IoT Stream
  -> Pandera Quality Gate
  -> MLflow Ledger + Optuna
  -> Evidently Drift Radar
  -> SHAP Explainability
```

## Repository Layout

```text
.
├── data/
│   ├── train.csv
│   ├── current.csv
│   └── stress.csv
├── notebooks/
│   └── MLOps_Assignment_Narendra_Tiwari.ipynb
├── artifacts/
├── reports/
│   ├── architecture.md
│   ├── capstone_summary.md
│   └── model_drift_and_explainability_report.md
├── scripts/
│   ├── scaffold_repository.sh
│   └── run_capstone.sh
├── LICENSE
├── requirements.txt
├── README.md
└── .gitignore
```

## MLOps Phases

1. Structural integrity with Pandera schema checks.
2. Class balancing and optimization with stratified splitting, SMOTE, Optuna, and MLflow.
3. Statistical observability with Evidently drift reports for current and stress data.
4. Physics-informed explainability with SHAP, mechanical power, and thermal delta features.

## Recommended Environment

Use a native Python virtual environment as the primary submission path. The grading rubric focuses on Pandera validation, MLflow experimentation, Optuna tuning, Evidently monitoring, SHAP explainability, and engineering conclusions, not containerization.

Docker can be added later as a portfolio enhancement and positioned as a production deployment accelerator.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Execution

```bash
bash scripts/run_capstone.sh
```

Or launch the notebook manually:

```bash
jupyter notebook notebooks/MLOps_Assignment_Narendra_Tiwari.ipynb
```

Expected generated outputs:

```text
artifacts/eda_distributions.png
artifacts/drift_current.html
artifacts/drift_stress.html
artifacts/shap_per_class.png
artifacts/best_model.pkl
artifacts/label_encoder.pkl
```

## Governance Rules

| Rule | Why it matters |
|---|---|
| Pandera validates all datasets | Confirms structural integrity and physical bounds. |
| SMOTE runs only after the stratified split | Prevents synthetic samples from leaking into validation data. |
| SMOTE uses k_neighbors=3 | Handles rare machine-failure classes more safely. |
| Macro-F1 drives model selection | Prevents high accuracy from hiding poor rare-class performance. |
| MLflow stores metrics and artifacts | Creates an experiment ledger for model governance. |
| Evidently compares train vs current/stress | Separates schema validity from statistical stability. |
| SHAP is interpreted per class | Connects predictions to actionable maintenance diagnostics. |
