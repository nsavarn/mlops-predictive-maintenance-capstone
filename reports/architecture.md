# MLOps Predictive Maintenance — Pipeline Architecture

> **Project:** UpGrad MLOps Capstone — Predictive Maintenance for Heavy-Equipment Manufacturing  
> **Author:** Narendra Tiwari  
> **Last Updated:** June 2026  

---

## 1. High-Level Pipeline Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                     MLOps Predictive Maintenance Pipeline            │
├──────────────┬──────────────┬──────────────┬────────────┬───────────┤
│  Data Layer  │  Training    │  Registry &  │ Monitoring │ Explain-  │
│              │  Layer       │  Serving     │ Layer      │ ability   │
├──────────────┼──────────────┼──────────────┼────────────┼───────────┤
│ train.csv    │ Pandera      │ MLflow Model │ Evidently  │ SHAP      │
│ current.csv  │ Validation   │ Registry     │ Drift      │ TreeExp-  │
│ stress.csv   │ SMOTE        │ best_model   │ Reports    │ lainer    │
│              │ MLflow       │ .pkl         │            │ Per-class │
│              │ Optuna       │              │            │ 4-panel   │
└──────────────┴──────────────┴──────────────┴────────────┴───────────┘
```

---

## 2. Detailed Stage-by-Stage Architecture

### Stage 1: Data Ingestion & Schema Validation

```
[Raw CSVs]
    │
    ├── train.csv   ──┐
    ├── current.csv ──┼──► Pandera SchemaModel ──► Pass/Fail
    └── stress.csv  ──┘        │
                               ▼
                    Feature Engineering
                    ├── Power_W = Torque_Nm × (RPM × 2π/60)
                    └── Temp_diff = Process_Temp − Air_Temp
```

**Key Design Decisions:**
- Pandera schema defined once, applied to ALL three datasets
- Schema checks: dtypes, nullable=False, value ranges (RPM > 0, Torque > 0)
- `stress.csv` PASSES schema but FAILS distribution checks → valid but drifted
- Feature engineering applied uniformly to train, current, and stress

**Artifacts Generated:**
- `eda_distributions.png` — class distribution + key feature histograms

---

### Stage 2: Experiment Tracking & Model Selection

```
[Validated train.csv]
    │
    ▼
LabelEncoder.fit(y_train) ──► label_encoder.pkl  [saved]
    │
    ▼
Stratified Train/Val Split (80/20)
    │
    ├── X_train, y_train ──► SMOTE(k_neighbors=3) ──► X_train_res, y_train_res
    │                              [ONLY on training split]
    └── X_val, y_val    ──► untouched (no leakage)
    │
    ▼
┌─────────────────────────────────────────────────┐
│              MLflow Experiment Tracking          │
│                                                 │
│  ┌──────────┐ ┌──────┐ ┌───────────┐ ┌──────┐  │
│  │ Random   │ │ Extra│ │ Gradient  │ │ XG-  │  │
│  │ Forest   │ │ Trees│ │ Boosting  │ │Boost │  │
│  └────┬─────┘ └──┬───┘ └─────┬─────┘ └──┬───┘  │
│       └──────────┴───────────┴──────────┘      │
│                        │                        │
│              Log: macro_f1, per-class F1         │
│              Log: model name, params             │
└─────────────────────────────────────────────────┘
    │
    ▼
Best Model = argmax(macro_f1)  [NOT accuracy]
    │
    ▼
Optuna Hyperparameter Tuning (50 trials)
    │
    ▼
MLflow Model Registry ──► best_model.pkl  [saved]
```

**Key Design Decisions:**
- `macro_f1` is the SOLE selection criterion (accuracy is logged but NOT used)
- SMOTE applied AFTER split to prevent data leakage
- `k_neighbors=3` explicitly set for SMOTE (handles rare TWF class with few samples)
- LabelEncoder fitted ONLY on `y_train`; same encoder transforms current/stress
- Optuna maximises macro_f1 on validation set
- All 4 models + tuned model logged to MLflow with full metric suite

**Artifacts Generated:**
- `best_model.pkl` — serialised tuned winning model
- `label_encoder.pkl` — fitted LabelEncoder for inference
- MLflow `mlruns/` directory with all experiment runs

---

### Stage 3: Drift Detection & Monitoring

```
[train.csv]  ──► Reference Dataset
    │
    ├──────────────────────────────────────┐
    │                                      │
    ▼                                      ▼
[current.csv]                         [stress.csv]
    │                                      │
    ▼                                      ▼
Evidently DataDriftPreset           Evidently DataDriftPreset
    │                                      │
    ▼                                      ▼
drift_current.html                  drift_stress.html
[Expected: LOW drift]               [Expected: HIGH drift]
[Stable production data]            [Heavy-load distribution shift]
```

**Key Design Decisions:**
- `train.csv` is ALWAYS the reference; current/stress are test datasets
- `current.csv` validates that pipeline works on stable data (baseline check)
- `stress.csv` intentionally drifted — schema-valid but distribution-shifted
- Drift ≠ invalidity: Pandera checks schema; Evidently checks distribution
- Retraining decision: triggered ONLY when both drift AND SHAP evidence align
- Per-feature drift scores analysed to identify operationally critical features

**Artifacts Generated:**
- `reports/drift_current.html` — Evidently HTML report (current vs train)
- `reports/drift_stress.html` — Evidently HTML report (stress vs train)

---

### Stage 4: Explainability (SHAP)

```
[best_model.pkl]
    │
    ▼
shap.TreeExplainer(model)
    │
    ▼
SHAP values computed on X_val
    │
    ▼
Multiclass SHAP (shape: [n_samples, n_features, n_classes])
    │
    ▼
4-Panel Figure (one subplot per failure class):
    ├── Panel 1: No Failure  — top features driving normal operation
    ├── Panel 2: Heat Failure (HF) — thermal feature dominance
    ├── Panel 3: Power Failure (PWF) — power/torque feature importance
    ├── Panel 4: Tool Wear Failure (TWF) — tool_wear feature dominance
    │   (and/or OSF, RNF depending on class presence)
    │
    ▼
shap_per_class.png  [saved]
```

**Key Design Decisions:**
- `TreeExplainer` used (not KernelExplainer) for computational efficiency with tree models
- SHAP values computed per-class (not collapsed to single importance score)
- Per-class beeswarm/bar plots reveal which features matter for each failure mode
- `Power_W` and `Temp_diff` engineered features expected to rank highly
- SHAP insights cross-referenced with drift findings to justify retraining decisions

**Artifacts Generated:**
- `shap_per_class.png` — 4-panel multiclass SHAP summary plot

---

### Stage 5: Conclusions & Engineering Decisions

```
[Model Performance] + [Drift Analysis] + [SHAP Insights]
            │
            ▼
    Evidence-Based Conclusions:
    ├── Best model identified + macro_f1 justified
    ├── Accuracy misleading → imbalanced classes explanation
    ├── TWF weakest class → data scarcity, not tuning issue
    ├── stress.csv drift impact on operationally critical features
    └── Actionable recommendation: trigger retraining when
        macro_f1 drops below threshold AND drift detected in
        Power_W or Temp_diff
```

---

## 3. Artifact Inventory

| Artifact | Stage | Format | Description |
|---|---|---|---|
| `eda_distributions.png` | Stage 1 | PNG | Class distribution + feature histograms |
| `label_encoder.pkl` | Stage 2 | Pickle | LabelEncoder fitted on y_train only |
| `best_model.pkl` | Stage 2 | Pickle | Tuned winning model (Optuna-optimised) |
| `mlruns/` | Stage 2 | Directory | All MLflow experiment runs & metrics |
| `drift_current.html` | Stage 3 | HTML | Evidently report: current vs train |
| `drift_stress.html` | Stage 3 | HTML | Evidently report: stress vs train |
| `shap_per_class.png` | Stage 4 | PNG | 4-panel multiclass SHAP summary |

---

## 4. Technology Stack

| Component | Technology | Version |
|---|---|---|
| Data Validation | Pandera | 0.18.x |
| Class Imbalance | imbalanced-learn (SMOTE) | 0.11.x |
| Experiment Tracking | MLflow | 2.12.x |
| Hyperparameter Tuning | Optuna | 3.5.x |
| Drift Detection | Evidently | 0.4.x |
| Model Explainability | SHAP | 0.44.x |
| Base Models | scikit-learn, XGBoost | 1.4.x / 2.0.x |
| Visualisation | matplotlib, seaborn | 3.8.x / 0.13.x |

---

## 5. Critical Design Rules (Anti-Pattern Prevention)

| Rule | Rationale |
|---|---|
| SMOTE AFTER split | Prevents synthetic samples from leaking into validation set |
| macro_f1 as primary metric | Accuracy is misleading on imbalanced multi-class data |
| LabelEncoder fitted on train only | Prevents label mapping inconsistencies across datasets |
| stress.csv = drifted, not invalid | Schema validity ≠ distributional stability |
| Retraining requires dual evidence | Drift alone insufficient; needs SHAP feature alignment |
| k_neighbors=3 in SMOTE | Handles rare TWF class (very few real samples) |

---

## 6. Pipeline Execution

```bash
# One-command full pipeline execution
bash run_capstone.sh

# Or step-by-step:
jupyter nbconvert --to notebook --execute \
  MLOps_Assignment_Narendra_Tiwari.ipynb \
  --output MLOps_Assignment_Narendra_Tiwari_executed.ipynb
```

---

## 7. MLflow Experiment Structure

```
mlruns/
├── 0/                          # Default experiment
│   └── meta.yaml
└── <experiment_id>/            # MLOps_Capstone_PredMaint
    ├── meta.yaml
    └── <run_id>/               # One run per model
        ├── artifacts/
        │   └── model/          # Logged model artifacts
        ├── metrics/
        │   ├── macro_f1
        │   ├── accuracy
        │   ├── f1_no_failure
        │   ├── f1_heat_failure
        │   ├── f1_power_failure
        │   └── f1_twf
        ├── params/
        │   ├── model_name
        │   └── <hyperparams>
        └── tags/
```

---

*This architecture document is part of the UpGrad MLOps Capstone submission by Narendra Tiwari.*
*Repository: https://github.com/nsavarn/mlops-predictive-maintenance-capstone*
