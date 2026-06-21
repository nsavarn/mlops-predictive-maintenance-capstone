# Capstone Summary

**MLOps Capstone: Predictive Maintenance Classification**
Author: Narendra Tiwari
Program: IIIT-B & upGrad MLOps Program

---

## Executive Summary

This capstone implements a complete MLOps pipeline for industrial Predictive Maintenance.

---

## Business Problem

Industrial equipment failures cause unplanned downtime and maintenance costs. Classifying machine state enables proactive maintenance scheduling.

---

## Technical Solution

### Data
- **train.csv**: 10,000 samples with sensor/process features
- **current.csv**: Live production data for drift monitoring
- **stress.csv**: Degraded/shifted production data (schema-valid but drifted)

### Features
| Feature | Type | Engineering |
|---------|------|-------------|
| Air_temperature | float64 | Raw |
| Process_temperature | float64 | Raw |
| Rotational_speed | float64 | Raw |
| Torque | float64 | Raw |
| Voltage | float64 | Raw |
| Current | float64 | Raw |
| Power_W | float64 | Engineered (V x I) |
| Temp_diff | float64 | Engineered (P - A) |

### Models Evaluated
1. **Dummy Classifier** (baseline)
2. **Random Forest**
3. **XGBoost**
4. **Logistic Regression**
5. **SGD Classifier**

---

## Key Metrics

### Primary Metric: macro F1
- Selected over accuracy for balanced class performance

### Per-Class F1
- Class 0 (Normal)
- Class 1 (Warning)
- Class 2 (Failure)

---

## MLOps Components

### 1. Data Validation (Pandera)
- Schema enforced across train, current, and stress datasets
- Stress data passes schema but exhibits statistical drift

### 2. Experiment Tracking (MLflow)
- All metrics, parameters, and model artifacts logged
- Run comparison for model selection

### 3. Hyperparameter Tuning (Optuna)
- Automated optimization for best model
- macro F1 as objective function

### 4. Drift Monitoring (Evidently)
- Reference vs. current production comparison
- Reference vs. stressed production comparison
- HTML reports: drift_current.html, drift_stress.html

### 5. Explainability (SHAP)
- Multiclass TreeExplainer analysis
- Per-class feature importance
- Visualization: shap_per_class.png

---

## Design Decisions

| Decision | Rationale |
|----------|----------|
| macro F1 primary | Balanced 3-class problem |
| SMOTE after split | Prevent leakage |
| Pandera schema | Production data gate |
| Per-class SHAP | Multiclass interpretability |
| Drift + SHAP | Evidence-based retraining |

---

## Generated Artifacts

- `best_model.pkl` - Best performing model
- `label_encoder.pkl` - Training encoders
- `drift_current.html` - Current data drift report
- `drift_stress.html` - Stress data drift report
- `shap_per_class.png` - SHAP visualization
- `eda_distributions.png` - EDA visualizations

---

## Conclusion

The pipeline demonstrates production-ready MLOps practices including validation, tracking, tuning, monitoring, and explainability across the ML lifecycle.

---

**Date**: June 2026
**License**: MIT
