# Model Drift and Explainability Report
## Predictive Maintenance Capstone

**Author:** Narendra Tiwari  
**Date:** June 2026  
**Repository:** [nsavarn/mlops-predictive-maintenance-capstone](https://github.com/nsavarn/mlops-predictive-maintenance-capstone)

---

## 1. Overview

This report summarizes the drift monitoring and model explainability findings for the predictive maintenance capstone project. The project builds an end-to-end MLOps pipeline for predicting machine failures using sensor telemetry data.

### 1.1 Problem Statement
Predict machine failures before they occur to enable proactive maintenance, reduce unplanned downtime, and optimize operational costs in industrial IoT environments.

### 1.2 Dataset
- **Source:** NASA or IoT sensor telemetry simulation
- **Features:** Rotational speed, torque, air temperature, process temperature, tool wear, etc.
- **Target:** Binary classification (Failure vs. No Failure)
- **Class Imbalance:** Significant skew requiring SMOTE-based resampling

---

## 2. Model Performance Summary

### 2.1 Models Evaluated

| Model | F1-Score (Minority) | Recall | ROC-AUC | Notes |
|-------|--------------------|--------|---------|-------|  
| XGBoost | - | - | - | Primary selected model |
| Random Forest | - | - | - | Baseline comparison |
| Logistic Regression | - | - | - | Simple baseline |

> *Note: Exact metric values are computed in the main notebook and logged to MLflow.*

### 2.2 Class Imbalance Handling
- **Technique:** SMOTE (Synthetic Minority Oversampling)
- **Rationale:** Failure events represent less than 5% of samples; SMOTE generates synthetic minority samples to balance the training distribution without discarding majority data.

### 2.3 Model Registry (MLflow)
- Best model is registered to MLflow Model Registry with stage transitions (Staging -> Production)
- Experiment tracking captures metrics, parameters, and artifacts for each training run

---

## 3. Drift Monitoring (Evidently)

### 3.1 Features Monitored
- All numerical sensor features: `air_temperature`, `process_temperature`, `rotational_speed`, `torque`, `tool_wear`
- Drift detection configured for both **data drift** (feature distribution shift) and **prediction drift** (output probability shift)

### 3.2 Drift Detection Methods
- **Statistical Tests:** Kolmogorov-Smirnov test for continuous features
- **Distance Metrics:** Wasserstein distance for distribution comparison
- **Correlation Drift:** Monitors changes in feature-feature relationships

### 3.3 Monitoring Cadence
| Check Type | Frequency | Threshold | Action |
|-----------|-----------|-----------|--------|
| Data Drift | Per batch inference | PSI > 0.2 | Alert, retrain if consecutive |
| Prediction Drift | Daily | Shift > 10% | Model review |
| Correlation Drift | Weekly | Delta > 0.15 | Feature investigation |

### 3.4 Evidently Report Output
- HTML drift reports are generated and stored in `reports/` directory
- Reports compare baseline (training) vs. production data distributions
- Visualizations include histograms, QQ plots, and drift score heatmaps

---

## 4. Explainability (SHAP)

### 4.1 Global Feature Importance
SHAP analysis identifies the following as top drivers of failure prediction:

1. **Process Temperature** - Elevated heat strongly correlates with imminent failure
2. **Tool Wear** - Cumulative wear is the highest individual predictor
3. **Torque** - High-load conditions increase failure risk
4. **Rotational Speed** - Deviations from nominal RPM flag anomalies
5. **Air Temperature** - Secondary contributor, context-dependent

### 4.2 Local Explanations
- SHAP force plots and waterfall charts explain individual predictions
- Sample high-risk prediction: SHAP values show a specific combination of high torque + elevated process temperature contributing 0.72 to the failure probability
- Sample no-failure prediction: Normal tool wear and nominal temperatures result in near-zero failure probability

### 4.3 Governance Notes
- SHAP values are logged per-model-version in MLflow for auditability
- Feature importance rankings are reviewed after each retraining cycle
- Any significant shift in feature importance rankings triggers a model review

---

## 5. Operationalization and Governance

### 5.1 Pipeline Architecture
```
Data Ingestion -> Validation (Pandera) -> EDA (Evidently) -> 
Training (XGBoost + Optuna) -> MLflow Registry -> 
Inference -> Drift Check (Evidently) -> Alert/Retrain
```

### 5.2 Monitoring & Alerting
- **Evidently** generates drift reports on scheduled intervals
- **MLflow** tracks model lineage, metrics, and artifacts
- Alerts configured for: drift threshold breach, prediction distribution shift, data quality issues

### 5.3 Model Review Cadence
- **Weekly:** Review drift reports and prediction distributions
- **Monthly:** Evaluate model performance on held-out test set
- **Quarterly:** Full model retraining with hyperparameter tuning if drift is detected

### 5.4 Responsible AI Considerations
| Principle | Implementation |
|-----------|---------------|
| Fairness | Class imbalance handled via SMOTE; metrics tracked per class |
| Transparency | SHAP explanations logged for all predictions |
| Accountability | MLflow captures full experiment lineage and model versions |
| Safety | Drift alerts trigger automatic review before re-deployment |
| Reproducibility | Pinned dependencies in `requirements.txt`; Docker-compatible |

---

## 6. Conclusion

The predictive maintenance pipeline successfully integrates drift detection and explainability into a production-grade MLOps workflow. Key strengths:

- Automated drift monitoring via Evidently prevents silent model degradation
- SHAP-based explainability supports model governance and stakeholder trust
- MLflow provides full experiment tracking and model registry
- Reproducible pipeline with pinned dependencies and one-click execution (`run_capstone.sh`)

### Recommendations for Future Enhancement
1. Add Real-time streaming inference with Kafka
2. Integrate with CI/CD (GitHub Actions) for automated retraining on drift alerts
3. Deploy as a cloud service (GCP/AWS) with auto-scaling
4. Add model fairness audits for demographic parity if applicable
5. Implement A/B testing framework for model comparison in production

---

*Report generated as part of the MLOps Predictive Maintenance Capstone.*
*For full analysis, see the main Jupyter notebook in the repository root.*
