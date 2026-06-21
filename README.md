# MLOps Capstone: Predictive Maintenance System

**Author:** Narendra Tiwari  
**Program:** IIIT-B & upGrad MLOps Program  
**License:** MIT License

---

## Project Overview

This repository contains the end-to-end MLOps capstone project for building a Predictive Maintenance classification system. The project implements a production-grade ML pipeline covering data validation, model training, experiment tracking, hyperparameter tuning, drift monitoring, and explainability analysis.

The target task: **Classify machine operational state** (Normal, Warning, Failure) based on sensor readings and process measurements.

---

## Repository Structure

```
mlops-predictive-maintenance-capstone/
├── .gitignore
├── LICENSE                     # MIT License
├── README.md                   # This file
├── requirements.txt            # Pinned Python dependencies
├── MLOps_Assignment_Narendra_Tiwari.ipynb   # Main notebook (submission artifact)
├── data/
│   ├── train.csv               # Training dataset
│   ├── current.csv             # Current production data (for drift)
│   └── stress.csv              # Stressed/degraded production data
├── artifacts/
│   ├── best_model.pkl          # Serialized best model
│   ├── label_encoder.pkl       # LabelEncoder fitted on train
│   ├── shap_per_class.png      # Multiclass SHAP visualization
│   ├── drift_current.html      # Evidently drift report (current data)
│   ├── drift_stress.html       # Evidently drift report (stressed data)
│   └── eda_distributions.png   # EDA distribution plots
├── reports/
│   ├── capstone_summary.md     # Executive summary
│   └── architecture.md         # Pipeline architecture documentation
└── run_capstone.sh             # One-click execution script
```

---

## Technology Stack

- **Data Validation:** Pandera
- **Experiment Tracking:** MLflow
- **Hyperparameter Tuning:** Optuna
- **Drift Detection:** Evidently AI
- **Explainability:** SHAP (TreeExplainer)
- **Modeling:** scikit-learn (Random Forest, XGBoost, Logistic Regression, SGD)

---

## Installation

```bash
# Clone the repository
git clone https://github.com/nsavarn/mlops-predictive-maintenance-capstone.git
cd mlops-predictive-maintenance-capstone

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate      # Linux/Mac
venv\Scripts\activate         # Windows

# Install dependencies
pip install -r requirements.txt
```

---

## Quick Start

```bash
# Run the full capstone pipeline
bash run_capstone.sh

# Or manually execute the notebook
jupyter notebook MLOps_Assignment_Narendra_Tiwari.ipynb
```

---

## Pipeline Sections

### Section 1: Data Loading and Validation
- Load train.csv, current.csv, and stress.csv
- **Pandera** schema validation ensuring data integrity across all datasets
- Column types: Air_temperature (float64), Process_temperature (float64), etc.
- Stress data passes schema but is statistically drifted

### Section 2: EDA and Feature Engineering
- Distribution analysis across target classes
- Target class balance exploration
- Engineered features:
  - **Power_W** = Voltage * Current
  - **Temp_diff** = Process_temperature - Air_temperature
- Correlation analysis and summary statistics
- SMOTE applied **only after** stratified 80/20 train/validation split

### Section 3: Model Training and Evaluation
- Baseline: Dummy classifier (most_frequent strategy)
- Models: Random Forest, XGBoost, Logistic Regression, SGD
- **Primary metric: macro F1** (not accuracy)
- Per-class F1 scores reported
- **MLflow** experiment tracking with all metrics logged
- **Optuna** hyperparameter tuning for best model

### Section 4: Drift Monitoring
- **Evidently AI** drift reports comparing current/stress to training
- Data drift quantification across all features
- Two drift reports generated:
  - `drift_current.html` - Current production data
  - `drift_stress.html` - Stressed/shifted production data

### Section 5: Explainability (SHAP) and Conclusions
- **Multiclass SHAP** per-class explanations
- `shap_per_class.png` - Global SHAP feature importance
- SHAP insights connected to retraining decisions
- **Concrete retraining recommendations** based on drift + SHAP evidence

---

## Key Design Decisions

| Decision | Rationale |
|----------|----------|
| macro F1 as primary metric | Balanced performance across all 3 classes |
| SMOTE after train/validation split | Prevents data leakage |
| Pandera schema across all datasets | Data quality gate for production |
| Per-class SHAP (not global) | Multiclass interpretability |
| Drift + SHAP for retraining decision | Evidence-based lifecycle management |

---

## Execution and Outputs

Running `MLOps_Assignment_Narendra_Tiwari.ipynb` produces:

- **MLflow artifacts:** All metrics, parameters, and models logged
- **best_model.pkl:** Serialized best-performing model
- **label_encoder.pkl:** Encoders fitted on training data
- **drift_current.html / drift_stress.html:** Interactive Evidently reports
- **shap_per_class.png:** SHAP summary bar chart
- **eda_distributions.png:** EDA visualization

---

## Governance Notes

- All random seeds set for reproducibility (random_state=42)
- LabelEncoder fitted **only** on training data
- SMOTE k_neighbors=3 for stable synthetic samples
- Stress data validates schema but shows statistical drift

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Contact

**Narendra Tiwari**  
AI Architect | MLOps Specialist  
GitHub: [@nsavarn](https://github.com/nsavarn)
