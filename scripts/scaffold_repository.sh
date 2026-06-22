#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${1:-mlops-predictive-maintenance-capstone}"

mkdir -p "${PROJECT_ROOT}/data"
mkdir -p "${PROJECT_ROOT}/notebooks"
mkdir -p "${PROJECT_ROOT}/artifacts"
mkdir -p "${PROJECT_ROOT}/reports"
mkdir -p "${PROJECT_ROOT}/scripts"

touch "${PROJECT_ROOT}/data/train.csv"
touch "${PROJECT_ROOT}/data/current.csv"
touch "${PROJECT_ROOT}/data/stress.csv"
touch "${PROJECT_ROOT}/notebooks/MLOps_Assignment_Narendra_Tiwari.ipynb"
touch "${PROJECT_ROOT}/artifacts/.gitkeep"
touch "${PROJECT_ROOT}/reports/capstone_summary.md"
touch "${PROJECT_ROOT}/requirements.txt"
touch "${PROJECT_ROOT}/README.md"
touch "${PROJECT_ROOT}/.gitignore"

echo "Scaffold created at ${PROJECT_ROOT}"
