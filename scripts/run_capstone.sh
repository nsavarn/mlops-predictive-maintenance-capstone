#!/usr/bin/env bash
set -euo pipefail

echo "=============================================="
echo "Industrial Predictive Maintenance MLOps Runner"
echo "=============================================="

if [ ! -d ".venv" ]; then
  python -m venv .venv
fi

if [ -f ".venv/Scripts/activate" ]; then
  # Windows Git Bash
  # shellcheck disable=SC1091
  source ".venv/Scripts/activate"
else
  # Linux and macOS
  # shellcheck disable=SC1091
  source ".venv/bin/activate"
fi

python -m pip install --upgrade pip
python -m pip install -r requirements.txt

mkdir -p artifacts reports

jupyter nbconvert \
  --to notebook \
  --execute notebooks/MLOps_Assignment_Narendra_Tiwari.ipynb \
  --output MLOps_Assignment_Narendra_Tiwari_executed.ipynb \
  --output-dir notebooks

echo ""
echo "Pipeline completed. Review generated outputs in artifacts/ and reports/."
