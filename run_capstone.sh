#!/bin/bash
# run_capstone.sh - One-click setup and execution script
# MLOps Predictive Maintenance Capstone
# Author: Narendra Tiwari

set -e

echo "========================================"
echo "MLOps Predictive Maintenance Capstone"
echo "========================================"

# Step 1: Create and activate virtual environment
echo "[1/4] Setting up Python environment..."
if [ ! -d "venv" ]; then
    python -m venv venv
    echo "Virtual environment created."
else
    echo "Virtual environment already exists."
fi

# Activate virtual environment
if [ -f "venv/Scripts/activate" ]; then
    # Windows
    source venv/Scripts/activate
else
    # Linux/Mac
    source venv/bin/activate
fi
echo "Virtual environment activated."

# Step 2: Install dependencies
echo "[2/4] Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt
echo "Dependencies installed."

# Step 3: Create required directories
echo "[3/4] Creating directories..."
mkdir -p data artifacts reports
cp models_anon_train.csv data/train.csv 2>/dev/null || echo "Warning: train.csv not found"
cp current.csv data/current.csv 2>/dev/null || echo "Warning: current.csv not found"
cp stress.csv data/stress.csv 2>/dev/null || echo "Warning: stress.csv not found"
echo "Directories created."

# Step 4: Run the notebook
echo "[4/4] Executing MLOps capstone notebook..."
jupyter nbconvert --to notebook --execute MLOps_Assignment_Narendra_Tiwari.ipynb
echo "Notebook execution completed."

echo ""
echo "========================================"
echo "Capstone pipeline completed successfully!"
echo "Check artifacts/ for generated outputs."
echo "========================================"
