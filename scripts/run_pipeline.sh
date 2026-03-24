#!/bin/bash
# run_pipeline.sh
# ---------------
# Executes the full data pipeline in order by running each notebook
# non-interactively using nbconvert.
#
# This satisfies the brief's requirement for automation scripts and
# makes the pipeline reproducible from a single command.
#
# Usage:
#   chmod +x scripts/run_pipeline.sh
#   ./scripts/run_pipeline.sh
#
# Requirements: jupyter, nbconvert must be installed (included in requirements.txt)

set -e  # exit immediately if any command fails

echo "============================================"
echo "Financial Signal Agent — Pipeline Runner"
echo "============================================"
echo "Started: $(date)"
echo ""

# Check that Docker services are running before executing notebooks
echo "[1/4] Checking Docker services..."
docker-compose ps | grep -E "(postgres|mongo)" | grep -v "Exit" > /dev/null \
  || { echo "ERROR: Docker services not running. Run 'docker-compose up -d' first."; exit 1; }
echo "      Docker services OK"
echo ""

# Run Notebook 1 — Data Ingestion
echo "[2/4] Running 01_data_ingestion.ipynb..."
jupyter nbconvert \
  --to notebook \
  --execute \
  --inplace \
  --ExecutePreprocessor.timeout=600 \
  notebooks/01_data_ingestion.ipynb
echo "      Done."
echo ""

# Run Notebook 2 — Pipeline (transform + embed + lineage)
echo "[3/4] Running 02_pipeline.ipynb..."
jupyter nbconvert \
  --to notebook \
  --execute \
  --inplace \
  --ExecutePreprocessor.timeout=600 \
  notebooks/02_pipeline.ipynb
echo "      Done."
echo ""

# Strip outputs from notebooks before any Git commits
echo "[4/4] Stripping notebook outputs for clean Git diff..."
nbstripout notebooks/01_data_ingestion.ipynb
nbstripout notebooks/02_pipeline.ipynb
nbstripout notebooks/03_agent_demo.ipynb
echo "      Done."
echo ""

echo "============================================"
echo "Pipeline complete. $(date)"
echo "Next: open notebooks/03_agent_demo.ipynb"
echo "  or: uvicorn api.app:app --reload --port 8000"
echo "============================================"
