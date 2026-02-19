#!/bin/bash
set -e

# Store the root directory
ROOT_DIR="$(pwd)"

echo "Starting BERTopic analysis..."

# Run backend/analysis/run_bertopic.py
cd "$ROOT_DIR/backend/analysis"
python run_bertopic.py

echo "Clustering finished successfully."
