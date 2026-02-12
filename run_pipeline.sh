#!/bin/bash
set -e

# Store the root directory
ROOT_DIR="$(pwd)"

echo "Starting pipeline..."

# 1. Run bun cli.ts --mode=incremental-since from server folder
echo "Running server CLI..."
cd "$ROOT_DIR/server"
bun cli.ts --mode=incremental-since

# 2. Run backend/analysis/run_bertopic.py
echo "Running BERTopic analysis..."
cd "$ROOT_DIR/backend/analysis"
python run_bertopic.py

echo "Pipeline finished successfully."
