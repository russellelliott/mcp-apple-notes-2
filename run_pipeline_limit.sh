#!/bin/bash
set -e

if [ -z "$1" ]; then
    echo "Usage: $0 <number_of_notes>"
    exit 1
fi

LIMIT=$1

# Store the root directory
ROOT_DIR="$(pwd)"

echo "Starting pipeline with note limit: $LIMIT..."

# 1. Run bun cli.ts --mode=incremental-since --max=LIMIT from server folder
echo "Running server CLI..."
cd "$ROOT_DIR/server"
bun cli.ts --mode=incremental --max="$LIMIT"

# 2. Run backend/analysis/run_bertopic.py
echo "Running BERTopic analysis..."
cd "$ROOT_DIR/backend/analysis"
python run_bertopic.py

echo "Pipeline finished successfully."
