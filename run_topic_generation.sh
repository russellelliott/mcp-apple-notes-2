#!/bin/bash
set -e

# Store the root directory
ROOT_DIR="$(pwd)"

echo "Running topic-generation utilities (load saved model)..."

cd "$ROOT_DIR/backend/analysis"
python load_topic_model.py "$@"

echo "Topic generation step finished."
