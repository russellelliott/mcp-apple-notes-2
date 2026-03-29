#!/bin/bash
set -e

ROOT_DIR="$(dirname "$(realpath "$0")")"
OUTPUT_FILE="${1:-cluster_stats.json}"

echo "Generating cluster stats..."

# Ensure local package imports resolve when running the script directly.
export PYTHONPATH="$ROOT_DIR:$PYTHONPATH"

python3 "$ROOT_DIR/backend/scripts/cluster_stats.py" --output "$OUTPUT_FILE"
