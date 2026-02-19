#!/bin/bash
set -e

# Store the root directory of the script
ROOT_DIR="$(dirname "$(realpath "$0")")"

echo "Displaying clusters..."

# Add the root directory to PYTHONPATH so python can find the 'backend' module
export PYTHONPATH="$ROOT_DIR:$PYTHONPATH"

# Run backend/scripts/display_clusters.py
python3 "$ROOT_DIR/backend/scripts/display_clusters.py"
