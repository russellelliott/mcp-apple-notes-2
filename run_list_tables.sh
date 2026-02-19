#!/bin/bash
set -e

# Store the root directory of the script
ROOT_DIR="$(dirname "$(realpath "$0")")"

echo "Listing tables in the database..."

# Add the root directory to PYTHONPATH so python can find the 'backend' module
export PYTHONPATH="$ROOT_DIR:$PYTHONPATH"

# Run backend/backup/list_tables.py
python3 "$ROOT_DIR/backend/backup/list_tables.py"
