#!/bin/bash
set -e

# Store the root directory of the script
ROOT_DIR="$(dirname "$(realpath "$0")")"

echo "Calculating common words..."

# Add the root directory to PYTHONPATH so python can find the 'backend' module
export PYTHONPATH="$ROOT_DIR:$PYTHONPATH"

# Run backend/scripts/common_words.py
python3 "$ROOT_DIR/backend/scripts/common_words.py" -n 100 # Words with frequency >0.10% have a histogram bar. use those. or >0.2%?
