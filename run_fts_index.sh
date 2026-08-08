#!/bin/bash
# Create and verify FTS indexes on the notes database
# This script creates inverted indexes for full-text search and verifies they work

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "=========================================="
echo "  Creating FTS Indexes for Notes Database"
echo "=========================================="
echo ""

python3 backend/scripts/create_inverted_index.py

echo ""
echo "Done! FTS indexes are ready."
echo "You can now use full-text search with the /search endpoint."