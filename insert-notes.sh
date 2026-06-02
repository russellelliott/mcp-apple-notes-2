#!/bin/bash
#
# Script to insert open Apple Notes into the notes_interactions table
# 
# Usage: ./insert-notes.sh
#

set -euo pipefail

echo "🔍 Getting open Notes windows..."
echo ""

# Run the insert-notes script
cd "$(dirname "${BASH_SOURCE[0]}")"
bun server/insert-notes.ts

echo ""
echo "✅ Done!"
