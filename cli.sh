
#!/usr/bin/env bash
# cli.sh — run main.py's NotesIndexer for the first 10 notes
set -euo pipefail

echo "Using active Python from the environment"

echo "Indexing first 10 notes..."

python - <<'PYCODE'
import asyncio, json, sys
sys.path.insert(0, '.')
from backend.scripts.main import NotesIndexer

async def run():
    indexer = NotesIndexer()
    result = await indexer.index_notes(mode='fresh', limit=10)
    print(json.dumps(result, indent=2))

asyncio.run(run())
PYCODE

echo "Done."

