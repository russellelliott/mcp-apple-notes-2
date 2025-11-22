#!/usr/bin/env python3

import lancedb
from pathlib import Path

DATA_DIR = Path.home() / ".mcp-apple-notes"
db = lancedb.connect(str(DATA_DIR / "data"))

print("Available tables:")
try:
    tables = db.table_names()
    if tables:
        for table in tables:
            print(f"  - {table}")
            try:
                tbl = db.open_table(table)
                count = len(tbl.to_pandas())
                print(f"    Records: {count}")
            except Exception as e:
                print(f"    Error reading table: {e}")
    else:
        print("  No tables found")
except Exception as e:
    print(f"Error listing tables: {e}")

print(f"\nDatabase path: {DATA_DIR / 'data'}")