#!/usr/bin/env python3

import lancedb
from pathlib import Path
import os

def check_db(path_name, path):
    print(f"\nChecking database at: {path}")
    if not path.exists():
        print(f"  ❌ Path does not exist: {path}")
        return

    try:
        db = lancedb.connect(str(path))
        tables = db.table_names()
        if tables:
            print(f"  ✅ Found {len(tables)} tables:")
            for table in tables:
                print(f"    - {table}")
                try:
                    tbl = db.open_table(table)
                    # limit to 5 rows to be fast, but we want count
                    count = tbl.count_rows() 
                    # count_rows is faster than len(to_pandas()) if available, 
                    # but for older lancedb versions len(to_pandas()) is safer.
                    # Let's stick to len(to_pandas()) for compatibility or try/except
                    print(f"      Records: {count}")
                except:
                     try:
                        count = len(tbl.to_pandas())
                        print(f"      Records: {count}")
                     except Exception as e:
                        print(f"      Error reading table: {e}")
        else:
            print("  ⚠️  No tables found.")
    except Exception as e:
        print(f"  ❌ Error connecting to database: {e}")

# Check both potential locations
paths = [
    Path.home() / ".mcp-apple-notes" / "data",
]

print("🔍 Scanning for LanceDB databases...")
for p in paths:
    check_db(p.parent.name, p)
