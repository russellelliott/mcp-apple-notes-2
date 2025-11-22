#!/usr/bin/env python3

import lancedb
from pathlib import Path

DATA_DIR = Path.home() / ".mcp-apple-notes"
db = lancedb.connect(str(DATA_DIR / "data"))

print("Fixing database by recreating notes table from notes_new...")

try:
    # Get data from notes_new table
    notes_new_table = db.open_table("notes_new")
    data = notes_new_table.to_pandas()
    
    # Create notes table with the data
    db.create_table("notes", data)
    print("✅ Successfully created notes table from notes_new data")
    
    # Drop the temporary notes_new table
    # db.drop_table("notes_new")
    # print("✅ Cleaned up notes_new table")
    
    # Verify the fix
    notes_table = db.open_table("notes")
    count = len(notes_table.to_pandas())
    print(f"✅ Verified: notes table has {count} records")
    
    # Clean up test tables
    print("\nCleaning up test tables...")
    tables = db.table_names()
    test_tables = [t for t in tables if t.startswith("test-")]
    for table in test_tables:
        try:
            db.drop_table(table)
            print(f"   Dropped {table}")
        except Exception as e:
            print(f"   Failed to drop {table}: {e}")
    
    print("\n🎉 Database fixed successfully!")
    
except Exception as e:
    print(f"❌ Error fixing database: {e}")
    import traceback
    traceback.print_exc()