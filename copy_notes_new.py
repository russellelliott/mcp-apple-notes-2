#!/usr/bin/env python3

import lancedb
from pathlib import Path

DATA_DIR = Path.home() / ".mcp-apple-notes"
db = lancedb.connect(str(DATA_DIR / "data"))

try:
    # Open the original notes_new table
    notes_new_table = db.open_table("notes_new")
    
    # Get all data from the notes_new table
    notes_new_data = notes_new_table.to_pandas()
    
    print(f"Found {len(notes_new_data)} records in notes_new table")
    
    # Create the copy table
    copy_table_name = "notes_new_copy"
    copy_table = db.create_table(copy_table_name, notes_new_data)
    
    print(f"Successfully created copy table '{copy_table_name}' with {len(notes_new_data)} records")
    
except Exception as e:
    print(f"Error creating copy: {e}")