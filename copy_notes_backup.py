#!/usr/bin/env python3

import lancedb
from pathlib import Path

DATA_DIR = Path.home() / ".mcp-apple-notes"
db = lancedb.connect(str(DATA_DIR / "data"))

try:
    # Open the original notes table
    notes_table = db.open_table("notes")
    
    # Get all data from the notes table
    notes_data = notes_table.to_pandas()
    
    print(f"Found {len(notes_data)} records in notes table")
    
    # Create the backup table
    backup_table_name = "notes_broken_backup"
    backup_table = db.create_table(backup_table_name, notes_data)
    
    print(f"Successfully created backup table '{backup_table_name}' with {len(notes_data)} records")
    
except Exception as e:
    print(f"Error creating backup: {e}")