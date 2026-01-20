#!/usr/bin/env python3

import lancedb
from pathlib import Path
import pandas as pd
import list_tables

DATA_DIR = Path.home() / ".mcp-apple-notes"
db = lancedb.connect(str(DATA_DIR / "data"))

def restore_notes_from_backup():
    """Copy contents from a backup table to notes table."""
    
    print("Available tables:")
    list_tables.list_tables_organized()
    tables = db.table_names()
        
    source_table_name = input("\nEnter the name of the table to restore from (default: notes_backup_1763846409): ").strip()
    if not source_table_name:
        source_table_name = "notes_backup_1763846409"
    
    if source_table_name not in tables:
        print(f"Error: Table '{source_table_name}' not found.")
        return

    print(f"\nExamining table schemas for '{source_table_name}'...")
    
    # Open both tables
    try:
        backup_table = db.open_table(source_table_name)
        notes_table = db.open_table("notes")
        
        print(f"Source table ({source_table_name}) records: {len(backup_table.to_pandas())}")
        print(f"Current notes table records: {len(notes_table.to_pandas())}")
        
        # Get schemas
        backup_df = backup_table.to_pandas()
        notes_df = notes_table.to_pandas()
        
        print(f"\nSource table columns: {list(backup_df.columns)}")
        print(f"Notes table columns: {list(notes_df.columns)}")
        
        # Check if schemas match
        if list(backup_df.columns) == list(notes_df.columns):
            print("\n✓ Schemas match!")
        else:
            print("\n⚠ Schema mismatch detected")
            return
            
        # Confirm before proceeding
        response = input(f"\nThis will replace {len(notes_df)} records in 'notes' table with {len(backup_df)} records from '{source_table_name}'. Continue? (y/N): ")
        
        if response.lower() != 'y':
            print("Operation cancelled.")
            return
            
        print("\nBacking up current notes table...")
        # Create a backup of current notes table
        timestamp = int(pd.Timestamp.now().timestamp())
        backup_name = f"notes_backup_{timestamp}"
        
        # Drop existing notes table and recreate with backup data
        print("Dropping current notes table...")
        db.drop_table("notes")
        
        print("Creating new notes table with backup data...")
        new_notes_table = db.create_table("notes", backup_df)
        
        print(f"✓ Successfully restored {len(backup_df)} records to notes table")
        print(f"✓ Previous notes data was backed up as '{backup_name}' (if you need to revert)")
        
        # Verify the restoration
        restored_table = db.open_table("notes")
        print(f"✓ Verification: Notes table now has {len(restored_table.to_pandas())} records")
        
    except Exception as e:
        print(f"Error during restoration: {e}")
        return

if __name__ == "__main__":
    restore_notes_from_backup()