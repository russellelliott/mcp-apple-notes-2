#!/usr/bin/env python3
"""
Create an inverted index for full-text search on the notes database.
"""

import sys
from pathlib import Path
import lancedb

# Add repo root to path
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.scripts.main import NotesDatabase

def main():
    # Connect to database
    DATA_DIR = Path.home() / ".mcp-apple-notes"
    DB_PATH = DATA_DIR / "data"
    
    print(f"📂 Connecting to LanceDB at: {DB_PATH}")
    db = NotesDatabase(db_path=DB_PATH)
    notes_table = db.get_or_create_table()
    
    # Check current schema
    print("\n📊 Current table schema:")
    try:
        schema = notes_table.schema
        print(schema)
    except Exception as e:
        print(f"Could not get schema: {e}")
    
    # Check existing indexes
    print("\n🔍 Checking existing indexes...")
    try:
        # List existing indexes
        indexes = notes_table.list_indices()
        print(f"Existing indexes: {indexes}")
    except Exception as e:
        print(f"Could not list indexes: {e}")
    
    # Create inverted index on text columns
    text_columns = ['title', 'content', 'chunk_content']
    
    for column in text_columns:
        print(f"\n🔨 Creating inverted index on '{column}' column...")
        try:
            # Create inverted index for full-text search
            # Try different syntaxes for creating inverted index
            try:
                notes_table.create_index(column, index_type="INVERTED", replace=True)
            except:
                try:
                    notes_table.create_index(column, "INVERTED", replace=True)
                except:
                    try:
                        notes_table.create_fts_index(column, replace=True)
                    except:
                        notes_table.create_index(column, replace=True)
            print(f"✅ Successfully created inverted index on '{column}'")
        except Exception as e:
            print(f"❌ Failed to create index on '{column}': {e}")
    
    # Test the FTS functionality
    print("\n🧪 Testing full-text search...")
    try:
        # Try a simple FTS query
        results = notes_table.search("crwn102", query_type="fts").limit(5).to_list()
        print(f"✅ FTS test successful! Found {len(results)} results")
        
        # Show a sample result
        if results:
            sample = results[0]
            title = sample.get('title', 'N/A')
            print(f"   Sample result: '{title}'")
    except Exception as e:
        print(f"❌ FTS test failed: {e}")
    
    print("\n🎉 Index creation complete!")

if __name__ == "__main__":
    main()