#!/usr/bin/env python3
"""
Test FTS functionality directly to find the correct syntax.
"""

import sys
from pathlib import Path
import lancedb

# Add repo root to path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from main import NotesDatabase

def test_fts_syntax():
    # Connect to database
    DATA_DIR = Path.home() / ".mcp-apple-notes-2"
    DB_PATH = DATA_DIR / "data"
    
    print(f"📂 Connecting to LanceDB at: {DB_PATH}")
    db = NotesDatabase(db_path=DB_PATH)
    notes_table = db.get_or_create_table()
    
    query = "crwn102"
    print(f"🧪 Testing different FTS syntaxes for query: '{query}'")
    
    # Test different FTS syntaxes
    syntaxes = [
        ('notes_table.search(query, "fts")', lambda: notes_table.search(query, "fts")),
        ('notes_table.search(query, query_type="fts")', lambda: notes_table.search(query, query_type="fts")),
        ('notes_table.search(query, "fts", "chunk_content")', lambda: notes_table.search(query, "fts", "chunk_content")),
        ('notes_table.search(query, mode="fts")', lambda: notes_table.search(query, mode="fts")),
        ('notes_table.search(query, fts=True)', lambda: notes_table.search(query, fts=True)),
    ]
    
    for syntax_name, syntax_func in syntaxes:
        print(f"\n🔧 Trying: {syntax_name}")
        try:
            result = syntax_func().limit(5).to_list()
            print(f"✅ SUCCESS! Found {len(result)} results")
            if result:
                sample = result[0]
                title = sample.get('title', 'N/A')
                print(f"   Sample: '{title}'")
                return syntax_func  # Return the working function
        except Exception as e:
            print(f"❌ Failed: {e}")
    
    print("\n⚠️ No FTS syntax worked!")
    return None

if __name__ == "__main__":
    test_fts_syntax()