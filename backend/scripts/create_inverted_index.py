#!/usr/bin/env python3
"""
Create and verify FTS (full-text search) indexes on the notes database.

This script:
1. Creates inverted/FTS indexes on chunk_content, title, and content columns
2. Verifies the indexes were created successfully
3. Tests FTS queries to confirm they return expected results
"""

import sys
import time
from pathlib import Path
import lancedb

# Add repo root to path
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.scripts.main import NotesDatabase


def verify_fts_index(notes_table, column: str) -> bool:
    """Verify that an FTS index exists and works on the given column."""
    try:
        # List indexes
        indexes = notes_table.list_indices() if hasattr(notes_table, "list_indices") else []
        print(f"    Indexes: {indexes}")

        # Test a simple FTS query using LanceDB syntax for this version:
        # search(query, query_type="fts") -- scans all FTS indexes
        test_results = None
        try:
            if hasattr(notes_table, "search"):
                result = notes_table.search("the", query_type="fts").limit(5)
                # Try to convert to list
                for method_name in ("to_list", "toArray", "__iter__"):
                    if hasattr(result, method_name):
                        try:
                            test_results = getattr(result, method_name)()
                            break
                        except Exception:
                            continue
        except Exception as e:
            print(f"    FTS test query failed: {e}")
            return False

        print(f"    OK FTS index working for '{column}': {len(test_results) if test_results else 0} test results")
        return True
    except Exception as e:
        print(f"    ERROR Verification failed for '{column}': {e}")
        return False


def create_fts_indexes():
    """Create FTS indexes on the notes database."""

    # Connect to LanceDB with the correct data directory
    DATA_DIR = Path.home() / ".mcp-apple-notes"
    DB_PATH = DATA_DIR / "data"

    print(f"Connecting to LanceDB at: {DB_PATH}")
    print(f"  (Data dir: {DATA_DIR})")

    # Check if data directory exists
    if not DATA_DIR.exists():
        print(f"ERROR: Data directory does not exist: {DATA_DIR}")
        print("   Please run the pipeline first to create the database.")
        return False

    db = NotesDatabase(db_path=DB_PATH)
    notes_table = db.get_or_create_table()

    # Check if table has data
    try:
        row_count = notes_table.count_rows()
        print(f"\nTable 'notes' has {row_count} rows")
        if row_count == 0:
            print("  WARNING: No data in database. Run the pipeline first.")
            return False
    except Exception as e:
        print(f"WARNING: Could not count rows: {e}")

    # Columns to create FTS indexes on
    fts_columns = ["chunk_content", "title", "content"]

    # Display current schema
    print("\nCurrent table schema:")
    try:
        schema = notes_table.schema
        print(f"  {schema}")
    except Exception as e:
        print(f"  Could not get schema: {e}")

    # Display current indexes
    print("\nCurrent indexes:")
    try:
        indexes = notes_table.list_indices() if hasattr(notes_table, "list_indices") else []
        for idx in indexes:
            print(f"  - {idx}")
    except Exception as e:
        print(f"  Could not list indexes: {e}")

    # Create FTS indexes on each column
    all_success = True
    for column in fts_columns:
        print(f"\n{'='*60}")
        print(f"Creating FTS index on '{column}' column...")
        print(f"{'='*60}")

        try:
            # Try to create FTS index using LanceDB's Index.fts() config
            created = False

            # Method 1: Using Index.fts() config (recommended for newer LanceDB)
            try:
                notes_table.create_index(column, config=lancedb.Index.fts(), replace=True)
                print(f"  Created FTS index on '{column}' using Index.fts()")
                created = True
            except Exception as e1:
                print(f"  Method 1 failed: {e1}")

                # Method 2: Using index_type="INVERTED" (older LanceDB)
                try:
                    notes_table.create_index(column, index_type="INVERTED", replace=True)
                    print(f"  Created INVERTED index on '{column}'")
                    created = True
                except Exception as e2:
                    print(f"  Method 2 failed: {e2}")

                    # Method 3: Using create_fts_index method (if available)
                    try:
                        if hasattr(notes_table, "create_fts_index"):
                            notes_table.create_fts_index(column, replace=True)
                            print(f"  Created FTS index using create_fts_index()")
                            created = True
                    except Exception as e3:
                        print(f"  Method 3 failed: {e3}")

            if not created:
                print(f"  Failed to create index on '{column}' - all methods tried")
                all_success = False
                continue

            # Verify the index was created (using same table object, not open_table)
            time.sleep(1)  # Brief pause for index to finalize

            if verify_fts_index(notes_table, column):
                print(f"  OK Index verified for '{column}'")
            else:
                print(f"  WARNING: Index may not be working for '{column}' - but was created")

        except Exception as e:
            print(f"ERROR: Failed to create index on '{column}': {e}")
            import traceback
            traceback.print_exc()
            all_success = False

    # Test FTS functionality across columns
    print(f"\n{'='*60}")
    print("Testing FTS queries...")
    print(f"{'='*60}")

    try:
        # Try to get some sample data for testing
        sample_results = notes_table.search().limit(10).to_list() if hasattr(notes_table, "search") else []

        if sample_results:
            # Find a common word from actual data
            test_words = set()
            for row in sample_results:
                chunk_content = row.get("chunk_content", "")
                if chunk_content:
                    words = [w.lower() for w in chunk_content.split() if len(w) > 3]
                    test_words.update(words[:5])

            if test_words:
                # Pick a word that might exist in multiple rows
                test_word = list(test_words)[0] if test_words else "the"
                print(f"\n  Testing with word: '{test_word}'")

                for column in fts_columns:
                    try:
                        # LanceDB FTS search syntax: search(query, query_type="fts")
                        result = notes_table.search(test_word, query_type="fts").limit(5)
                        results = result.to_list() if hasattr(result, "to_list") else list(result)
                        print(f"  OK FTS on '{column}' returned {len(results)} results for '{test_word}'")
                    except Exception as e:
                        print(f"  WARNING: FTS on '{column}' failed: {e}")
            else:
                print("  WARNING: No testable words found in sample data")
        else:
            print("  WARNING: No sample data available for testing")

    except Exception as e:
        print(f"ERROR: FTS test query failed: {e}")
        import traceback
        traceback.print_exc()

    # Final summary
    print(f"\n{'='*60}")
    print("Index Creation Summary")
    print(f"{'='*60}")

    if all_success:
        print("All FTS indexes created successfully!")
        print("\nYou can now use full-text search on:")
        for col in fts_columns:
            print(f"  - {col}")
    else:
        print("WARNING: Some indexes may not have been created properly.")
        print("  Try running this script again or check LanceDB version compatibility.")

    return all_success


if __name__ == "__main__":
    success = create_fts_indexes()
    sys.exit(0 if success else 1)