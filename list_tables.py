import lancedb
from pathlib import Path
import re

# Path to your data folder
db_path = Path.home() / ".mcp-apple-notes" / "data"

def extract_timestamp(name):
    """Finds the Unix timestamp in a string."""
    match = re.search(r'(\d{10,})', name)
    return int(match.group(1)) if match else 0

def get_sort_key(name):
    """
    Returns a tuple (tier, timestamp) to define the sorting order.
    Tier 3: The active 'notes' table (Top)
    Tier 2: Standard backups with timestamps (Sorted newest first)
    Tier 1: Miscellaneous tables (No timestamp, not 'test')
    Tier 0: Anything with 'test' in the name (Bottom)
    """
    lname = name.lower()
    ts = extract_timestamp(name)
    
    # Tier 0: Test tables (Lowest priority)
    if "test" in lname:
        return (0, ts)
    
    # Tier 3: Active notes table (Highest priority)
    if name == "notes":
        return (3, 0)
    
    # Tier 2: Dated backups
    if ts > 0:
        return (2, ts)
    
    # Tier 1: Other non-test, non-dated tables (e.g. notes_new_copy)
    return (1, 0)

def list_tables_organized():
    print(f"📂 Scanning tables (Sorted by date, Tests at bottom) in: {db_path}\n")
    
    if not db_path.exists():
        print("❌ Directory not found.")
        return

    db = lancedb.connect(str(db_path))
    
    # 1. Find all .lance folders
    physical_tables = [tp.stem for tp in db_path.glob("*.lance")]
    
    # 2. Sort using our custom logic (Descending)
    # This puts Tier 3 at the top and Tier 0 at the bottom.
    # Within Tier 2, it puts the largest timestamp (newest date) at the top.
    physical_tables.sort(key=get_sort_key, reverse=True)

    # 3. Display results
    print(f"{'Table Name':<45} | {'Records':<10}")
    print("-" * 60)

    for table_name in physical_tables:
        if table_name.startswith('.'): continue
        try:
            tbl = db.open_table(table_name)
            count = tbl.count_rows()
            print(f"{table_name:<45} | {count:<10}")
        except Exception:
            print(f"{table_name:<45} | ❌ Error opening table")

if __name__ == "__main__":
    list_tables_organized()