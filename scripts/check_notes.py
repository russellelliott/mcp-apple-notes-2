import lancedb
from pathlib import Path
import pandas
import time

start_time = time.time()

DATA_DIR = Path.home() / ".mcp-apple-notes"
CACHE_PATH = DATA_DIR / "last-sync.txt"

db = lancedb.connect(str(DATA_DIR / "data"))
tbl = db.open_table("notes")
df = tbl.to_pandas()

# Show cache info
print("📋 Cache Info:")
if CACHE_PATH.exists():
    with open(CACHE_PATH, 'r') as f:
        lines = f.read().strip().split('\n')
        if len(lines) >= 2:
            print(f"   Last sync: {lines[1]}")
        else:
            print(f"   Last sync: {lines[0]}")
else:
    print("   No cache yet")

print()

# Show total chunks and unique notes
print("📊 Database Stats:")
print(f"   Total chunks: {len(df)}")
unique_notes = df['title'].nunique()
print(f"   Unique notes: {unique_notes}")
print()

# Show chunks per note
print("📊 Chunks per Note (top 20):")
chunks_per_note = df.groupby('title').size().sort_values(ascending=False).head(20)
for title, count in chunks_per_note.items():
    print(f"   {count:4d} chunks  →  {title[:60]}")
print()

# show schema
print("📊 Schema:")
print(tbl.schema)
print()

# fetch a few rows (if supported):
print("📝 Sample rows (5 newest chunks; by time INSERTED, not time CREATED):")
# Last 5 rows are the newest (insertion order)
newest = df.tail(5)[['title', 'chunk_index', 'total_chunks', 'chunk_content']].reset_index(drop=True)
for idx, row in newest.iterrows():
    print(f"\n   [{idx+1}] {row['title'][:60]} (chunk {row['chunk_index']}/{row['total_chunks']})")
    print(f"       {row['chunk_content'][:80]}...")

elapsed = time.time() - start_time
print(f"\n⏱️  Completed in {elapsed:.2f}s")