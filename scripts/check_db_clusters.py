import lancedb
import pandas as pd
from pathlib import Path
from collections import Counter

DATA_DIR = Path.home() / ".mcp-apple-notes"
DB_PATH = DATA_DIR / "data"
print(f"Opening LanceDB at: {DB_PATH}")

db = lancedb.connect(DB_PATH)
try:
    table = db.open_table('notes')
except Exception as e:
    print(f"Failed to open notes table: {e}")
    raise SystemExit(1)

df = table.to_pandas()
print(f"Total rows: {len(df)}")
cols = set(df.columns.tolist())
print(f"Available columns: {sorted(list(cols))}")

# Rows with hierarchical display ids (contain a dot)
if 'display_topic_id' in cols:
    df_with_dot = df[df['display_topic_id'].astype(str).str.contains('\.')]
    print(f"Rows with '.' in display_topic_id: {len(df_with_dot)}")
else:
    df_with_dot = df.iloc[0:0]
    print("Column 'display_topic_id' not present in table")

# Rows marked as split children (fallback to display_topic_id containing a dot)
if 'is_split_child' in cols:
    split_rows = df[df['is_split_child'] == True]
    print(f"Rows with is_split_child=True: {len(split_rows)}")
else:
    if 'display_topic_id' in cols:
        split_rows = df[df['display_topic_id'].astype(str).str.contains('\.')]
        print("Column 'is_split_child' not present; inferring split children from display_topic_id containing '.'")
        print(f"Inferred split children: {len(split_rows)}")
    else:
        split_rows = df.iloc[0:0]
        print("Neither 'is_split_child' nor 'display_topic_id' present; cannot infer split children")

# Show a few examples using only columns that exist
desired = ['title','chunk_index','base_topic_id','display_topic_id','cluster_label','subcluster_label']
sample_cols = [c for c in desired if c in cols]
print('\nSample hierarchical rows (columns present):', sample_cols)
if not df_with_dot.empty and sample_cols:
    sample = df_with_dot.head(10)[sample_cols]
    print(sample.to_string(index=False))
else:
    print('No hierarchical rows to sample or no sample columns available')

# Check consistency: cluster_label equals subcluster_label when is_split_child True
if 'cluster_label' in cols and 'subcluster_label' in cols and not split_rows.empty:
    mismatch = split_rows[split_rows['cluster_label'] != split_rows['subcluster_label']]
    print(f"\nSplit rows where cluster_label != subcluster_label: {len(mismatch)}")
    if len(mismatch) > 0:
        cols_to_show = [c for c in ['title','chunk_index','display_topic_id','cluster_label','subcluster_label'] if c in cols]
        print(mismatch.head(10)[cols_to_show].to_string(index=False))
else:
    print("Skipping mismatch check: required columns missing or no split rows")

# Summary of display_topic_id distribution
if 'display_topic_id' in cols:
    dist = Counter(df['display_topic_id'].astype(str).tolist())
    most_common = dist.most_common(20)
    print('\nTop 20 display_topic_id counts:')
    for k,v in most_common:
        print(f"{k}: {v}")
else:
    print("Cannot compute display_topic_id distribution; column missing")

print('\nDone.')
