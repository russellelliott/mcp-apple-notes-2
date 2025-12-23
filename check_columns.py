import lancedb
from pathlib import Path

DATA_DIR = Path.home() / ".mcp-apple-notes"
DB_PATH = DATA_DIR / "data"
TABLE_NAME = "notes"

db = lancedb.connect(DB_PATH)
table = db.open_table(TABLE_NAME)
df = table.to_pandas()
print(df.columns)
