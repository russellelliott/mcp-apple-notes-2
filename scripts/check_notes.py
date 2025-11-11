import lancedb
from pathlib import Path
import pandas
db = lancedb.connect(str(Path.home() / ".mcp-apple-notes-2" / "data"))
tbl = db.open_table("notes")
# show schema
print(tbl.schema)
# fetch a few rows (if supported):
print(tbl.to_pandas().head())