#!/usr/bin/env python3
"""
Apple Notes MCP Server - Python Edition
Shared components: NotesDatabase
"""

import os
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

import lancedb
import pyarrow as pa
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

# ============================================================================
# Configuration
# ============================================================================

DATA_DIR = Path.home() / ".mcp-apple-notes-2"
DB_PATH = DATA_DIR / "data"

# Model configuration
MODEL_NAME = "BAAI/bge-small-en-v1.5"
EMBEDDING_DIM = 384

# Device detection
if torch.backends.mps.is_available():
    DEVICE = "mps"
    print("🚀 Using Apple Silicon GPU (MPS)")
elif torch.cuda.is_available():
    DEVICE = "cuda"
    print("🚀 Using NVIDIA GPU (CUDA)")
else:
    DEVICE = "cpu"
    print("⚠️ Using CPU (no GPU acceleration)")


# ============================================================================
# Embeddings
# ============================================================================

class EmbeddingModel:
    """GPU-accelerated embedding generation"""
    def __init__(self, model_name: str = MODEL_NAME, device: str = DEVICE):
        self.device = device
        print(f"📦 Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name, device=device)
        print(f"✅ Model loaded on {device}")
    
    def embed_texts(self, texts: List[str], batch_size: int = 32, show_progress: bool = True) -> np.ndarray:
        """Generate embeddings for a list of texts"""
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=True,
            convert_to_numpy=True
        )
        return embeddings


# ============================================================================
# LanceDB Integration
# ============================================================================

class NotesDatabase:
    """LanceDB vector database for notes"""
    
    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.db = lancedb.connect(str(db_path))
        self.table = None
    
    def create_schema(self) -> pa.Schema:
        """Create PyArrow schema for notes table"""
        return pa.schema([
            pa.field("title", pa.string()),
            pa.field("content", pa.string()),
            pa.field("creation_date", pa.string()),
            pa.field("modification_date", pa.string()),
            pa.field("chunk_index", pa.int32()),
            pa.field("total_chunks", pa.int32()),
            pa.field("chunk_content", pa.string()),
            pa.field("vector", pa.list_(pa.float32(), EMBEDDING_DIM)),
            pa.field("base_topic_id", pa.string()),
            pa.field("display_topic_id", pa.string()),
            pa.field("is_split_child", pa.bool_()),
            pa.field("cluster_id", pa.string()),
            pa.field("cluster_label", pa.string()),
            pa.field("base_cluster_label", pa.string()),
            pa.field("cluster_confidence", pa.string()),
            pa.field("cluster_summary", pa.string()),
            pa.field("last_clustered", pa.string()),
        ])
    
    def get_or_create_table(self, fresh: bool = False) -> lancedb.table.Table:
        """Get existing table or create new one"""
        table_name = "notes"
        if fresh and table_name in self.db.table_names():
            self.db.drop_table(table_name)
        
        if table_name not in self.db.table_names():
            self.table = self.db.create_table(table_name, schema=self.create_schema())
        else:
            self.table = self.db.open_table(table_name)
        return self.table

    def get_interactions_db(self) -> Tuple[Any, Any]:
        """Get the interactions table from the existing DB connection"""
        table_name = "notes_interactions"
        try:
            # 1. Try to open it directly first (most efficient)
            if table_name in self.db.table_names():
                return self.db, self.db.open_table(table_name)
            
            # 2. If not found in list, try creating it
            interactions_schema = pa.schema([
                pa.field("title", pa.string()),
                pa.field("last_opened", pa.string()),
                pa.field("interaction_log", pa.string()),
            ])
            table = self.db.create_table(table_name, schema=interactions_schema)
            return self.db, table
            
        except Exception as e:
            # 3. If creation fails because it exists, just open it
            if "already exists" in str(e).lower():
                try:
                    return self.db, self.db.open_table(table_name)
                except Exception:
                    pass
            
            print(f"⚠️ Interactions DB access error: {e}")
            return None, None

    def list_all_interactions(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Retrieve all interaction records from the DB"""
        try:
            _, table = self.get_interactions_db()
            if table is None:
                print("DEBUG: Interactions table is None")
                return []
            
            # Fetch all rows
            df = table.to_pandas()

            if df.empty:
                print("DEBUG: Interaction table found but it is empty")
                return []

            # Optional: Sort by last_opened
            if "last_opened" in df.columns:
                df = df.sort_values(by="last_opened", ascending=False)

            results = df.head(limit).to_dict(orient='records')
            
            # Convert JSON string log back to Python list
            for row in results:
                log_data = row.get("interaction_log")
                if isinstance(log_data, str):
                    try:
                        row["interaction_log"] = json.loads(log_data)
                    except:
                        row["interaction_log"] = []
            return results
        except Exception as e:
            print(f"❌ Error listing interactions: {e}")
            return []

    def log_interaction(self, title: str, event_type: str) -> None:
        """Log an interaction event (opened/modified) for a note"""
        try:
            _, interactions_table = self.get_interactions_db()
            if interactions_table is None:
                return

            now = datetime.utcnow().isoformat() + "Z"
            event = {"dt": now, "type": event_type}
            safe_title = title.replace("'", "''")
            
            existing = interactions_table.search().where(f"title = '{safe_title}'").limit(1).to_list()
            
            if len(existing) > 0:
                record = existing[0]
                log = json.loads(record.get("interaction_log", "[]") or "[]")
                log.append(event)
                
                interactions_table.update(
                    where=f"title = '{safe_title}'",
                    values={"last_opened": now, "interaction_log": json.dumps(log)}
                )
            else:
                interactions_table.add([{
                    "title": title,
                    "last_opened": now,
                    "interaction_log": json.dumps([event])
                }])
        except Exception as e:
            print(f"⚠️ Failed to log interaction for '{title}': {e}")

    def log_note_opened(self, title: str) -> None:
        self.log_interaction(title, "opened")

    def log_note_modified(self, title: str) -> None:
        self.log_interaction(title, "modified")

    def get_last_opened(self, title: str) -> Optional[str]:
        try:
            _, table = self.get_interactions_db()
            if table is None: return None
            safe_title = title.replace("'", "''")
            results = table.search().where(f"title = '{safe_title}'").limit(1).to_list()
            return results[0].get("last_opened") if results else None
        except:
            return None
    
    def get_interaction_log(self, title: str) -> Optional[List[Dict]]:
        try:
            _, table = self.get_interactions_db()
            if table is None: return None
            safe_title = title.replace("'", "''")
            results = table.search().where(f"title = '{safe_title}'").limit(1).to_list()
            if results:
                return json.loads(results[0].get("interaction_log", "[]") or "[]")
            return None
        except:
            return None

# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    print("🚀 Apple Notes Interaction Tracking Demo")
    print("=" * 60)
    
    db = NotesDatabase()
    notes_table = db.get_or_create_table()
    
    print(f"📝 Notes table rows: {notes_table.count_rows()}")
    
    try:
        all_notes = notes_table.to_pandas()
        if not all_notes.empty:
            sample_titles = all_notes['title'].unique()[:3]
            for title in sample_titles:
                db.log_note_opened(title)
                db.log_note_modified(title)
                
                last_opened = db.get_last_opened(title)
                log = db.get_interaction_log(title)
                print(f"📖 '{title}' | Last: {last_opened} | Events: {len(log) if log else 0}")
        else:
            print("⚠️ No notes found in database.")
    except Exception as e:
        print(f"⚠️ Demo Error: {e}")

if __name__ == "__main__":
    main()