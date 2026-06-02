#!/usr/bin/env python3
"""
Apple Notes MCP Server - Python Edition
Shared components: NotesDatabase
"""

import os
from pathlib import Path
from typing import List, Dict, Any, Optional
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
            normalize_embeddings=True,  # Important for cosine similarity
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
            # Clustering fields
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
            print(f"🗑️ Dropped existing '{table_name}' table")
        
        if table_name not in self.db.table_names():
            # Create empty table with schema
            self.table = self.db.create_table(
                table_name,
                schema=self.create_schema()
            )
            print(f"✅ Created new '{table_name}' table")
        else:
            self.table = self.db.open_table(table_name)
            print(f"📂 Opened existing '{table_name}' table")
        
        return self.table
    
    def add_chunks(self, chunks_data: List[Dict[str, Any]], batch_size: int = 100) -> None:
        """Add chunks to the database in batches"""
        total = len(chunks_data)
        print(f"💾 Adding {total} chunks to database...")
        
        for i in range(0, total, batch_size):
            batch = chunks_data[i:i + batch_size]
            self.table.add(batch)
            print(f"  ✓ Added batch {i//batch_size + 1}/{(total-1)//batch_size + 1}")
        
        row_count = self.table.count_rows()
        print(f"✅ Database now has {row_count} chunks")
    
    def delete_note_chunks(self, title: str) -> None:
        """Delete all chunks for a specific note"""
        try:
            self.table.delete(f"title = '{title.replace(chr(39), chr(39)+chr(39))}'")
            print(f"  🗑️ Deleted chunks for '{title}'")
        except Exception as e:
            print(f"  ⚠️ Could not delete chunks for '{title}': {e}")
    
    def get_interactions_db(self) -> tuple[lancedb.LanceDB, lancedb.table.Table | None]:
         """Get or create the interactions database (separate from main notes table)"""
        try:
            interactions_path = self.db_path.parent / "data"
            interactions_db = lancedb.connect(str(interactions_path))
            
            if "notes_interactions" not in interactions_db.table_names():
                # Create interactions table schema
                interactions_schema = pa.schema([
                    pa.field("title", pa.string()),
                    pa.field("last_opened", pa.string()),
                    pa.field("interaction_log", pa.string()),  # JSON string
                ])
                interactions_db.create_table("notes_interactions", schema=interactions_schema)
                print(f"   ✅ Created notes_interactions table")
            
            interactions_table = interactions_db.open_table("notes_interactions")
            return (interactions_db, interactions_table)
        except Exception as e:
            print(f"   ⚠️ Interactions DB error: {e}")
            return (None, None)
    
    def log_interaction(self, title: str, event_type: str) -> None:
         """Log an interaction event (opened/modified) for a note"""
        try:
            interactions_db, interactions_table = self.get_interactions_db()
            if not interactions_table:
                print(f"   ⚠️ Could not access interactions table for '{title}'")
                return
            
            import json
            from datetime import datetime
            
            now = datetime.utcnow().isoformat() + "Z"
            event = {"dt": now, "type": event_type}
            
            # Check if title exists
            existing = interactions_table.search().where(f"title = '{title.replace(chr(39), chr(39)+chr(39))}'").limit(1).to_list()
            
            if len(existing) > 0:
                # Update existing record
                record = existing[0]
                log = json.loads(record.get("interaction_log", "[]") or "[]")
                log.append(event)
                
                interactions_table.update(
                    where=f"title = '{title.replace(chr(39), chr(39)+chr(39))}'",
                    values={"last_opened": now, "interaction_log": json.dumps(log)}
                )
                print(f"   ✏️ Updated interaction for '{title}' ({event_type})")
            else:
                # Insert new record
                interactions_table.add([{
                    "title": title,
                    "last_opened": now,
                    "interaction_log": json.dumps([event])
                }])
                print(f"   ➕ Created new interaction record for '{title}' ({event_type})")
            
            interactions_db.close()
        except Exception as e:
            print(f"   ⚠️ Failed to log interaction for '{title}': {e}")
    
    def log_note_opened(self, title: str) -> None:
         """Log a note open event"""
        self.log_interaction(title, "opened")
    
    def log_note_modified(self, title: str) -> None:
         """Log a note modification event"""
        self.log_interaction(title, "modified")
    
    def get_last_opened(self, title: str) -> Optional[str]:
         """Get the last_opened timestamp for a note"""
        try:
            interactions_db, interactions_table = self.get_interactions_db()
            if not interactions_table:
                return None
            
            results = interactions_table.search().where(f"title = '{title.replace(chr(39), chr(39)+chr(39))}'").limit(1).to_list()
            
            if len(results) > 0:
                last_opened = results[0].get("last_opened")
                interactions_db.close()
                return last_opened
            
            interactions_db.close()
            return None
        except Exception as e:
            print(f"   ⚠️ Error getting last_opened for '{title}': {e}")
            return None
    
    def get_interaction_log(self, title: str) -> Optional[list[dict]]:
         """Get the full interaction log for a note"""
        try:
            interactions_db, interactions_table = self.get_interactions_db()
            if not interactions_table:
                return None
            
            results = interactions_table.search().where(f"title = '{title.replace(chr(39), chr(39)+chr(39))}'").limit(1).to_list()
            
            if len(results) > 0:
                log = json.loads(results[0].get("interaction_log", "[]") or "[]")
                interactions_db.close()
                return log
            
            interactions_db.close()
            return None
        except Exception as e:
            print(f"   ⚠️ Error getting interaction log for '{title}': {e}")
            return None


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    """Demo script to show interaction tracking functionality"""
    print("🚀 Apple Notes Interaction Tracking Demo")
    print("=" * 60)
    
    # Initialize database
    db = NotesDatabase()
    notes_table = db.get_or_create_table()
    
    print(f"\n📊 Database path: {DB_PATH}")
    print(f"📝 Notes table rows: {notes_table.count_rows()}")
    
    # Demo: Log some interactions for existing notes
    print("\n📝 Demo: Logging interactions for sample notes...")
    
    # Get some sample note titles
    try:
        sample_notes = notes_table.to_pandas()
        if not sample_notes.empty:
            sample_titles = sample_notes['title'].unique()[:5]
            print(f"\n📋 Sample notes: {list(sample_titles)}")
            
            for title in sample_titles:
                # Log an "opened" event
                db.log_note_opened(title)
                
                # Log a "modified" event
                db.log_note_modified(title)
                
                # Get the interaction log
                last_opened = db.get_last_opened(title)
                interaction_log = db.get_interaction_log(title)
                
                print(f"\n   📖 Note: '{title}'")
                print(f"      Last opened: {last_opened}")
                print(f"      Interaction log: {interaction_log}")
        else:
            print("\n⚠️ No notes in database to demo with")
    except Exception as e:
        print(f"\n⚠️ Error during demo: {e}")
    
    print("\n✅ Demo complete!")


if __name__ == "__main__":
    main()
