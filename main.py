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
            pa.field("cluster_id", pa.string()),
            pa.field("cluster_label", pa.string()),
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
    
    def search(self, query_vector: np.ndarray, limit: int = 10) -> List[Dict[str, Any]]:
        """Vector search for similar chunks"""
        results = self.table.search(query_vector).limit(limit).to_list()
        return results
