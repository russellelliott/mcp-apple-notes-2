#!/usr/bin/env python3
"""
Debug vector search to understand why it's not finding semantic matches.
"""

import sys
from pathlib import Path
import numpy as np

# Add repo root to path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from main import NotesDatabase, EmbeddingModel

def debug_vector_search():
    # Connect to database
    DATA_DIR = Path.home() / ".mcp-apple-notes"
    DB_PATH = DATA_DIR / "data"
    
    print(f"📂 Connecting to LanceDB at: {DB_PATH}")
    db = NotesDatabase(db_path=DB_PATH)
    notes_table = db.get_or_create_table()
    
    # Initialize embedding model
    embedding_model = EmbeddingModel()
    
    query = "crwn102"
    print(f"🧪 Debug vector search for: '{query}'")
    
    # Compute query embedding
    query_embedding = embedding_model.embed_texts([query], show_progress=False)[0].tolist()
    print(f"📦 Query embedding shape: {len(query_embedding)}")
    
    # Test different similarity thresholds
    thresholds = [0.001, 0.01, 0.05, 0.1, 0.2]
    
    for threshold in thresholds:
        print(f"\n🔍 Testing threshold: {threshold}")
        
        # Get vector search results
        try:
            vector_results = notes_table.search(query_embedding).limit(20).to_list()
            print(f"   Raw vector results: {len(vector_results)}")
            
            # Analyze distances and similarities
            valid_results = 0
            for i, chunk in enumerate(vector_results[:5]):  # Show first 5
                distance = chunk.get("_distance", 0)
                try:
                    distance = float(distance)
                    # Convert distance to cosine similarity (approximate)
                    cosine_sim = max(0.0, 1.0 - (distance * distance / 2.0))
                    
                    title = chunk.get("title", "<untitled>")
                    
                    print(f"   [{i+1}] '{title[:50]}...'")
                    print(f"       Distance: {distance:.4f}, Similarity: {cosine_sim:.4f}")
                    
                    if cosine_sim > threshold:
                        valid_results += 1
                        print(f"       ✅ PASSES threshold {threshold}")
                    else:
                        print(f"       ❌ Below threshold {threshold}")
                        
                except Exception as e:
                    print(f"       ⚠️ Error processing: {e}")
            
            print(f"   📊 Results passing threshold {threshold}: {valid_results}/{len(vector_results)}")
            
        except Exception as e:
            print(f"   ❌ Vector search failed: {e}")

if __name__ == "__main__":
    debug_vector_search()