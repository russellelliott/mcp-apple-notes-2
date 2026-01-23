import os
import lancedb
import pandas as pd
import numpy as np
import umap
from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from pathlib import Path
from sentence_transformers import SentenceTransformer
from contextlib import asynccontextmanager

# Import search logic
from scripts.search_notes import search_and_combine_results

# Configuration
# NOTE: Matching streamlit_app.py configuration
DATA_DIR = Path.home() / ".mcp-apple-notes"
DB_PATH = DATA_DIR / "data"
TABLE_NAME = "notes"
MODEL_NAME = "BAAI/bge-small-en-v1.5"

# Global State Container
class AppState:
    table = None
    model = None
    df_viz = pd.DataFrame() # Cached DataFrame with UMAP coordinates

state = AppState()

def get_embedding_model():
    if state.model is None:
        print(f"📦 Loading embedding model: {MODEL_NAME}")
        state.model = SentenceTransformer(MODEL_NAME)
    return state.model

def get_query_embedding(query: str) -> List[float]:
    model = get_embedding_model()
    return model.encode(query).tolist()

def load_and_process_data():
    """Lengths data from LanceDB and computes UMAP projections"""
    print("🔄 Loading data from LanceDB...")
    try:
        db = lancedb.connect(str(DB_PATH))
        state.table = db.open_table(TABLE_NAME)
        df = state.table.to_pandas()
        
        if df.empty:
            print("⚠️ Database is empty.")
            state.df_viz = pd.DataFrame()
            return

        print(f"✅ Loaded {len(df)} rows.")

        if 'vector' not in df.columns:
            print("❌ 'vector' column missing.")
            state.df_viz = df
            return
            
        # Compute UMAP
        print("🧮 Computing UMAP projections...")
        embeddings = np.stack(df['vector'].values)
        
        reducer = umap.UMAP(
            n_components=2, 
            random_state=42,
            n_neighbors=30,
            min_dist=0.0,
            metric='cosine'
        )
        
        projections = reducer.fit_transform(embeddings)
        
        df['umap_x'] = projections[:, 0]
        df['umap_y'] = projections[:, 1]
        
        # Ensure ID/Key columns exist for frontend
        df['chunk_index'] = df['chunk_index'].fillna(0).astype(int)
        df['unique_key'] = df['title'].astype(str) + "_" + df['chunk_index'].astype(str)
        
        # Fill optional fields
        if 'cluster_label' not in df.columns:
            df['cluster_label'] = 'Unclustered'
        else:
            df['cluster_label'] = df['cluster_label'].fillna('Unclustered')
            
        state.df_viz = df
        print("✨ Data processing complete.")

    except Exception as e:
        print(f"❌ Error loading data: {e}")
        state.df_viz = pd.DataFrame()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    get_embedding_model() # Preload model
    load_and_process_data() # Preload data & UMAP
    yield
    # Shutdown (if any cleanup needed)

app = FastAPI(lifespan=lifespan)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Allow all for now
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Models ---

class NotePoint(BaseModel):
    unique_key: str
    title: str
    chunk_index: int
    cluster_label: str
    umap_x: float
    umap_y: float
    creation_date: Optional[Any] = None
    modification_date: Optional[Any] = None
    # We avoid sending full content or vector to keep payload light, unless requested

class SearchResult(BaseModel):
    unique_key: str
    title: str
    chunk_index: int
    score: float
    cluster_label: str
    preview: Optional[str] = None
    
class SearchStats(BaseModel):
    total_chunks: int
    unique_notes: int
    
class SearchResponse(BaseModel):
    results: List[SearchResult]
    match_ids: List[str] # List of unique_keys that matched
    stats: SearchStats

# --- Endpoints ---

@app.get("/points", response_model=List[NotePoint])
async def get_points():
    """Get all points for visualization"""
    if state.df_viz.empty:
        return []
    
    # Convert dataframe to list of dicts efficiently
    # We replicate the cleaning logic from the model definition
    # Replace NaNs to avoid JSON errors
    valid_df = state.df_viz.where(pd.notnull(state.df_viz), None)
    
    records = valid_df[[
        'unique_key', 'title', 'chunk_index', 'cluster_label', 
        'umap_x', 'umap_y', 'creation_date', 'modification_date'
    ]].to_dict(orient='records')
    
    return records

@app.get("/search", response_model=SearchResponse)
async def search(q: str = Query(..., min_length=1), limit: int = 5):
    """Search notes and return matches + IDs"""
    if state.table is None:
        raise HTTPException(status_code=503, detail="Database not initialized")

    print(f"🔍 Searching for: {q}")
    
    # helper for search_and_combine_results
    embedding_fn = lambda query: get_query_embedding(query)
    
    try:
        results = search_and_combine_results(
            state.table, 
            q, 
            display_limit=limit, 
            compute_query_embedding=embedding_fn
        )
    except Exception as e:
         print(f"Search error: {e}")
         raise HTTPException(status_code=500, detail=str(e))

    formatted_results = []
    match_ids = []
    
    # We need to look up cluster labels from our cached dataframe since search results might comes raw from DB
    # We can create a lookup map
    cluster_map = {}
    if not state.df_viz.empty:
        # unique_key -> cluster_label
        cluster_map = state.df_viz.set_index('unique_key')['cluster_label'].to_dict()

    for r in results:
        title = r.get('title', '')
        idx = r.get('_chunk_index', r.get('chunk_index', 0))
        score = r.get('_relevance_score', 0)
        preview = r.get('_matching_chunk_preview', '') or r.get('chunk_content', '')[:200]
        
        unique_key = f"{title}_{idx}"
        
        # Lookup cluster
        cluster = cluster_map.get(unique_key, "Unknown")
        
        res_obj = SearchResult(
            unique_key=unique_key,
            title=title,
            chunk_index=idx,
            score=score,
            cluster_label=cluster,
            preview=preview
        )
        formatted_results.append(res_obj)
        match_ids.append(unique_key)
        
    unique_titles_found = set(r.title for r in formatted_results)
    
    return SearchResponse(
        results=formatted_results,
        match_ids=match_ids,
        stats=SearchStats(
            total_chunks=len(formatted_results),
            unique_notes=len(unique_titles_found)
        )
    )

@app.get("/health")
async def health():
    return {"status": "ok", "loaded_rows": len(state.df_viz)}

if __name__ == "__main__":
    import uvicorn
    # Use standard port 8000
    uvicorn.run(app, host="127.0.0.1", port=8000)
