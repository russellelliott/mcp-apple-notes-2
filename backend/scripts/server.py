import os
import sys  # Added sys
from pathlib import Path

# ── PATH FIX: must happen before any `backend.*` import ──────────────────────
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
# ─────────────────────────────────────────────────────────────────────────────

import json
import re

import lancedb
import pandas as pd
import numpy as np
import math
try:
    import umap
    UMAP_AVAILABLE = True
except Exception as _e:
    umap = None
    UMAP_AVAILABLE = False
from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from pathlib import Path
from sentence_transformers import SentenceTransformer
from contextlib import asynccontextmanager

# Import search logic
from backend.scripts.main import NotesDatabase
from backend.analysis.search_notes import search_and_combine_results

# Configuration
# NOTE: Matching streamlit_app.py configuration
DATA_DIR = Path.home() / ".mcp-apple-notes"
DB_PATH = DATA_DIR / "data"
TABLE_NAME = "notes"
MODEL_NAME = "BAAI/bge-small-en-v1.5"
# Server-side multiplier to push clusters further apart without changing shape/radius.
# Hard-coded here per request (change this constant to tune spacing).
CLUSTER_SPREAD_MULTIPLIER = 6.0

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
        if not UMAP_AVAILABLE:
            raise RuntimeError("umap library not available. Install umap-learn to compute projections.")
        embeddings = np.stack(df['vector'].values)
        
        # Remove `random_state` to allow UMAP to use multiple cores when available.
        # Set `n_jobs=-1` to enable parallelism across all CPUs.
        reducer = umap.UMAP(
            n_components=3,
            n_neighbors=30,
            min_dist=0.0,
            metric='cosine',
            n_jobs=-1,
        )
        
        projections = reducer.fit_transform(embeddings)
        
        df['umap_x'] = projections[:, 0]
        df['umap_y'] = projections[:, 1]
        df['umap_z'] = projections[:, 2]
        
        # Ensure ID/Key columns exist for frontend
        df['chunk_index'] = df['chunk_index'].fillna(0).astype(int)
        df['unique_key'] = df['title'].astype(str) + "_" + df['chunk_index'].astype(str)
        
        # Ensure total_chunks
        if 'total_chunks' not in df.columns:
             print("ℹ️ Computing total_chunks...")
             df['total_chunks'] = df.groupby('title')['chunk_index'].transform('count')
        else:
             df['total_chunks'] = df['total_chunks'].fillna(1).astype(int)
             
        # Fill optional fields
        if 'cluster_label' not in df.columns:
            df['cluster_label'] = 'Unclustered'
        else:
            df['cluster_label'] = df['cluster_label'].fillna('Unclustered')

        if 'cluster_id' not in df.columns:
            df['cluster_id'] = '-1'
        else:
            df['cluster_id'] = df['cluster_id'].astype(str).fillna('-1')

        if 'base_topic_id' not in df.columns:
            df['base_topic_id'] = df['cluster_id'].astype(str)
        else:
            df['base_topic_id'] = df['base_topic_id'].astype(str).fillna(df['cluster_id'])

        if 'display_topic_id' not in df.columns:
            df['display_topic_id'] = df['base_topic_id'].astype(str)
        else:
            df['display_topic_id'] = df['display_topic_id'].astype(str).fillna(df['base_topic_id'])
        
        # --- FIX 1: Ensure ID Consistency ---
        # Convert IDs to strings and normalize NaN representations
        for col in ['cluster_id', 'base_topic_id', 'display_topic_id']:
            if col in df.columns:
                df[col] = df[col].astype(str).replace(['nan', 'None', 'nan', '-1.0'], '-1')
        
        # Compute server-side shaped positions (Fibonacci sphere per cluster)
        try:
            print("🔮 Computing shaped cluster positions...")
            df = compute_shaped_positions(df)
            print(f"✅ Shaped positions computed. rows={len(df)} index_type={type(df.index)}")
            if len(df) > 0:
                print(f" first_index_sample={list(df.index[:5])}")
        except Exception as e:
            print(f"⚠️ Error computing shaped positions: {e}")

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
    total_chunks: Optional[int] = None
    cluster_id: Optional[str] = None
    base_topic_id: Optional[str] = None
    display_topic_id: Optional[str] = None
    cluster_label: str
    cluster_color: Optional[str] = None
    dot_color: Optional[str] = None
    umap_x: float
    umap_y: float
    umap_z: float
    creation_date: Optional[Any] = None
    modification_date: Optional[Any] = None
    # We avoid sending full content or vector to keep payload light, unless requested


class ShapedNotePoint(BaseModel):
    unique_key: str
    title: str
    chunk_index: int
    total_chunks: Optional[int] = None
    cluster_id: Optional[str] = None
    base_topic_id: Optional[str] = None
    display_topic_id: Optional[str] = None
    cluster_label: str
    cluster_color: Optional[str] = None
    dot_color: Optional[str] = None
    umap_x: float
    umap_y: float
    umap_z: float
    display_x: float
    display_y: float
    display_z: float
    creation_date: Optional[Any] = None
    modification_date: Optional[Any] = None


def compute_shaped_positions(df: pd.DataFrame) -> pd.DataFrame:
    """Compute display_x/y/z positions for each row by placing points on
    a Fibonacci sphere around a repulsed cluster centroid in UMAP space.
    """
    # Work on a copy and ensure positional integer index for numpy indexing
    df = df.copy().reset_index(drop=True)
    cluster_col = 'display_topic_id' if 'display_topic_id' in df.columns else 'cluster_id'

    if df.empty:
        df['display_x'] = pd.Series(dtype=float)
        df['display_y'] = pd.Series(dtype=float)
        df['display_z'] = pd.Series(dtype=float)
        return df

    # --- Stage 1: UMAP-space centroids per cluster ---
    # FIX 3: Only compute centroids for valid clusters (exclude -1 and Unclustered)
    valid_mask = (df[cluster_col] != '-1') & (df[cluster_col] != 'Unclustered')
    centroids = df[valid_mask].groupby(cluster_col)[['umap_x', 'umap_y', 'umap_z']].mean()
    if centroids.empty:
        # fallback: copy UMAP coords to display
        df['display_x'] = df['umap_x']
        df['display_y'] = df['umap_y']
        df['display_z'] = df['umap_z']
        return df

    # --- Stage 2: Repel cluster centroids so they don't overlap ---
    centroid_arr = centroids.values.copy()
    cluster_ids = centroids.index.tolist()
    K = len(cluster_ids)

    spread = np.ptp(centroid_arr, axis=0).max() if K > 0 else 0.0
    # Maximum separation: push clusters as far apart as possible
    MIN_SEP = max(spread * 5.0, 150.0)
    # Maximum iteration count for centroid repulsion convergence.
    # Lowered to speed startup; increase if clusters still overlap.
    REPULSE_ITERS = 400

    for _ in range(REPULSE_ITERS):
        for i in range(K):
            for j in range(i + 1, K):
                delta = centroid_arr[j] - centroid_arr[i]
                dist = np.linalg.norm(delta) or 1e-9
                if dist < MIN_SEP:
                    overlap = (MIN_SEP - dist) * 0.5
                    direction = delta / dist
                    centroid_arr[i] -= direction * overlap
                    centroid_arr[j] += direction * overlap

    # Maximum outward radial nudge from the global centroid to push clusters to the far edges
    try:
        global_cent = centroid_arr.mean(axis=0)
        # Allow server-wide tuning of how far clusters are pushed outward.
        # This only moves cluster centroids away from the global center; it
        # does not change per-cluster Fibonacci radii (so shapes/sizes stay).
        radial_base = 0.8
        radial_factor = radial_base * CLUSTER_SPREAD_MULTIPLIER
        for k in range(K):
            offset = centroid_arr[k] - global_cent
            centroid_arr[k] = centroid_arr[k] + offset * radial_factor
    except Exception:
        pass

    # Map back to dict for fast lookup (string keys)
    spaced_centroids = {str(cluster_ids[k]): centroid_arr[k] for k in range(K)}

    # --- Stage 3: Fibonacci sphere per cluster ---
    GOLDEN_ANGLE = math.pi * (3.0 - math.sqrt(5.0))

    # --- FIX 4: Explicit Color Assignment ---
    # Compute a stable server-side color per cluster from the centroid's
    # position in the centroid distribution. The frontend can then reuse
    # this exact color for the cluster and all of its chunks.
    cluster_color_map: Dict[str, str] = {}
    global_centroid = centroid_arr.mean(axis=0) if K > 0 else np.zeros(3)
    for k, cid in enumerate(cluster_ids):
        centroid = centroid_arr[k]
        dx = float(centroid[0] - global_centroid[0])
        dy = float(centroid[1] - global_centroid[1])
        if not np.isfinite(dx) or not np.isfinite(dy):
            cluster_color_map[str(cid)] = '#6b7280'
            continue
        if abs(dx) < 1e-12 and abs(dy) < 1e-12:
            cluster_color_map[str(cid)] = 'hsl(210, 75%, 45%)'
            continue
        angle = math.degrees(math.atan2(dy, dx)) % 360.0
        cluster_color_map[str(cid)] = f'hsl({int(round(angle))}, 75%, 45%)'
    
    # Initialize color columns with defaults
    df['cluster_color'] = '#6b7280'
    df['dot_color'] = '#6b7280'

    display_x = np.zeros(len(df))
    display_y = np.zeros(len(df))
    display_z = np.zeros(len(df))

    # Backup original UMAP coords so we don't lose them.
    df['umap_x_orig'] = df['umap_x']
    df['umap_y_orig'] = df['umap_y']
    df['umap_z_orig'] = df['umap_z']

    # --- FIX 5: Full Group Inclusion ---
    # Place each cluster in one pass using every row in the cached DataFrame.
    # Explicitly assign color to EVERY row in the group.
    for cid, group_idx in df.groupby(cluster_col, sort=False).groups.items():
        indices = list(group_idx)
        n = len(indices)
        cid_str = str(cid)
        
        # Assign color to EVERY row in this group
        color = cluster_color_map.get(cid_str, '#6b7280')
        df.loc[indices, 'cluster_color'] = color
        df.loc[indices, 'dot_color'] = color
        
        # Skip Fibonacci placement for unclustered points
        if cid_str == '-1' or cid_str == 'Unclustered':
            display_x[indices] = df.loc[indices, 'umap_x'].values
            display_y[indices] = df.loc[indices, 'umap_y'].values
            display_z[indices] = df.loc[indices, 'umap_z'].values
            continue
        
        centroid = spaced_centroids.get(cid_str)
        if centroid is None:
            centroid = centroids.loc[cid_str].values if cid_str in centroids.index else np.zeros(3)
        
        # Maximum intra-cluster radius for extreme dot spread within each cluster.
        # Make cluster size scale with number of chunks: small clusters stay small,
        # larger clusters grow more strongly. Use sqrt and log to provide smooth scaling.
        RADIUS = max(2.5, math.sqrt(n) * 2.5, math.log1p(max(1, n)) * 2.8)
        
        # Sort indices to ensure stable positions
        indices_sorted = sorted(indices, key=lambda i: (df.at[i, 'chunk_index'], i))
        
        for rank, df_idx in enumerate(indices_sorted):
            y_fib = 1.0 - (rank / max(n - 1, 1)) * 2.0
            r_fib = math.sqrt(max(0.0, 1.0 - y_fib * y_fib))
            theta = GOLDEN_ANGLE * rank
            
            display_x[df_idx] = centroid[0] + r_fib * math.cos(theta) * RADIUS
            display_y[df_idx] = centroid[1] + y_fib * RADIUS
            display_z[df_idx] = centroid[2] + r_fib * math.sin(theta) * RADIUS
        
        # Overwrite per-point UMAP positions with the cluster centroid
        # so that independent UMAP positions are disregarded and all
        # points are assembled at/around the cluster center.
        try:
            idx_arr = np.array(indices_sorted, dtype=int)
            df.loc[idx_arr, 'umap_x'] = float(centroid[0])
            df.loc[idx_arr, 'umap_y'] = float(centroid[1])
            df.loc[idx_arr, 'umap_z'] = float(centroid[2])
        except Exception:
            # If anything goes wrong here, continue — we still have display positions.
            pass

    df['display_x'] = display_x
    df['display_y'] = display_y
    df['display_z'] = display_z
    
    # Ensure color columns have values (should be filled by now, but add safety fallback)
    if 'cluster_color' not in df.columns or df['cluster_color'].isna().any():
        df['cluster_color'] = df['cluster_color'].fillna('#6b7280')
    if 'dot_color' not in df.columns or df['dot_color'].isna().any():
        df['dot_color'] = df['dot_color'].fillna('#6b7280')
    
    return df

class SearchResult(BaseModel):
    unique_key: str
    title: str
    chunk_index: int
    total_chunks: Optional[int] = None
    distance: float
    cluster_id: Optional[str] = None
    base_topic_id: Optional[str] = None
    display_topic_id: Optional[str] = None
    cluster_label: str
    preview: Optional[str] = None
    
class SearchStats(BaseModel):
    total_chunks: int
    unique_notes: int
    
class SearchResponse(BaseModel):
    results: List[SearchResult]
    match_ids: List[str] # List of unique_keys that matched
    stats: SearchStats

class NoteContent(BaseModel):
    title: str
    chunk_index: int
    content: str
    total_chunks: int
    cluster_id: Optional[str] = None
    cluster_label: Optional[str] = None
    base_topic_id: Optional[str] = None
    display_topic_id: Optional[str] = None


class SidebarChunk(BaseModel):
    chunk_index: int
    cluster_id: str
    cluster_name: str
    in_cluster: bool
    text: Optional[str] = None


class SidebarNote(BaseModel):
    note_key: str
    title: str
    creation_date: Optional[str] = None
    modification_date: Optional[str] = None
    chunks: List[SidebarChunk]


class SidebarResponse(BaseModel):
    active_cluster_id: str
    notes: List[SidebarNote]

# --- Endpoints ---

@app.get("/note_content", response_model=NoteContent)
async def get_note_content(
    title: str,
    chunk_index: int,
    creation_date: Optional[str] = None,
    modification_date: Optional[str] = None,
):
    """Get full content for a specific chunk"""
    if state.df_viz.empty:
        raise HTTPException(status_code=503, detail="Data not loaded")
    
    try:
        # Filter for the specific chunk
        # Using string comparison for title to be safe, equality for chunk_index
        mask = (state.df_viz['title'] == title) & (state.df_viz['chunk_index'] == chunk_index)
        if creation_date is not None and 'creation_date' in state.df_viz.columns:
            mask = mask & (state.df_viz['creation_date'].astype(str) == str(creation_date))
        if modification_date is not None and 'modification_date' in state.df_viz.columns:
            mask = mask & (state.df_viz['modification_date'].astype(str) == str(modification_date))
        row = state.df_viz[mask]
        
        if row.empty:
            raise HTTPException(status_code=404, detail="Chunk not found")
            
        # Get content
        content = row.iloc[0].get('chunk_content', '')
        if pd.isna(content) or content == '':
             content = row.iloc[0].get('text', '')
             if pd.isna(content): content = ""
        
        total = row.iloc[0].get('total_chunks', 1)
        
        return NoteContent(
            title=title,
            chunk_index=chunk_index,
            content=str(content),
            total_chunks=int(total),
            cluster_id=(str(row.iloc[0].get('cluster_id')) if row.iloc[0].get('cluster_id') is not None else None),
            cluster_label=(str(row.iloc[0].get('cluster_label')) if row.iloc[0].get('cluster_label') is not None else None),
            base_topic_id=(str(row.iloc[0].get('base_topic_id')) if row.iloc[0].get('base_topic_id') is not None else None),
            display_topic_id=(str(row.iloc[0].get('display_topic_id')) if row.iloc[0].get('display_topic_id') is not None else None),
        )
    except Exception as e:
        print(f"Error fetching content: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/cluster_sidebar", response_model=SidebarResponse)
async def get_cluster_sidebar(active_cluster_id: str):
    """Return note cards for a cluster with full chunk rails and selective text payload."""
    if state.df_viz.empty:
        return SidebarResponse(active_cluster_id=active_cluster_id, notes=[])

    try:
        working_df = state.df_viz.copy()
        cluster_col = 'display_topic_id' if 'display_topic_id' in working_df.columns else 'cluster_id'
        if cluster_col not in working_df.columns:
            raise HTTPException(status_code=500, detail="Cluster column missing")

        working_df[cluster_col] = working_df[cluster_col].astype(str).fillna('-1')
        working_df['title'] = working_df['title'].astype(str)
        working_df['chunk_index'] = working_df['chunk_index'].fillna(0).astype(int)
        if 'creation_date' not in working_df.columns:
            working_df['creation_date'] = ''
        if 'modification_date' not in working_df.columns:
            working_df['modification_date'] = ''
        if 'cluster_label' not in working_df.columns:
            working_df['cluster_label'] = 'Unclustered'

        working_df['creation_date'] = working_df['creation_date'].astype(str).fillna('')
        working_df['modification_date'] = working_df['modification_date'].astype(str).fillna('')
        working_df['cluster_label'] = working_df['cluster_label'].astype(str).fillna('Unclustered')

        active_rows = working_df[working_df[cluster_col] == str(active_cluster_id)].copy()
        if active_rows.empty:
            return SidebarResponse(active_cluster_id=active_cluster_id, notes=[])

        note_identity_cols = ['title', 'creation_date', 'modification_date']
        note_keys_df = active_rows[note_identity_cols].drop_duplicates().copy()
        note_keys_df['note_key'] = (
            note_keys_df['title']
            + '|||' + note_keys_df['creation_date']
            + '|||' + note_keys_df['modification_date']
        )

        merged = working_df.merge(
            note_keys_df,
            on=note_identity_cols,
            how='inner',
        )

        merged = merged.sort_values(['note_key', 'chunk_index'])

        notes: List[SidebarNote] = []
        for note_key, group in merged.groupby('note_key', sort=False):
            title = str(group.iloc[0]['title'])
            creation_date = str(group.iloc[0].get('creation_date', ''))
            modification_date = str(group.iloc[0].get('modification_date', ''))

            chunks: List[SidebarChunk] = []
            seen_chunk_indexes = set()
            for _, row in group.iterrows():
                chunk_index_val = int(row.get('chunk_index', 0))
                if chunk_index_val in seen_chunk_indexes:
                    continue
                seen_chunk_indexes.add(chunk_index_val)

                chunk_cluster_id = str(row.get(cluster_col, '-1'))
                in_cluster = chunk_cluster_id == str(active_cluster_id)

                chunk_text = None
                if in_cluster:
                    chunk_text = row.get('chunk_content', '')
                    if pd.isna(chunk_text) or chunk_text == '':
                        chunk_text = row.get('text', '')
                    if pd.isna(chunk_text):
                        chunk_text = ''
                    chunk_text = str(chunk_text)

                chunks.append(
                    SidebarChunk(
                        chunk_index=chunk_index_val,
                        cluster_id=chunk_cluster_id,
                        cluster_name=str(row.get('cluster_label', 'Unclustered')),
                        in_cluster=in_cluster,
                        text=chunk_text,
                    )
                )

            notes.append(
                SidebarNote(
                    note_key=str(note_key),
                    title=title,
                    creation_date=creation_date,
                    modification_date=modification_date,
                    chunks=chunks,
                )
            )

        return SidebarResponse(active_cluster_id=active_cluster_id, notes=notes)
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error building cluster sidebar: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/points", response_model=List[NotePoint])
async def get_points():
    """Get all points for visualization"""
    if state.df_viz.empty:
        return []
    
    # Convert dataframe to list of dicts efficiently
    # We replicate the cleaning logic from the model definition
    # Replace NaNs to avoid JSON errors
    valid_df = state.df_viz.where(pd.notnull(state.df_viz), None)

    points_df = valid_df.copy()
    points_df['cluster_id'] = points_df['display_topic_id']

    records = points_df[[
        'unique_key', 'title', 'chunk_index', 'total_chunks', 'cluster_id', 'base_topic_id',
        'display_topic_id', 'cluster_label', 'cluster_color', 'dot_color', 'umap_x', 'umap_y', 'umap_z', 'creation_date',
        'modification_date'
    ]].to_dict(orient='records')
    
    return records


@app.get("/points_shaped", response_model=List[ShapedNotePoint])
async def get_points_shaped():
    """
    Get all points with server-computed Fibonacci sphere positions.
    display_x/y/z replace raw umap_x/y/z for visualization.
    Raw UMAP coordinates are still included for reference.
    """
    if state.df_viz.empty:
        return []

    required_cols = ['display_x', 'display_y', 'display_z']
    if not all(c in state.df_viz.columns for c in required_cols):
        raise HTTPException(
            status_code=503,
            detail="Shaped positions not yet computed. Data may still be loading."
        )

    valid_df = state.df_viz.where(pd.notnull(state.df_viz), None)
    points_df = valid_df.copy()
    points_df['cluster_id'] = points_df['display_topic_id']

    records = points_df[[
        'unique_key', 'title', 'chunk_index', 'total_chunks', 'cluster_id', 'base_topic_id',
        'display_topic_id', 'cluster_label',
        'cluster_color', 'dot_color',
        'umap_x', 'umap_y', 'umap_z',
        'display_x', 'display_y', 'display_z',
        'creation_date', 'modification_date'
    ]].to_dict(orient='records')

    print(f"/points_shaped returning {len(records)} records")
    if len(records) > 0:
        print(f" sample record keys: {list(records[0].keys())}")
    return records

@app.get("/search", response_model=SearchResponse)
async def search(q: str = Query(..., min_length=1), limit: int = 1000, max_distance: float = 0.8):
    """Search notes and return matches + IDs"""
    if state.table is None:
        raise HTTPException(status_code=503, detail="Database not initialized")

    print(f"🔍 Searching for: {q} (threshold: {max_distance})")
    
    # helper for search_and_combine_results
    embedding_fn = lambda query: get_query_embedding(query)
    
    try:
        results = search_and_combine_results(
            state.table, 
            q, 
            display_limit=limit, 
            max_distance=max_distance,
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
    cluster_id_map = {}
    base_topic_id_map = {}
    display_topic_id_map = {}
    if not state.df_viz.empty:
        # unique_key -> cluster_label
        cluster_map = state.df_viz.set_index('unique_key')['cluster_label'].to_dict()
        cluster_id_map = state.df_viz.set_index('unique_key')['cluster_id'].to_dict()
        if 'base_topic_id' in state.df_viz.columns:
            base_topic_id_map = state.df_viz.set_index('unique_key')['base_topic_id'].to_dict()
        if 'display_topic_id' in state.df_viz.columns:
            display_topic_id_map = state.df_viz.set_index('unique_key')['display_topic_id'].to_dict()

    for r in results:
        title = r.get('title', '')
        idx = r.get('_chunk_index', r.get('chunk_index', 0))
        total = r.get('_total_chunks', r.get('total_chunks'))
        score = r.get('_relevance_score', 0)
        preview = r.get('_matching_chunk_preview', '') or r.get('chunk_content', '')[:200]
        
        unique_key = f"{title}_{idx}"
        
        # Lookup cluster
        cluster = cluster_map.get(unique_key, "Unknown")
        base_topic_id = str(base_topic_id_map.get(unique_key, cluster_id_map.get(unique_key, "-1")))
        display_topic_id = str(display_topic_id_map.get(unique_key, base_topic_id))
        
        res_obj = SearchResult(
            unique_key=unique_key,
            title=title,
            chunk_index=idx,
            total_chunks=total,
            distance=score,
            cluster_id=display_topic_id,
            base_topic_id=base_topic_id,
            display_topic_id=display_topic_id,
            cluster_label=cluster,
            preview=preview
        )
        formatted_results.append(res_obj)
        match_ids.append(unique_key)
        
    unique_titles_found = set(r.title for r in formatted_results)
    
    # Print top results to console for debugging
    print(f"✅ Found {len(formatted_results)} matching chunks across {len(unique_titles_found)} notes.")
    for i, res in enumerate(formatted_results):
        cid = res.cluster_id if res.cluster_id and res.cluster_id != '-1' else res.cluster_label
        print(f"   {i+1}. {res.title} (Chunk {res.chunk_index + 1} of {res.total_chunks or '?'}) [Score: {res.distance:.3f}, Cluster: {cid}]")

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


@app.get("/debug_state")
async def debug_state():
    """Return internal debug info: loaded rows, columns, and a small sample."""
    try:
        loaded = 0 if state.df_viz is None or state.df_viz.empty else len(state.df_viz)
        cols = [] if state.df_viz is None or state.df_viz.empty else list(state.df_viz.columns)
        sample = [] if state.df_viz is None or state.df_viz.empty else state.df_viz.head(5).to_dict(orient='records')
        return {
            'loaded_rows': loaded,
            'columns': cols,
            'sample_count': len(sample),
            'sample': sample,
            'table_present': state.table is not None,
            'db_path': str(DB_PATH),
            'table_name': TABLE_NAME,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/reload_data")
async def reload_data():
    """Force reload data from LanceDB and recompute UMAP/projections.
    Useful after running clustering scripts so the API reflects DB updates.
    """
    try:
        load_and_process_data()
        return {"reloaded": True, "loaded_rows": len(state.df_viz)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# Interaction Tracking Endpoints
# ============================================================================
class InteractionRequest(BaseModel):
    title: str
    event_type: str

class InteractionResponse(BaseModel):
    success: bool
    message: str
    last_opened: Optional[str] = None

class InteractionLogResponse(BaseModel):
    title: str
    last_opened: Optional[str]
    interaction_log: List[Dict[str, str]]

@app.post("/interaction/log")
async def log_interaction(request: InteractionRequest):
    """Log an interaction event for a note (opened/modified)"""
    try:
        if request.event_type not in ["opened", "modified"]:
            raise HTTPException(status_code=400, detail="event_type must be 'opened' or 'modified'")
        
        db = NotesDatabase(db_path=DB_PATH)
        notes_table = db.get_or_create_table()
        
        chunks = notes_table.to_pandas()
        note_titles = chunks[chunks['title'] == request.title]['title'].tolist()
        
        if not note_titles:
            return InteractionResponse(
                success=False,
                message=f"Note '{request.title}' not found in database"
            )
        
        if request.event_type == "opened":
            db.log_note_opened(request.title)
        else:
            db.log_note_modified(request.title)
        
        last_opened = db.get_last_opened(request.title)
        
        return InteractionResponse(
            success=True,
            message=f"Successfully logged {request.event_type} event for '{request.title}'",
            last_opened=last_opened
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to log interaction: {str(e)}")

@app.get("/interaction/{title}")
async def get_interaction(title: str):
    """Get interaction data for a specific note"""
    try:
        db = NotesDatabase(db_path=DB_PATH)
        
        last_opened = db.get_last_opened(title)
        interaction_log = db.get_interaction_log(title)
        
        if interaction_log is None:
            interaction_log = []
        
        return InteractionLogResponse(
            title=title,
            last_opened=last_opened,
            interaction_log=interaction_log
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get interaction data: {str(e)}")

@app.get("/interactions/list")
async def list_interactions(limit: Optional[int] = None):
    """Get a list of all notes with their last_opened timestamps. Default is no limit."""
    try:
        db = NotesDatabase(db_path=DB_PATH)
        # Passing None to the DB method typically fetches all records
        interactions = db.list_all_interactions(limit=limit)
        
        return {
            "interactions": interactions,
            "count": len(interactions)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list interactions: {str(e)}")

# ============================================================================
# Daily History Endpoints
# ============================================================================

class DayResponse(BaseModel):
    date: str
    titles: List[str]  # Distinct note titles opened on this date
    notes: List[Dict[str, Any]]  # Full note metadata for those titles


def _parse_interaction_events(raw_log: Any) -> List[Dict[str, Any]]:
    if raw_log is None:
        return []

    try:
        events = json.loads(raw_log) if isinstance(raw_log, str) else raw_log
    except Exception:
        return []

    if isinstance(events, dict):
        events = [events]

    return events if isinstance(events, list) else []


def _event_date(dt_str: Any) -> Optional[str]:
    if not dt_str:
        return None

    dt_value = str(dt_str)
    if len(dt_value) < 10:
        return None

    date_only = dt_value[:10]
    if re.match(r'^\d{4}-\d{2}-\d{2}$', date_only):
        return date_only
    return None


def _collect_history_by_title(date_str: str) -> Dict[str, Dict[str, Any]]:
    history_by_title: Dict[str, Dict[str, Any]] = {}

    db = NotesDatabase(db_path=DB_PATH)
    _, interactions_table = db.get_interactions_db()
    if interactions_table is None:
        return history_by_title

    interactions_df = interactions_table.to_pandas()
    if interactions_df.empty:
        return history_by_title

    if "interaction_log" not in interactions_df.columns:
        return history_by_title

    for _, row in interactions_df.iterrows():
        title = str(row.get("title", "")).strip()
        if not title:
            continue

        events = _parse_interaction_events(row.get("interaction_log", "[]"))
        if not events:
            continue

        matching_timestamps = [str(ev.get("dt", "")) for ev in events if _event_date(ev.get("dt")) == date_str]
        if not matching_timestamps:
            continue

        matching_timestamps.sort()
        title_info = history_by_title.setdefault(
            title,
            {
                "title": title,
                "opened_at": matching_timestamps[0],
                "last_opened_at": matching_timestamps[-1],
                "opened_count": 0,
            },
        )
        title_info["opened_count"] += len(matching_timestamps)
        title_info["opened_at"] = min(title_info["opened_at"], matching_timestamps[0])
        title_info["last_opened_at"] = max(title_info["last_opened_at"], matching_timestamps[-1])

    return history_by_title


@app.get("/history/dates")
async def get_history_dates():
    """Return a sorted list of all dates (YYYY-MM-DD) with interaction data."""
    try:
        db = NotesDatabase(db_path=DB_PATH)
        _, interactions_table = db.get_interactions_db()
        if interactions_table is None:
            return {"dates": []}

        interactions_df = interactions_table.to_pandas()
        if interactions_df.empty or "interaction_log" not in interactions_df.columns:
            return {"dates": []}

        dates_set = set()
        for log_json in interactions_df["interaction_log"].dropna():
            for event in _parse_interaction_events(log_json):
                date_only = _event_date(event.get("dt", ""))
                if date_only:
                    dates_set.add(date_only)

        return {"dates": sorted(list(dates_set))}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch dates: {str(e)}")


@app.get("/history/day/{date_str}")
async def get_history_for_day(date_str: str):
    """Get all distinct note titles and their full metadata opened on a given date.

    date_str format: YYYY-MM-DD
    Returns:
      - titles: list of distinct note titles open that day
      - notes: full note metadata from the main notes table for those titles
    """
    try:
        # Validate date format
        if not re.match(r'^\d{4}-\d{2}-\d{2}$', date_str):
            raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD")

        history_by_title = _collect_history_by_title(date_str)
        active_titles = sorted(history_by_title.keys())

        if not active_titles:
            return {"date": date_str, "titles": [], "notes": []}

        print(f"📅 Found {len(active_titles)} titles for {date_str}")

        if state.df_viz.empty:
            load_and_process_data()

        if not state.df_viz.empty:
            working_df = state.df_viz.copy()
            working_df['title'] = working_df['title'].astype(str)
            if 'creation_date' not in working_df.columns:
                working_df['creation_date'] = ''
            if 'modification_date' not in working_df.columns:
                working_df['modification_date'] = ''
            if 'cluster_label' not in working_df.columns:
                working_df['cluster_label'] = 'Unclustered'
            if 'cluster_id' not in working_df.columns:
                working_df['cluster_id'] = '-1'
            if 'display_topic_id' not in working_df.columns:
                working_df['display_topic_id'] = working_df['cluster_id']

            working_df['creation_date'] = working_df['creation_date'].astype(str).fillna('')
            working_df['modification_date'] = working_df['modification_date'].astype(str).fillna('')
            working_df['cluster_label'] = working_df['cluster_label'].astype(str).fillna('Unclustered')
            working_df['cluster_id'] = working_df['cluster_id'].astype(str).fillna('-1')
            working_df['display_topic_id'] = working_df['display_topic_id'].astype(str).fillna('-1')

            notes_df = working_df[working_df['title'].isin(active_titles)].copy()
            note_identity_cols = ['title', 'creation_date', 'modification_date']
            note_keys_df = notes_df[note_identity_cols].drop_duplicates().copy()
            note_keys_df['note_key'] = (
                note_keys_df['title']
                + '|||' + note_keys_df['creation_date']
                + '|||' + note_keys_df['modification_date']
            )

            merged = working_df.merge(
                note_keys_df,
                on=note_identity_cols,
                how='inner',
            )

            merged = merged.sort_values(['note_key', 'chunk_index'])
            notes_list = []
            for note_key, group in merged.groupby('note_key', sort=False):
                title = str(group.iloc[0].get('title', ''))
                history_info = history_by_title.get(title, {})
                creation_date = str(group.iloc[0].get('creation_date', ''))
                modification_date = str(group.iloc[0].get('modification_date', ''))

                cluster_counts: Dict[str, int] = {}
                cluster_first_seen: Dict[str, int] = {}
                for position, (_, row) in enumerate(group.iterrows()):
                    cluster_key = str(row.get('display_topic_id', row.get('cluster_id', '-1')))
                    if not cluster_key or cluster_key == 'nan':
                        cluster_key = '-1'
                    cluster_counts[cluster_key] = cluster_counts.get(cluster_key, 0) + 1
                    if cluster_key not in cluster_first_seen:
                        cluster_first_seen[cluster_key] = position

                primary_cluster_id = '-1'
                if cluster_counts:
                    primary_cluster_id = sorted(
                        cluster_counts.items(),
                        key=lambda item: (-item[1], cluster_first_seen.get(item[0], 0)),
                    )[0][0]

                chunks = []
                seen_chunk_indexes = set()
                for _, row in group.iterrows():
                    chunk_index_val = int(row.get('chunk_index', 0))
                    if chunk_index_val in seen_chunk_indexes:
                        continue
                    seen_chunk_indexes.add(chunk_index_val)

                    chunk_text = row.get('chunk_content', '')
                    if pd.isna(chunk_text) or chunk_text == '':
                        chunk_text = row.get('text', '')
                    if pd.isna(chunk_text):
                        chunk_text = ''

                    chunk_cluster_id = str(row.get('display_topic_id', row.get('cluster_id', '-1')))
                    if not chunk_cluster_id or chunk_cluster_id == 'nan':
                        chunk_cluster_id = '-1'

                    chunks.append({
                        "chunk_index": chunk_index_val,
                        "cluster_id": chunk_cluster_id,
                        "cluster_name": str(row.get('cluster_label', 'Unclustered')),
                        "in_cluster": chunk_cluster_id == primary_cluster_id,
                        "text": str(chunk_text),
                    })

                notes_list.append({
                    "note_key": str(note_key),
                    "title": title,
                    "creation_date": creation_date,
                    "modification_date": modification_date,
                    "primary_cluster_id": primary_cluster_id,
                    "opened_at": history_info.get("opened_at"),
                    "last_opened_at": history_info.get("last_opened_at"),
                    "opened_count": history_info.get("opened_count", 0),
                    "chunks": chunks,
                })

            notes_list.sort(key=lambda item: item.get("opened_at") or "", reverse=True)
            return {"date": date_str, "titles": active_titles, "notes": notes_list}

        return {"date": date_str, "titles": active_titles, "notes": []}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch day history: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
