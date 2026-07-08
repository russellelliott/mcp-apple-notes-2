import json
import math
import re
import sys
from pathlib import Path
from contextlib import asynccontextmanager

import lancedb
import numpy as np
import pandas as pd
from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from typing import Any, Dict, List, Optional


# ── REQUIRED PATH FIX: must happen before any `backend.*` import ──────────────────────
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    import umap
    UMAP_AVAILABLE = True
except Exception:
    umap = None
    UMAP_AVAILABLE = False

# Import search logic
from backend.scripts.main import NotesDatabase
from backend.analysis.search_notes import search_and_combine_results
from sklearn.metrics.pairwise import cosine_similarity

# ── PATH FIX ───────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
# ────────────────────────────────────────────────────────────────────────────

# Configuration
DATA_DIR = Path.home() / ".mcp-apple-notes"
DB_PATH = DATA_DIR / "data"
TABLE_NAME = "notes"
MODEL_NAME = "BAAI/bge-small-en-v1.5"

EMBEDDING_DIM = 15
SIMILARITY_THRESHOLD = 0.3
MAX_SIMILAR_RESULTS = 25


# ── Global State ────────────────────────────────────────────────────────────
class AppState:
    table = None
    model = None
    df_viz = pd.DataFrame()

state = AppState()


# ── Helpers ─────────────────────────────────────────────────────────────────
def get_embedding_model():
    if state.model is None:
        print(f"Loading embedding model: {MODEL_NAME}")
        state.model = SentenceTransformer(MODEL_NAME)
    return state.model


def get_query_embedding(query: str) -> List[float]:
    model = get_embedding_model()
    return model.encode(query).tolist()


def _compute_meta_centroids(df, topic_col="display_topic_id"):
    """Compute centroid (average embedding) for each cluster."""
    if "vector" not in df.columns or df.empty:
        return pd.DataFrame(columns=["cluster_id", "centroid", "meta_cluster_id", "chunk_count"])

    rows = []
    for cid, group in df.groupby(topic_col):
        vectors = np.stack(group["vector"].values)
        centroid = vectors.mean(axis=0)
        meta_id = str(group.iloc[0].get("meta_cluster_id", ""))
        rows.append({
            "cluster_id": str(cid),
            "centroid": centroid,
            "meta_cluster_id": meta_id,
            "chunk_count": len(group),
        })
    return pd.DataFrame(rows)


def load_and_process_data():
    """Loads data from LanceDB. 3D UMAP projections are no longer needed."""
    print("Loading data from LanceDB...")
    try:
        db = lancedb.connect(str(DB_PATH))
        state.table = db.open_table(TABLE_NAME)
        df = state.table.to_pandas()

        if df.empty:
            print("Database is empty.")
            state.df_viz = pd.DataFrame()
            return

        print(f"Loaded {len(df)} rows.")

        if 'vector' not in df.columns:
            print("'vector' column missing.")

        df['chunk_index'] = df['chunk_index'].fillna(0).astype(int)
        df['unique_key'] = df['title'].astype(str) + "_" + df['chunk_index'].astype(str)

        if 'total_chunks' not in df.columns:
            print("Computing total_chunks...")
            df['total_chunks'] = df.groupby('title')['chunk_index'].transform('count')
        else:
            df['total_chunks'] = df['total_chunks'].fillna(1).astype(int)

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

        for col in ['cluster_id', 'base_topic_id', 'display_topic_id']:
            if col in df.columns:
                df[col] = df[col].astype(str).replace(['nan', 'None', '-1.0'], '-1')

        # Compute 5-dim UMAP
        print("Computing 5-dim UMAP embeddings...")
        if not UMAP_AVAILABLE:
            print("umap library not available — skipping UMAP projections.")
            df['umap_x'] = 0.0
            df['umap_y'] = 0.0
            df['umap_z'] = 0.0
        else:
            if 'vector' in df.columns:
                embeddings = np.stack(df['vector'].values)
                reducer = umap.UMAP(
                    n_components=5, n_neighbors=30, min_dist=0.0,
                    metric='cosine', n_jobs=-1,
                )
                projections = reducer.fit_transform(embeddings)
                df['umap_x'] = projections[:, 0]
                df['umap_y'] = projections[:, 1]
                df['umap_z'] = projections[:, 2]
            else:
                df['umap_x'] = 0.0
                df['umap_y'] = 0.0
                df['umap_z'] = 0.0

        state.df_viz = df
        print("Data processing complete.")

    except Exception as e:
        print(f"Error loading data: {e}")
        state.df_viz = pd.DataFrame()


# ── Lifespan ────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    get_embedding_model()
    load_and_process_data()
    yield

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Pydantic Models ────────────────────────────────────────────────────────
class NotePoint(BaseModel):
    unique_key: str
    title: str
    chunk_index: int
    total_chunks: Optional[int] = None
    cluster_id: Optional[str] = None
    base_topic_id: Optional[str] = None
    display_topic_id: Optional[str] = None
    cluster_label: str
    umap_x: float
    umap_y: float
    umap_z: float
    creation_date: Optional[Any] = None
    modification_date: Optional[Any] = None


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
    match_ids: List[str]
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


# ── Meta-Cluster & Similarity Models ───────────────────────────────────────
class MetaChildCluster(BaseModel):
    cluster_id: str
    label: str
    chunk_count: int
    color: Optional[str] = None
    centroid: Optional[List[float]] = None


class MetaClusterInfo(BaseModel):
    meta_cluster_id: str
    label: str
    child_clusters: List[MetaChildCluster]


class SimilarClusterInfo(BaseModel):
    cluster_id: str
    label: str
    similarity: float
    chunk_count: int
    color: Optional[str] = None


class SimilarClustersResponse(BaseModel):
    target_cluster_id: str
    target_label: str
    similar_clusters: List[SimilarClusterInfo]


# ── Meta-Cluster Endpoints ─────────────────────────────────────────────────
@app.get("/meta_clusters", response_model=List[MetaClusterInfo])
async def get_meta_clusters():
    """Return the full hierarchy of meta-clusters with child clusters."""
    if state.df_viz.empty:
        return []

    df = state.df_viz
    if "meta_cluster_id" not in df.columns or "meta_cluster_label" not in df.columns:
        return []

    meta_centroids_df = _compute_meta_centroids(df)
    if meta_centroids_df.empty:
        return []

    centroid_lookup: Dict[str, Dict[str, Any]] = {}
    for _, row in meta_centroids_df.iterrows():
        cid = str(row["cluster_id"])
        centroid_lookup[cid] = {
            "centroid": row["centroid"],
            "meta_cluster_id": str(row["meta_cluster_id"]),
            "chunk_count": int(row["chunk_count"]),
        }

    # Compute stable color per cluster based on 5D UMAP position
    all_centroids_arr = np.vstack([v["centroid"] for v in centroid_lookup.values()]) if centroid_lookup else np.zeros((1, 5))
    global_centroid = all_centroids_arr.mean(axis=0) if len(all_centroids_arr) > 0 else np.zeros(5)

    def _cluster_color(cid: str) -> str:
        info = centroid_lookup.get(cid)
        if not info:
            return "#6b7280"
        cent = info["centroid"]
        dx = float(cent[0] - global_centroid[0])
        dy = float(cent[1] - global_centroid[1])
        if not math.isfinite(dx) or not math.isfinite(dy):
            return "#6b7280"
        if abs(dx) < 1e-12 and abs(dy) < 1e-12:
            return "hsl(210, 75%, 45%)"
        angle = math.degrees(math.atan2(dy, dx)) % 360.0
        return f"hsl({int(round(angle))}, 75%, 45%)"

    # Group by meta_cluster_id
    meta_groups: Dict[str, Dict[str, Any]] = {}
    for cid_str in df["display_topic_id"].astype(str).unique():
        child_df = df[df["display_topic_id"] == cid_str]
        if child_df.empty:
            continue
        mid = str(child_df.iloc[0].get("meta_cluster_id", "unknown"))
        mlabel = str(child_df.iloc[0].get("meta_cluster_label", f"Meta {mid}"))
        if mid not in meta_groups:
            meta_groups[mid] = {"id": mid, "label": mlabel, "children": []}
        color = _cluster_color(cid_str)
        centroid_list = None
        cl_info = centroid_lookup.get(cid_str)
        if cl_info:
            centroid_list = cl_info["centroid"].tolist()
        meta_groups[mid]["children"].append({
            "cluster_id": cid_str,
            "label": str(child_df.iloc[0].get("cluster_label", cid_str)),
            "chunk_count": len(child_df),
            "color": color,
            "centroid": centroid_list,
        })

    for mid in meta_groups:
        meta_groups[mid]["children"].sort(key=lambda c: -c["chunk_count"])

    result = [
        MetaClusterInfo(
            meta_cluster_id=mg["id"],
            label=mg["label"],
            child_clusters=[MetaChildCluster(**c) for c in mg["children"]],
        )
        for mg in meta_groups.values()
    ]
    result.sort(key=lambda m: -sum(c.chunk_count for c in m.child_clusters))
    return result


@app.get("/similar_clusters", response_model=SimilarClustersResponse)
async def get_similar_clusters(
    cluster_id: str = Query(..., description="Cluster ID to find similar clusters for"),
    limit: int = Query(MAX_SIMILAR_RESULTS, ge=1, le=100),
    min_similarity: float = Query(SIMILARITY_THRESHOLD, ge=0.0, le=1.0),
):
    """Return clusters ranked by cosine similarity to the target cluster centroid."""
    if state.df_viz.empty:
        raise HTTPException(status_code=503, detail="Data not loaded")

    df = state.df_viz
    centroid_data = _compute_meta_centroids(df)
    if centroid_data.empty:
        raise HTTPException(status_code=503, detail="No cluster centroids available")

    target_row = centroid_data[centroid_data["cluster_id"] == cluster_id]
    if target_row.empty:
        base_id = cluster_id.split(".")[0] if "." in cluster_id else cluster_id
        target_row = centroid_data[centroid_data["cluster_id"] == base_id]

    if target_row.empty:
        raise HTTPException(status_code=404, detail=f"Cluster {cluster_id} not found")

    target_centroid = target_row.iloc[0]["centroid"]
    target_meta_id = str(target_row.iloc[0]["meta_cluster_id"])
    child_df = df[df["display_topic_id"] == cluster_id]
    target_label = str(child_df.iloc[0].get("cluster_label", cluster_id)) if not child_df.empty else cluster_id

    target_norm = np.linalg.norm(target_centroid) or 1e-9
    target_normalized = target_centroid / target_norm

    candidates = centroid_data[
        (centroid_data["meta_cluster_id"] == target_meta_id) &
        (centroid_data["cluster_id"] != cluster_id)
    ]

    if candidates.empty:
        return SimilarClustersResponse(
            target_cluster_id=cluster_id,
            target_label=target_label,
            similar_clusters=[],
        )

    candidate_centroids = np.stack(candidates["centroid"].values)
    candidate_norms = np.linalg.norm(candidate_centroids, axis=1, keepdims=True)
    candidate_norms[candidate_norms == 0] = 1.0
    candidate_normalized = candidate_centroids / candidate_norms

    sims = candidate_normalized @ target_normalized
    sims_np = np.clip(sims, 0, 1)

    results = []
    for idx_idx, (_, crow) in enumerate(candidates.iterrows()):
        sim_val = float(sims_np[idx_idx])
        if sim_val >= min_similarity:
            cid_str = str(crow["cluster_id"])
            cchild_df = df[df["display_topic_id"] == cid_str]
            clabel = str(cchild_df.iloc[0].get("cluster_label", cid_str)) if not cchild_df.empty else cid_str
            results.append({
                "cluster_id": cid_str,
                "label": clabel,
                "similarity": round(sim_val, 4),
                "chunk_count": int(crow["chunk_count"]),
            })

    results.sort(key=lambda r: -r["similarity"])
    results = results[:limit]

    return SimilarClustersResponse(
        target_cluster_id=cluster_id,
        target_label=target_label,
        similar_clusters=[SimilarClusterInfo(**r) for r in results],
    )


# ── Cluster Colors Endpoint ────────────────────────────────────────────────
@app.get("/cluster_colors")
async def get_cluster_colors():
    """Return a dict of cluster_id → color for all clusters."""
    if state.df_viz.empty:
        return {}

    df = state.df_viz
    if "display_topic_id" not in df.columns:
        return {}

    # Compute stable color per cluster based on 5D UMAP position (same logic as meta_clusters)
    centroids_arr = []
    for cid, group in df.groupby("display_topic_id"):
        vectors = np.stack(group["vector"].values) if "vector" in group.columns else np.zeros((1, 5))
        centroid = vectors.mean(axis=0) if len(vectors) > 0 else np.zeros(5)
        centroids_arr.append((str(cid), centroid))

    if not centroids_arr:
        return {}

    all_cents = np.array([c[1] for c in centroids_arr])
    global_cent = all_cents.mean(axis=0)

    def _color(cid_str):
        cent = dict(centroids_arr).get(cid_str, np.zeros(5))
        dx = float(cent[0] - global_cent[0])
        dy = float(cent[1] - global_cent[1])
        if not math.isfinite(dx) or not math.isfinite(dy):
            return "#6b7280"
        if abs(dx) < 1e-12 and abs(dy) < 1e-12:
            return "hsl(210, 75%, 45%)"
        angle = math.degrees(math.atan2(dy, dx)) % 360.0
        return f"hsl({int(round(angle))}, 75%, 45%)"

    return {cid: _color(cid) for cid, _ in centroids_arr}


# ── Original Endpoints ─────────────────────────────────────────────────────
@app.get("/note_content", response_model=NoteContent)
async def get_note_content(
    title: str,
    chunk_index: int,
    creation_date: Optional[str] = None,
    modification_date: Optional[str] = None,
):
    """Get full content for a specific chunk."""
    if state.df_viz.empty:
        raise HTTPException(status_code=503, detail="Data not loaded")

    try:
        mask = (state.df_viz['title'] == title) & (state.df_viz['chunk_index'] == chunk_index)
        if creation_date is not None and 'creation_date' in state.df_viz.columns:
            mask = mask & (state.df_viz['creation_date'].astype(str) == str(creation_date))
        if modification_date is not None and 'modification_date' in state.df_viz.columns:
            mask = mask & (state.df_viz['modification_date'].astype(str) == str(modification_date))
        row = state.df_viz[mask]

        if row.empty:
            raise HTTPException(status_code=404, detail="Chunk not found")

        content = row.iloc[0].get('chunk_content', '')
        if pd.isna(content) or content == '':
            content = row.iloc[0].get('text', '')
            if pd.isna(content):
                content = ""

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
        for col in ['creation_date', 'modification_date', 'cluster_label']:
            if col not in working_df.columns:
                working_df[col] = 'Unclustered' if col == 'cluster_label' else ''
            else:
                working_df[col] = working_df[col].astype(str).fillna('Unclustered' if col == 'cluster_label' else '')

        active_rows = working_df[working_df[cluster_col] == str(active_cluster_id)].copy()
        if active_rows.empty:
            return SidebarResponse(active_cluster_id=active_cluster_id, notes=[])

        note_identity_cols = ['title', 'creation_date', 'modification_date']
        note_keys_df = active_rows[note_identity_cols].drop_duplicates().copy()
        note_keys_df['note_key'] = (
            note_keys_df['title'] + '|||' + note_keys_df['creation_date'] + '|||' + note_keys_df['modification_date']
        )

        merged = working_df.merge(note_keys_df, on=note_identity_cols, how='inner')
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

                chunks.append(SidebarChunk(
                    chunk_index=chunk_index_val,
                    cluster_id=chunk_cluster_id,
                    cluster_name=str(row.get('cluster_label', 'Unclustered')),
                    in_cluster=in_cluster,
                    text=chunk_text,
                ))

            notes.append(SidebarNote(
                note_key=str(note_key),
                title=title,
                creation_date=creation_date,
                modification_date=modification_date,
                chunks=chunks,
            ))

        return SidebarResponse(active_cluster_id=active_cluster_id, notes=notes)
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error building cluster sidebar: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/points", response_model=List[NotePoint])
async def get_points():
    """Get all points for visualization."""
    if state.df_viz.empty:
        return []

    valid_df = state.df_viz.where(pd.notnull(state.df_viz), None)
    points_df = valid_df.copy()
    points_df['cluster_id'] = points_df['display_topic_id']

    records = points_df[[
        'unique_key', 'title', 'chunk_index', 'total_chunks', 'cluster_id',
        'base_topic_id', 'display_topic_id', 'cluster_label',
        'umap_x', 'umap_y', 'umap_z', 'creation_date', 'modification_date'
    ]].to_dict(orient='records')

    return records


@app.get("/search", response_model=SearchResponse)
async def search(q: str = Query(..., min_length=1), limit: int = 1000, max_distance: float = 0.8):
    """Search notes and return matches + IDs."""
    if state.table is None:
        raise HTTPException(status_code=503, detail="Database not initialized")

    print(f"Searching for: {q} (threshold: {max_distance})")

    embedding_fn = lambda query: get_query_embedding(query)

    try:
        results = search_and_combine_results(
            state.table, q, display_limit=limit,
            max_distance=max_distance, compute_query_embedding=embedding_fn,
        )
    except Exception as e:
        print(f"Search error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    formatted_results = []
    match_ids = []

    cluster_map = {}
    cluster_id_map = {}
    base_topic_id_map = {}
    display_topic_id_map = {}
    if not state.df_viz.empty:
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

        cluster = cluster_map.get(unique_key, "Unknown")
        base_topic_id = str(base_topic_id_map.get(unique_key, cluster_id_map.get(unique_key, "-1")))
        display_topic_id = str(display_topic_id_map.get(unique_key, base_topic_id))

        res_obj = SearchResult(
            unique_key=unique_key, title=title, chunk_index=idx,
            total_chunks=total, distance=score,
            cluster_id=display_topic_id, base_topic_id=base_topic_id,
            display_topic_id=display_topic_id, cluster_label=cluster, preview=preview,
        )
        formatted_results.append(res_obj)
        match_ids.append(unique_key)

    unique_titles_found = set(r.title for r in formatted_results)

    print(f"Found {len(formatted_results)} matching chunks across {len(unique_titles_found)} notes.")
    for i, res in enumerate(formatted_results):
        cid = res.cluster_id if res.cluster_id and res.cluster_id != '-1' else res.cluster_label
        print(f"  {i+1}. {res.title} (Chunk {res.chunk_index + 1} of {res.total_chunks or '?'}) [Score: {res.distance:.3f}, Cluster: {cid}]")

    return SearchResponse(
        results=formatted_results, match_ids=match_ids,
        stats=SearchStats(total_chunks=len(formatted_results), unique_notes=len(unique_titles_found)),
    )


@app.get("/health")
async def health():
    return {"status": "ok", "loaded_rows": len(state.df_viz)}


@app.get("/debug_state")
async def debug_state():
    """Return internal debug info."""
    try:
        loaded = 0 if state.df_viz is None or state.df_viz.empty else len(state.df_viz)
        cols = [] if state.df_viz is None or state.df_viz.empty else list(state.df_viz.columns)
        sample = [] if state.df_viz is None or state.df_viz.empty else state.df_viz.head(5).to_dict(orient='records')
        return {
            'loaded_rows': loaded, 'columns': cols,
            'sample_count': len(sample), 'sample': sample,
            'table_present': state.table is not None,
            'db_path': str(DB_PATH), 'table_name': TABLE_NAME,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/reload_data")
async def reload_data():
    """Force reload data from LanceDB and recompute projections."""
    try:
        load_and_process_data()
        return {"reloaded": True, "loaded_rows": len(state.df_viz)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Interaction Tracking Endpoints ─────────────────────────────────────────
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
    """Log an interaction event for a note (opened/modified)."""
    try:
        if request.event_type not in ["opened", "modified"]:
            raise HTTPException(status_code=400, detail="event_type must be 'opened' or 'modified'")

        db = NotesDatabase(db_path=DB_PATH)
        notes_table = db.get_or_create_table()

        chunks = notes_table.to_pandas()
        note_titles = chunks[chunks['title'] == request.title]['title'].tolist()

        if not note_titles:
            return InteractionResponse(
                success=False, message=f"Note '{request.title}' not found in database"
            )

        if request.event_type == "opened":
            db.log_note_opened(request.title)
        else:
            db.log_note_modified(request.title)

        last_opened = db.get_last_opened(request.title)

        return InteractionResponse(
            success=True,
            message=f"Successfully logged {request.event_type} event for '{request.title}'",
            last_opened=last_opened,
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to log interaction: {str(e)}")


@app.get("/interaction/{title}")
async def get_interaction(title: str):
    """Get interaction data for a specific note."""
    try:
        db = NotesDatabase(db_path=DB_PATH)
        last_opened = db.get_last_opened(title)
        interaction_log = db.get_interaction_log(title)
        if interaction_log is None:
            interaction_log = []

        return InteractionLogResponse(
            title=title, last_opened=last_opened, interaction_log=interaction_log,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get interaction data: {str(e)}")


@app.get("/interactions/list")
async def list_interactions(limit: Optional[int] = None):
    """Get a list of all notes with their last_opened timestamps."""
    try:
        db = NotesDatabase(db_path=DB_PATH)
        interactions = db.list_all_interactions(limit=limit)
        return {"interactions": interactions, "count": len(interactions)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list interactions: {str(e)}")


# ── Daily History Endpoints ────────────────────────────────────────────────
class DayResponse(BaseModel):
    date: str
    titles: List[str]
    notes: List[Dict[str, Any]]


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
    if interactions_df.empty or "interaction_log" not in interactions_df.columns:
        return history_by_title

    for _, row in interactions_df.iterrows():
        title = str(row.get("title", "")).strip()
        if not title:
            continue

        events = _parse_interaction_events(row.get("interaction_log", "[]"))
        if not events:
            continue

        matching_timestamps = [
            str(ev.get("dt", "")) for ev in events
            if _event_date(ev.get("dt")) == date_str
        ]
        if not matching_timestamps:
            continue

        matching_timestamps.sort()
        title_info = history_by_title.setdefault(
            title, {
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
    """Get all distinct note titles and their full metadata opened on a given date."""
    try:
        if not re.match(r'^\d{4}-\d{2}-\d{2}$', date_str):
            raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD")

        history_by_title = _collect_history_by_title(date_str)
        active_titles = sorted(history_by_title.keys())

        if not active_titles:
            return {"date": date_str, "titles": [], "notes": []}

        print(f"Found {len(active_titles)} titles for {date_str}")

        if state.df_viz.empty:
            load_and_process_data()

        if not state.df_viz.empty:
            working_df = state.df_viz.copy()
            working_df['title'] = working_df['title'].astype(str)
            for col in ['creation_date', 'modification_date', 'cluster_label', 'cluster_id', 'display_topic_id']:
                if col not in working_df.columns:
                    working_df[col] = 'Unclustered' if col == 'cluster_label' else '-1'
                else:
                    working_df[col] = working_df[col].astype(str).fillna(
                        'Unclustered' if col == 'cluster_label' else '-1'
                    )

            notes_df = working_df[working_df['title'].isin(active_titles)].copy()
            note_identity_cols = ['title', 'creation_date', 'modification_date']
            note_keys_df = notes_df[note_identity_cols].drop_duplicates().copy()
            note_keys_df['note_key'] = (
                note_keys_df['title'] + '|||' + note_keys_df['creation_date'] + '|||' + note_keys_df['modification_date']
            )

            merged = working_df.merge(note_keys_df, on=note_identity_cols, how='inner')
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
                    ck = str(row.get('display_topic_id', row.get('cluster_id', '-1')))
                    if not ck or ck == 'nan':
                        ck = '-1'
                    cluster_counts[ck] = cluster_counts.get(ck, 0) + 1
                    if ck not in cluster_first_seen:
                        cluster_first_seen[ck] = position

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

                    ckid = str(row.get('display_topic_id', row.get('cluster_id', '-1')))
                    if not ckid or ckid == 'nan':
                        ckid = '-1'

                    chunks.append({
                        "chunk_index": chunk_index_val,
                        "cluster_id": ckid,
                        "cluster_name": str(row.get('cluster_label', 'Unclustered')),
                        "in_cluster": ckid == primary_cluster_id,
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