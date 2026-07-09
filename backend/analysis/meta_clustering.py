"""
Meta-Clustering Module

Computes a hierarchy of clusters by grouping BERTopic clusters into
"meta-clusters" based on centroid similarity (cosine distance).

Pipeline:
  1. Compute cluster centroids (average of chunk embeddings)
  2. Run AgglomerativeClustering on centroids with cosine distance
  3. Assign meta-cluster labels via LLM (primary) or TF-IDF (fallback)
  4. Return DataFrame with added meta_cluster_id and meta_cluster_label columns

Constants can be tuned to adjust grouping granularity.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ---------------------------------------------------------------------------
# Tunable Constants
# ---------------------------------------------------------------------------

#: Distance threshold for meta-clustering (1 - cosine_similarity).
#: Lower = tighter clusters, higher = more merged groups.
META_CLUSTER_DISTANCE_THRESHOLD: float = 0.15

#: Minimum number of child clusters a meta-cluster must have to be included.
#: Meta-clusters with fewer children than this are merged into the nearest neighbor.
MIN_META_CLUSTER_SIZE: int = 2

#: Maximum number of TF-IDF keywords used for meta-cluster naming.
TFIDF_TOP_K: int = 5


# ---------------------------------------------------------------------------
# LLM Prompt Templates
# ---------------------------------------------------------------------------

#: Prompt template for LLM-based metacluster naming.
META_CLUSTER_LABEL_PROMPT = """<|user|>
I have a group of related topic clusters. Each cluster has been given an individual descriptive title.
Here are the titles of all clusters in this group:
<cluster_titles>
[CLUSTER_TITLES]
</cluster_titles>

Task: Based on these cluster titles, provide a single concise, professional label (3-6 words) that captures the overarching theme shared by all of them.
Do NOT start the label with the word "cluster" (case-insensitive) or any phrase beginning with it.
Output ONLY the label string. No preamble, no quotes, no punctuation at the end.
<|assistant|>"""


# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------

def _generate_meta_label_via_tfidf(
    child_indices_in_df: List[int],
    docs: List[str],
    meta_id: int,
) -> str:
    """Generate a meta-cluster label using TF-IDF on child document texts.

    Parameters
    ----------
    child_indices_in_df : list of int
        Row indices (into ``docs``) belonging to this meta-cluster.
    docs : list of str
        Raw document texts.
    meta_id : int
        Meta-cluster identifier (used for fallback label).

    Returns
    -------
    str
        TF-IDF based label or "Topic Group <id>" on failure.
    """
    child_docs = [docs[i] for i in child_indices_in_df if i < len(docs) and docs[i].strip()]
    if not child_docs:
        return f"Topic Group {meta_id}"

    combined_text = " ".join(child_docs)
    tfidf = TfidfVectorizer(stop_words="english", max_features=500, ngram_range=(1, 2))
    try:
        tfidf_matrix = tfidf.fit_transform([combined_text])
        feature_names = tfidf.get_feature_names_out()
        scores = tfidf_matrix.toarray().mean(axis=0)
        top_k_indices = np.argsort(scores)[::-1][:TFIDF_TOP_K]
        keyword_parts = [feature_names[i] for i in top_k_indices if scores[i] > 0]
        if keyword_parts:
            return " & ".join(keyword_parts[:TFIDF_TOP_K])
    except Exception:
        pass
    return f"Topic Group {meta_id}"


def _generate_meta_label_via_llm(child_labels: List[str], model: str) -> Optional[str]:
    """Call Ollama to produce a metacluster name from child cluster titles."""
    try:
        import ollama
        titles_block = "\n".join(f"- {label}" for label in child_labels if label.strip())
        if not titles_block:
            return None
        prompt = META_CLUSTER_LABEL_PROMPT.replace("[CLUSTER_TITLES]", titles_block)
        response = ollama.generate(model=model, prompt=prompt)
        label = response["response"].strip().strip('"\'').rstrip(".,;:")
        return label if label else None
    except Exception as e:
        print(f"   Ollama metacluster naming failed ({e}) - falling back to TF-IDF.")
        return None


def compute_cluster_centroids(
    df: pd.DataFrame,
    vector_col: str = "vector",
    topic_col: str = "display_topic_id",
) -> pd.DataFrame:
    """Compute the centroid (average embedding) for each cluster.

    Parameters
    ----------
    df : DataFrame
        The full notes DataFrame containing embedding vectors and cluster IDs.
    vector_col : str
        Column name holding the embedding vector (list of floats).
    topic_col : str
        Column name holding the cluster identifier.

    Returns
    -------
    DataFrame
        Index = cluster_id, columns = centroid (array), cluster_label, chunk_count
    """
    # Group vectors by cluster and compute mean
    rows: List[Dict[str, object]] = []
    for cluster_id, group in df.groupby(topic_col):
        vectors = np.stack(group[vector_col].values)
        centroid = vectors.mean(axis=0)
        label = group.iloc[0].get("cluster_label", str(cluster_id))
        rows.append({
            "cluster_id": str(cluster_id),
            "centroid": centroid,
            "cluster_label": label,
            "chunk_count": len(group),
        })

    return pd.DataFrame(rows)


def compute_meta_clusters(
    df: pd.DataFrame,
    vectors: Optional[np.ndarray] = None,
    docs: Optional[List[str]] = None,
    vector_col: str = "vector",
    topic_col: str = "display_topic_id",
    ollama_model: Optional[str] = None,
) -> pd.DataFrame:
    """Assign each cluster to a meta-cluster and return the updated DataFrame.

    Parameters
    ----------
    df : DataFrame
        The full notes DataFrame (will NOT be modified in-place; a copy is returned).
    vectors : ndarray, optional
        Pre-computed embedding array matching df rows. If provided, centroids
        are computed directly from these vectors rather than from df[vector_col].
    docs : list of str, optional
        Raw document texts for TF-IDF / LLM meta-cluster naming.
    vector_col : str
        Column name holding embedding vectors in ``df`` (used when ``vectors``
        is not provided).
    topic_col : str
        Column name holding the cluster identifier per row.
    ollama_model : str, optional
        Ollama model name (e.g. ``"phi4-mini:3.8b"``) for LLM-based meta-cluster
        naming.  When provided, child cluster titles are sent to the model; TF-IDF
        is used as a fallback if the call fails.

    Returns
    -------
    DataFrame
        A copy of ``df`` with two new columns added:
         - ``meta_cluster_id``    (string, e.g. "0", "1")
         - ``meta_cluster_label`` (human-readable label for the meta-cluster)
    """
    df_out = df.copy()

    # ------------------------------------------------------------------
    # Step 1: Compute cluster centroids
    # ------------------------------------------------------------------
    centroids_df = compute_cluster_centroids(df_out, vector_col=vector_col, topic_col=topic_col)
    cluster_ids = centroids_df["cluster_id"].tolist()
    centroid_matrix = np.vstack(centroids_df["centroid"].values)
    labels_list = centroids_df["cluster_label"].tolist()
    chunk_counts = centroids_df["chunk_count"].tolist()

    if len(cluster_ids) < MIN_META_CLUSTER_SIZE:
        # Not enough clusters to meaningfully group — assign all to meta-cluster 0
        df_out = df_out.assign(
            meta_cluster_id="0",
            meta_cluster_label=labels_list[0] if labels_list else "All Notes",
        )
        return df_out

    # ------------------------------------------------------------------
    # Step 2: Compute pairwise cosine distance matrix
    # ------------------------------------------------------------------
    # Normalize rows so that cosine_similarity = dot product of normalized vectors
    norms = np.linalg.norm(centroid_matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    centroid_normalized = centroid_matrix / norms

    sim_matrix = centroid_normalized @ centroid_normalized.T   # shape: (n_clusters, n_clusters)
    dist_matrix = 1.0 - sim_matrix   # convert similarity → distance
    # Guard against tiny negative values from floating point
    np.clip(dist_matrix, 0, None, out=dist_matrix)

    # ------------------------------------------------------------------
    # Step 3: Agglomerative Clustering on centroids
    # ------------------------------------------------------------------
    meta_clusterer = AgglomerativeClustering(
        n_clusters=None,
        metric="precomputed",
        distance_threshold=META_CLUSTER_DISTANCE_THRESHOLD,
        linkage="average",
    )
    meta_labels_raw = meta_clusterer.fit_predict(dist_matrix)

    # ------------------------------------------------------------------
    # Step 4: Enforce minimum meta-cluster size (merge small groups)
    # ------------------------------------------------------------------
    unique_meta_ids = np.unique(meta_labels_raw)
    counts = np.bincount(meta_labels_raw)

    # Identify small meta-clusters (< MIN_META_CLUSTER_SIZE members)
    small_meta_ids = unique_meta_ids[counts < MIN_META_CLUSTER_SIZE]

    if len(small_meta_ids) > 0:
        # Map each small meta-cluster to its nearest larger neighbor
        assignment_map: Dict[int, int] = {}
        for small_id in small_meta_ids:
            # Find indices belonging to this small meta-cluster
            small_indices = np.where(meta_labels_raw == small_id)[0]
            small_centroid_avg = centroid_matrix[small_indices].mean(axis=0)
            small_norm = np.linalg.norm(small_centroid_avg)
            if small_norm == 0:
                continue

            # Compute distance to each potential target meta-cluster
            best_target: Optional[int] = None
            best_dist = float("inf")
            for target_id in unique_meta_ids:
                if target_id == small_id or target_id in assignment_map:
                    continue
                if counts[target_id] < MIN_META_CLUSTER_SIZE:
                    # Target is also too small — skip unless it's already been merged
                    continue
                target_indices = np.where(meta_labels_raw == target_id)[0]
                target_centroid_avg = centroid_matrix[target_indices].mean(axis=0)
                t_norm = np.linalg.norm(target_centroid_avg)
                if t_norm == 0:
                    continue

                d = 1.0 - float(
                    (small_centroid_avg / small_norm) @ (target_centroid_avg / t_norm)
                )
                if d < best_dist:
                    best_dist = d
                    best_target = int(target_id)

            if best_target is not None:
                assignment_map[small_id] = best_target

        # Apply assignments
        for small_id, target_id in assignment_map.items():
            meta_labels_raw[meta_labels_raw == small_id] = target_id

    # Re-label to be contiguous starting from 0
    unique_after = np.unique(meta_labels_raw)
    id_map = {old: new for new, old in enumerate(unique_after)}
    meta_labels_final = np.array([id_map[int(x)] for x in meta_labels_raw], dtype=int)

    # ------------------------------------------------------------------
    # Step 5: Collect child cluster labels & doc indices per metacluster
    # ------------------------------------------------------------------
    child_cluster_labels_map: Dict[int, List[str]] = {}
    child_doc_indices_map: Dict[int, List[int]] = {}

    for ci, cid in enumerate(cluster_ids):
        meta_id = int(meta_labels_final[ci])
        child_label = labels_list[ci] if labels_list[ci] else f"Cluster {cid}"
        child_cluster_labels_map.setdefault(meta_id, []).append(child_label)
        mask = df_out[topic_col] == cid
        child_doc_indices_map.setdefault(meta_id, []).extend(mask[mask].index.tolist())

    # ------------------------------------------------------------------
    # Step 6: Generate metacluster labels (LLM primary, TF-IDF fallback)
    # ------------------------------------------------------------------
    meta_cluster_label_map: Dict[int, str] = {}

    for meta_id in np.unique(meta_labels_final):
        child_labels = child_cluster_labels_map.get(int(meta_id), [])
        child_doc_indices = child_doc_indices_map.get(int(meta_id), [])

        # Primary: LLM
        if ollama_model and child_labels:
            llm_label = _generate_meta_label_via_llm(child_labels, ollama_model)
            if llm_label:
                meta_cluster_label_map[int(meta_id)] = llm_label
                continue

        # Fallback: TF-IDF
        if docs is not None and len(docs) > 0 and child_doc_indices:
            meta_cluster_label_map[int(meta_id)] = _generate_meta_label_via_tfidf(
                child_doc_indices, docs, int(meta_id)
            )
        else:
            meta_cluster_label_map[int(meta_id)] = f"Topic Group {meta_id}"

    # ------------------------------------------------------------------
    # Step 7: Build cluster-to-meta-cluster mapping
    # ------------------------------------------------------------------
    cluster_to_meta: Dict[str, Tuple[int, str]] = {}
    for ci, cid in enumerate(cluster_ids):
        meta_id = int(meta_labels_final[ci])
        label = meta_cluster_label_map.get(meta_id, f"Topic Group {meta_id}")
        cluster_to_meta[cid] = (meta_id, label)

    # ------------------------------------------------------------------
    # Step 8: Assign meta-cluster info to every row in the DataFrame
    # ------------------------------------------------------------------
    def _resolve_meta(row_topic: str) -> Tuple[str, str]:
        """Resolve meta_cluster_id and label for a single row topic."""
        # Try exact match first
        if row_topic in cluster_to_meta:
            mid, mlabel = cluster_to_meta[row_topic]
            return str(mid), mlabel
        # Fallback: try base_topic_id (strip sub-cluster suffixes like "4.0" → "4")
        base = row_topic.split(".")[0] if "." in str(row_topic) else row_topic
        if base in cluster_to_meta:
            mid, mlabel = cluster_to_meta[base]
            return str(mid), mlabel
        # Last resort: fallback to row's own topic as its own meta
        return row_topic, f"Topic {row_topic}"

    meta_ids: List[str] = []
    meta_labels_out: List[str] = []
    for topic_val in df_out[topic_col].astype(str).tolist():
        mid, mlabel = _resolve_meta(topic_val)
        meta_ids.append(mid)
        meta_labels_out.append(mlabel)

    df_out = df_out.assign(
        meta_cluster_id=meta_ids,
        meta_cluster_label=meta_labels_out,
    )

    return df_out