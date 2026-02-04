"""
search_and_combine_results - Python port of the enhanced semantic search function.

This function is written to be agnostic to the exact LanceDB (or other) Table
API you are using. It will try a few common method signatures but expects
`notes_table` to expose a search-like capability that returns an iterable of
"chunks". Each chunk is expected to have (when available) attributes/keys:

- title
- content
- creation_date
- modification_date
- chunk_index
- total_chunks
- chunk_content
- vector
- _distance

You can pass an optional `compute_query_embedding` callable that accepts a query
string and returns a vector (list[float]) for use when scoring FTS results.

Example usage:
    results = search_and_combine_results(notes_table, "my search query",
                                         display_limit=5,
                                         min_cosine_similarity=0.05,
                                         compute_query_embedding=my_embedding_fn)
"""
from typing import Any, Callable, Dict, Iterable, List, Optional
import math
import re
import traceback
import numpy as np


def _ensure_list(obj: Any) -> List[Any]:
    """Try to convert returned search result into a plain Python list."""
    if obj is None:
        return []
    if isinstance(obj, list):
        return obj
    # Common JS-like method names
    for method in ("to_list", "to_list()", "toArray", "to_array", "to_pylist", "to_python"):
        if hasattr(obj, method):
            try:
                return getattr(obj, method)()
            except Exception:
                pass
    # Check callable conversion
    try:
        return list(obj)
    except Exception:
        return []


def _get_field(chunk: Any, field: str, default: Any = None) -> Any:
    """Get field from dict-like or object-like chunk safely."""
    if chunk is None:
        return default
    if isinstance(chunk, dict):
        return chunk.get(field, default)
    return getattr(chunk, field, default)


def _cosine_similarity(a: List[float], b: List[float]) -> float:
    """Compute cosine similarity using numpy."""
    if not a or not b or len(a) != len(b):
        return 0.0
    try:
        a_arr = np.array(a, dtype=np.float32)
        b_arr = np.array(b, dtype=np.float32)
        norm_a = np.linalg.norm(a_arr)
        norm_b = np.linalg.norm(b_arr)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a_arr, b_arr) / (norm_a * norm_b))
    except Exception:
        return 0.0


def search_and_combine_results(
    notes_table: Any,
    query: str,
    display_limit: int = 5,
    max_distance: float = 2.0,
    compute_query_embedding: Optional[Callable[[str], List[float]]] = None,
) -> List[Dict[str, Any]]:
    """
    Perform an enhanced multi-strategy search over a notes table and return combined results.

    Strategies:
      1) Vector semantic search (if supported by the table)
      2) Full-text search (FTS) on chunk_content with optional embedding re-scoring
      3) Database-level exact phrase / SQL-style LIKE search (or fallback scanning)

    Returns a list of dicts with keys:
      title, creation_date, modification_date, _relevance_score, _source,
      _best_chunk_index, _total_chunks, _matching_chunk_preview
    """
    print(f'🔍 Semantic search for: "{query}"')
    # Try to obtain a row count if available
    try:
        count = None
        if hasattr(notes_table, "count_rows"):
            count = notes_table.count_rows()
        elif hasattr(notes_table, "countRows"):
            count = notes_table.countRows()
        elif hasattr(notes_table, "count"):
            count = notes_table.count()
        if count is not None:
            print(f"📊 Table has {count} chunks")
    except Exception:
        # Not critical; continue
        pass

    chunk_results: List[Dict[str, Any]] = []

    # Compute query embedding once at the beginning for vector search
    query_embedding = None
    if compute_query_embedding is not None:
        try:
            query_embedding = compute_query_embedding(query)
        except Exception as e:
            print(f"⚠️ Could not compute query embedding: {e}")

    # Strategy 1: Vector search on chunks
    print("\n1️⃣ Vector semantic search on chunks...")
    try:
        vector_results_raw = None
        # If we have the query embedding, use it for vector search
        if query_embedding:
            try:
                vector_results_raw = notes_table.search(query_embedding).limit(display_limit * 2).to_list()
            except Exception:
                try:
                    # Try alternative signatures
                    vector_results_raw = notes_table.search(query_embedding, vector_column_name="embedding").limit(display_limit * 2).to_list()
                except Exception:
                    vector_results_raw = None
        
        vector_results = _ensure_list(vector_results_raw)
        if vector_results:
            print(f"🎯 Retrieved {len(vector_results)} raw candidates from vector store")
            seen_vector_ids = set()
            for chunk in vector_results:
                distance = _get_field(chunk, "_distance", 0) or 0
                # Convert possible object/str to float safely
                try:
                    distance = float(distance)
                except Exception:
                    distance = 0.0
                
                if distance <= max_distance:
                    title = _get_field(chunk, "title", "<untitled>")
                    chunk_index = _get_field(chunk, "chunk_index", 0)
                    chunk_id = f"{title}_{chunk_index}"
                    
                    if chunk_id in seen_vector_ids:
                        continue
                    seen_vector_ids.add(chunk_id)

                    chunk_results.append({
                        "title": title,
                        "content": _get_field(chunk, "content"),
                        "creation_date": _get_field(chunk, "creation_date"),
                        "modification_date": _get_field(chunk, "modification_date"),
                        "_relevance_score": distance,
                        "_source": "vector_semantic",
                        "_chunk_index": chunk_index,
                        "_total_chunks": _get_field(chunk, "total_chunks"),
                        "_matching_chunk_content": _get_field(chunk, "chunk_content"),
                        "_chunk_id": chunk_id,  # Unique identifier
                        "cluster_id": _get_field(chunk, "cluster_id"),
                        "cluster_label": _get_field(chunk, "cluster_label")
                    })
        else:
            print("📋 No vector results (or vector search not available)")
        
        # Get unique note titles from chunks for summary
        vector_chunk_count = len([c for c in chunk_results if c.get('_source') == 'vector_semantic'])
        unique_titles = set(c['title'] for c in chunk_results if c.get('_source') == 'vector_semantic')
        print(f"📋 Vector search: {vector_chunk_count} chunks from {len(unique_titles)} unique notes")
    except Exception as e:
        print(f"❌ Vector Error: {getattr(e, 'message', repr(e))}")
        traceback.print_exc()

    # Strategy 2: FTS search on chunk content
    print("\n2️⃣ Full-text search on chunks...")
    try:
        fts_raw = None
        try:
            # Use correct FTS syntax
            fts_raw = notes_table.search(query, "fts").limit(display_limit * 2)
        except Exception as e:
            print(f"⚠️ FTS search failed: {e}")
            fts_raw = None

        fts_results = _ensure_list(fts_raw)

        # Check for duplicates from vector search to avoid double-counting
        existing_chunk_ids = set(c.get('_chunk_id', '') for c in chunk_results)
        
        for chunk in fts_results:
            title = _get_field(chunk, "title", "<untitled>")
            chunk_index = _get_field(chunk, "chunk_index", 0)
            chunk_id = f"{title}_{chunk_index}"
            
            # Skip if this exact chunk was already found in vector search
            if chunk_id in existing_chunk_ids:
                continue
            
            existing_chunk_ids.add(chunk_id)
                
            score = 70.0  # fallback
            chunk_vector = _get_field(chunk, "embedding")
            if query_embedding and isinstance(chunk_vector, (list, tuple)) and len(chunk_vector) == len(query_embedding):
                try:
                    score = max(0.0, _cosine_similarity(query_embedding, list(chunk_vector))) * 100.0
                except Exception:
                    score = 70.0
            
            chunk_results.append({
                "title": title,
                "content": _get_field(chunk, "content"),
                "creation_date": _get_field(chunk, "creation_date"),
                "modification_date": _get_field(chunk, "modification_date"),
                "_relevance_score": score,
                "_source": "fts",
                "_chunk_index": chunk_index,
                "_total_chunks": _get_field(chunk, "total_chunks"),
                "_matching_chunk_content": _get_field(chunk, "chunk_content"),
                "_chunk_id": chunk_id,
                "cluster_id": _get_field(chunk, "cluster_id"),
                "cluster_label": _get_field(chunk, "cluster_label")
            })

        print(f"📝 FTS results: {len(fts_results)} chunks")
    except Exception as e:
        print(f"❌ FTS Error: {getattr(e, 'message', repr(e))}")
        traceback.print_exc()

    # Strategy 3: Database-level exact phrase matching (much more efficient)
    print("\n3️⃣ Database-level exact phrase search...")
    try:
        # simple words for LIKE filtering
        query_words = [w for w in re.split(r"\s+", query.lower()) if len(w) > 2]
        exact_matches = []
        if query_words:
            # Build SQL-like filter string
            like_clauses = " AND ".join([f"LOWER(chunk_content) LIKE '%{w}%'" for w in query_words])
            sql_filter = like_clauses

            exact_matches_raw = None
            # Try builder-ish API: notes_table.search("").where(sql_filter).limit(100)
            try:
                builder = notes_table.search("")  # may return builder
                if hasattr(builder, "where") and hasattr(builder, "limit"):
                    exact_matches_raw = builder.where(sql_filter).limit(100)
                # If we got something try to materialize it
                exact_matches = _ensure_list(exact_matches_raw)
            except Exception:
                exact_matches = []

            # If builder path failed, try a simple search + filter fallback
            if not exact_matches:
                try:
                    # attempt a .search("") with limit
                    fallback_raw = None
                    try:
                        fallback_raw = notes_table.search("").limit(100)
                    except Exception:
                        try:
                            fallback_raw = notes_table.search("", limit=100)
                        except Exception:
                            fallback_raw = None
                    exact_matches = _ensure_list(fallback_raw)
                except Exception:
                    exact_matches = []

        if exact_matches:
            print(f"📋 Database exact matches: {len(exact_matches)} chunks")
            existing_chunk_ids = set(c.get('_chunk_id', '') for c in chunk_results)
            
            for chunk in exact_matches:
                title = _get_field(chunk, "title", "<untitled>")
                chunk_index = _get_field(chunk, "chunk_index", 0)
                chunk_id = f"{title}_{chunk_index}"
                
                # Skip if this exact chunk was already found
                if chunk_id in existing_chunk_ids:
                    continue
                
                existing_chunk_ids.add(chunk_id)
                    
                chunk_content = (_get_field(chunk, "chunk_content") or "").lower()
                title_low = (_get_field(chunk, "title") or "").lower()
                is_exact_match = (query.lower() in chunk_content) or (query.lower() in title_low)
                
                chunk_results.append({
                    "title": title,
                    "content": _get_field(chunk, "content"),
                    "creation_date": _get_field(chunk, "creation_date"),
                    "modification_date": _get_field(chunk, "modification_date"),
                    "_relevance_score": 0.0 if is_exact_match else 0.05,
                    "_source": "exact_match" if is_exact_match else "partial_match",
                    "_chunk_index": chunk_index,
                    "_total_chunks": _get_field(chunk, "total_chunks"),
                    "_matching_chunk_content": _get_field(chunk, "chunk_content"),
                    "_chunk_id": chunk_id,
                    "cluster_id": _get_field(chunk, "cluster_id"),
                    "cluster_label": _get_field(chunk, "cluster_label")
                })
        else:
            # If there were no exact_matches (maybe queryWords empty), do nothing here
            pass
    except Exception as e:
        print(f"❌ Database search error: {getattr(e, 'message', repr(e))}")
        traceback.print_exc()
        # Fallback scanning approach
        print("🔄 Trying fallback search...")
        try:
            fallback_raw = None
            try:
                fallback_raw = notes_table.search("").limit(1000)
            except Exception:
                try:
                    fallback_raw = notes_table.search("", limit=1000)
                except Exception:
                    fallback_raw = None
            fallback_results = _ensure_list(fallback_raw)
            # Use word boundary regex to try to find exact words
            safe_query = re.escape(query)
            query_regex = re.compile(r"\b" + safe_query + r"\b", flags=re.IGNORECASE)
            matches = []
            for chunk in fallback_results:
                title_text = _get_field(chunk, "title", "") or ""
                content_text = _get_field(chunk, "chunk_content", "") or ""
                if query_regex.search(title_text) or query_regex.search(content_text):
                    matches.append(chunk)
            print(f"📋 Fallback matches: {len(matches)} chunks")
            existing_chunk_ids = set(c.get('_chunk_id', '') for c in chunk_results)
            
            for chunk in matches:
                title = _get_field(chunk, "title", "<untitled>")
                chunk_index = _get_field(chunk, "chunk_index", 0)
                chunk_id = f"{title}_{chunk_index}"
                
                # Skip if this exact chunk was already found
                if chunk_id in existing_chunk_ids:
                    continue
                
                existing_chunk_ids.add(chunk_id)
                    
                chunk_results.append({
                    "title": title,
                    "content": _get_field(chunk, "content"),
                    "creation_date": _get_field(chunk, "creation_date"),
                    "modification_date": _get_field(chunk, "modification_date"),
                    "_relevance_score": 0.0,
                    "_source": "fallback_exact",
                    "_chunk_index": chunk_index,
                    "_total_chunks": _get_field(chunk, "total_chunks"),
                    "_matching_chunk_content": _get_field(chunk, "chunk_content"),
                    "_chunk_id": chunk_id,
                    "cluster_id": _get_field(chunk, "cluster_id"),
                    "cluster_label": _get_field(chunk, "cluster_label")
                })
        except Exception as fallback_error:
            print(f"❌ Fallback also failed: {getattr(fallback_error, 'message', repr(fallback_error))}")
            traceback.print_exc()

    # Combine and rank results
    combined_results = sorted(chunk_results, key=lambda r: r.get("_relevance_score", 1000.0), reverse=False)
    
    # Count unique notes for summary
    unique_notes = set(c['title'] for c in combined_results)
    print(f"\n📊 Final results: {len(combined_results)} chunks from {len(unique_notes)} unique notes")

    if combined_results:
        for idx, result in enumerate(combined_results[:display_limit]):
            score = result.get("_relevance_score", 0.0)
            source = result.get("_source", "unknown")
            chunk_idx = result.get("_chunk_index", "?")
            total_chunks = result.get("_total_chunks", "?")
            print(f'  {idx + 1}. "{result.get("title")}" (score: {score:.1f}, source: {source}, chunk: {chunk_idx}/{total_chunks})')

    # Map to the final output shape
    final = []
    for result in combined_results:
        final.append(
            {
                "title": result.get("title"),
                "creation_date": result.get("creation_date"),
                "modification_date": result.get("modification_date"),
                "_relevance_score": result.get("_relevance_score"),
                "_source": result.get("_source"),
                "_chunk_index": result.get("_chunk_index"),
                "_total_chunks": result.get("_total_chunks"),
                "_matching_chunk_preview": result.get("_matching_chunk_content"),
                "_chunk_id": result.get("_chunk_id"),
                "cluster_id": result.get("cluster_id"),
                "cluster_label": result.get("cluster_label"),
            }
        )

    return final[:display_limit]


if __name__ == "__main__":
    import sys
    from pathlib import Path
    import lancedb
    import argparse
    
    # Add repo root to path
    REPO_ROOT = Path(__file__).resolve().parent.parent.parent
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    
    from backend.scripts.main import NotesDatabase, EmbeddingModel
    
    parser = argparse.ArgumentParser(description="Search notes in the database")
    parser.add_argument("query", nargs="?", default="", help="Search query")
    parser.add_argument("--limit", type=int, default=5, help="Number of results to display")
    args = parser.parse_args()
    
    if not args.query:
        print("Usage: python scripts/search_notes.py <query> [--limit N]")
        print("Example: python scripts/search_notes.py 'machine learning' --limit 10")
        sys.exit(1)
    
    # Connect to LanceDB database with the proper data directory
    DATA_DIR = Path.home() / ".mcp-apple-notes"
    DB_PATH = DATA_DIR / "data"
    
    print(f"📂 Connecting to LanceDB at: {DB_PATH}")
    
    # Initialize database connection
    db = NotesDatabase(db_path=DB_PATH)
    notes_table = db.get_or_create_table()
    
    # Check if table has data
    try:
        row_count = notes_table.count_rows()
        print(f"📊 Database contains {row_count} chunks")
        if row_count == 0:
            print("⚠️ No data in database. Run: python scripts/fetch_and_chunk_notes.py")
            sys.exit(0)
    except Exception as e:
        print(f"⚠️ Could not count rows: {e}")
    
    # Initialize embedding model for query embedding
    embedding_model = EmbeddingModel()
    
    def compute_query_embedding(q: str):
        """Compute embedding for search query"""
        embeddings = embedding_model.embed_texts([q], show_progress=False)
        return embeddings[0].tolist()
    
    # Perform search
    print(f"\n🔎 Searching for: '{args.query}'")
    results = search_and_combine_results(
        notes_table,
        args.query,
        display_limit=args.limit,
        compute_query_embedding=compute_query_embedding
    )
    
    print("\n" + "="*80)
    print("Search Results:")
    print("="*80)
    
    if not results:
        print("No results found.")
    else:
        # Count occurrences of each cluster in the results
        cluster_counts = {}
        cluster_labels = {}
        for r in results:
            c_id = r.get('cluster_id', 'N/A')
            cluster_counts[c_id] = cluster_counts.get(c_id, 0) + 1
            if c_id != 'N/A':
                cluster_labels[c_id] = r.get('cluster_label', 'Uncategorized')
        
        print(f"Found {len(results)} results across {len(cluster_counts)} clusters:")
        for c_id, count in sorted(cluster_counts.items(), key=lambda x: x[1], reverse=True):
            label = cluster_labels.get(c_id, 'Uncategorized') if c_id != 'N/A' else 'N/A'
            print(f"  - {label} (ID: {c_id}): {count} notes")

        for idx, result in enumerate(results, 1):
            title = result.get('title', 'Untitled')
            c_id = result.get('cluster_id', 'N/A')
            c_label = result.get('cluster_label', 'Uncategorized')
            c_count = cluster_counts.get(c_id, 0)

            print(f"\n{idx}. {title}")
            print(f"   Created: {result.get('creation_date', 'N/A')}")
            print(f"   Modified: {result.get('modification_date', 'N/A')}")
            print(f"   Score: {result.get('_relevance_score', 0):.2f}")
            print(f"   Source: {result.get('_source', 'unknown')}")
            print(f"   Chunk ID: {result.get('_chunk_id', 'N/A')}")
            print(f"   Cluster: {c_label} (ID: {c_id})")
            print(f"   Matches in this Cluster: {c_count}")
            print(f"   Preview: {result.get('_matching_chunk_preview', '')[:200]}...")