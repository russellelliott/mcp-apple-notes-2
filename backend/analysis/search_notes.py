"""
search_and_combine_results - Enhanced multi-strategy search with proper FTS support.

This function implements:
1) Vector semantic search (cosine similarity)
2) Full-text search (FTS) with AND/OR query expansion
3) Database-level LIKE fallback

Returns a list of dicts with keys:
  title, creation_date, modification_date, _relevance_score, _source,
   _chunk_index, _total_chunks, _matching_chunk_preview, _chunk_id,
  cluster_id, cluster_label
"""
from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Tuple
import math
import re
import traceback
import numpy as np

# ---------------------------------------------------------------------------
# Common English stop words - remove these before building keyword queries
# ---------------------------------------------------------------------------
COMMON_STOP_WORDS: Set[str] = {
    # articles, prepositions, pronouns, auxiliary verbs, conjunctions
    "a", "an", "the", "and", "or", "but", "not", "nor",
    "in", "on", "at", "to", "for", "of", "with", "by",
    "is", "it", "its", "this", "that", "these", "those",
    "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did",
    "will", "would", "could", "should", "may", "might", "shall",
    "i", "me", "my", "we", "our", "you", "your", "he", "him", "his",
    "she", "her", "they", "them", "their", "what", "which", "who",
    "whom", "how", "when", "where", "why", "if", "then", "than",
    "so", "as", "about", "up", "out", "just", "also", "too",
    "very", "can", "into", "over", "after", "before", "between",
    "through", "during", "from", "above", "below", "both", "each",
    "few", "more", "most", "other", "some", "such", "no", "any",
}


def extract_meaningful_terms(query: str) -> List[str]:
    """Extract meaningful search terms from a query, removing stop words.

    Args:
        query: Raw user search query (e.g., "What did I decide about naming BERTopic clusters?")

    Returns:
        Deduplicated list of meaningful lowercase terms (e.g., ["decide", "naming", "bertopic", "clusters"])
        Note: Removed "what" as it's a common stop word.
    """
    if not query:
        return []
    # Lowercase and split on whitespace/punctuation
    raw_terms = re.findall(r"[a-z0-9]+", query.lower())
    # Filter: skip stop words and single-char tokens
    terms = [t for t in raw_terms if t not in COMMON_STOP_WORDS and len(t) > 1]
    # Deduplicate while preserving order
    seen: Set[str] = set()
    unique: List[str] = []
    for t in terms:
        if t not in seen:
            seen.add(t)
            unique.append(t)
    return unique


def build_keyword_queries(query: str) -> Dict[str, Optional[str]]:
    """Build multiple FTS queries from a single user query.

    Args:
        query: Raw user search query (e.g., "tradeoffs I considered for topic-cluster naming")

    Returns:
        Dict with keys:
          - phrase_query: Exact quoted phrase (e.g., "\"topic cluster naming\"")
          - all_terms_query: AND-joined meaningful terms (e.g., "tradeoffs AND topic AND cluster AND naming")
          - any_terms_query: OR-joined meaningful terms (e.g., "tradeoffs OR topic OR cluster OR naming")
          - sub_phrases: List of sub-phrase queries for partial matches
    """
    terms = extract_meaningful_terms(query)

    # Extract quoted phrases from original query if present
    quoted_phrases = re.findall(r'"([^"]+)"', query)

    # Build sub-phrases from meaningful terms (last 3-4 terms for flexibility)
    sub_phrases: List[str] = []
    if len(terms) >= 3:
        # Try last 3 and last 4 terms as sub-phrases
        if len(terms) >= 4:
            sub_phrases.append(" ".join(terms[-4:]))
        sub_phrases.append(" ".join(terms[-3:]))

    return {
        "phrase_query": f'"{query.strip()}"' if quoted_phrases or len(terms) <= 2 else None,
        "quoted_phrases": [f'"{p}"' for p in quoted_phrases],
        "all_terms_query": (" AND ".join(terms) if terms else None),
        "any_terms_query": (" OR ".join(terms) if terms else None),
        "sub_phrases": sub_phrases,
        "terms": terms,
    }


def _ensure_list(obj: Any) -> List[Any]:
    """Try to convert returned search result into a plain Python list."""
    if obj is None:
        return []
    if isinstance(obj, list):
        return obj
    # Common JS-like method names
    for method in ("to_list", "toArray", "to_array", "to_pylist", "to_python"):
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


def _run_fts_query(notes_table: Any, query_str: str, limit: int = 500) -> List[Any]:
    """Run a single FTS query and return results as a list.

    Returns empty list on any failure (failure is logged by caller).

    Uses LanceDB's FTS search with the correct syntax for this version:
    table.search(query, query_type="fts") -- scans all FTS indexes automatically.
    """
    try:
        # LanceDB FTS search: use query_type="fts" (not positional "fts")
        result = notes_table.search(query_str, query_type="fts").limit(limit)
        return _ensure_list(result)
    except Exception as e:
        print(f"    FTS query failed for '{query_str[:60]}': {e}")
        return []


def _score_chunk_for_fts(
    chunk: Any,
    query: str,
    terms: List[str],
    query_embedding: Optional[List[float]],
    match_type: str = "fts",
) -> Tuple[float, str]:
    """Compute a relevance score for an FTS-matched chunk.

    Scoring tiers (highest boost to lowest):
      1. Exact phrase in chunk_content          -> +30 base + cosine boost
      2. All meaningful terms present            -> +15 base + cosine boost
      3. Most terms present (>= 75%)             -> +8 base + cosine boost
      4. Any term present                        -> +2 base per term + cosine fallback
    """
    chunk_content = (_get_field(chunk, "chunk_content") or "").lower()
    chunk_title = (_get_field(chunk, "title") or "").lower()
    full_text = chunk_content + " " + chunk_title

    score = 0.0
    description = match_type

    # --- Tier 1: Exact phrase match in content ---
    query_lower = query.lower().strip()
    if query_lower in chunk_content or query_lower in chunk_title:
        score += 30.0
        description = "exact_phrase"

    # --- Check individual term matches ---
    for term in terms:
        if term in chunk_content:
            score += 2.0    # Individual term match

    # --- Check title bonus ---
    # Count how many terms appear in the title vs just content
    title_terms = sum(1 for t in terms if t in chunk_title)
    content_terms = sum(1 for t in terms if t in chunk_content)

    if title_terms > 0:
        score += title_terms * 5.0    # Strong bonus for title matches

    if len(terms) >= 2:
        # --- Tier 2: All terms present ---
        all_present = all(t in full_text for t in terms)
        if all_present:
            score += 15.0
            description = "all_terms"

        # --- Tier 3: Most terms (>= 75%) ---
        elif content_terms / len(terms) >= 0.75:
            score += 8.0
            description = "most_terms"

    # --- Apply cosine similarity boost from embedding ---
    chunk_vector = _get_field(chunk, "vector") or _get_field(chunk, "embedding")
    if query_embedding and isinstance(chunk_vector, (list, tuple)) and len(chunk_vector) == len(query_embedding):
        try:
            cosine = max(0.0, _cosine_similarity(query_embedding, list(chunk_vector))) * 50.0
            score += cosine
            description = f"{description}_boosted"
        except Exception:
            pass

    return round(score, 2), description


def search_and_combine_results(
    notes_table: Any,
    query: str,
    display_limit: int = 5,
    max_distance: float = 2.0,
    compute_query_embedding: Optional[Callable[[str], List[float]]] = None,
) -> List[Dict[str, Any]]:
    """
    Perform an enhanced multi-strategy search over a notes table and return combined results.

    Strategies (with FTS query expansion):
         1) Vector semantic search (if supported by the table)
         2) Full-text search (FTS) with AND/OR/query-expansion on chunk_content
         3) Database-level LIKE / fallback scanning

    Returns a list of dicts sorted by ascending _relevance_score (lower = more relevant).
    Each result includes: title, creation_date, modification_date, _relevance_score,
           _source, _chunk_index, _total_chunks, _matching_chunk_preview, _chunk_id,
         cluster_id, cluster_label
    """
    print(f'Searching for: "{query}"')

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
            print(f"Table has {count} chunks")
    except Exception:
        pass

    chunk_results: List[Dict[str, Any]] = []

    # Compute query embedding once at the beginning for vector search & FTS scoring
    query_embedding = None
    if compute_query_embedding is not None:
        try:
            query_embedding = compute_query_embedding(query)
        except Exception as e:
            print(f"Could not compute query embedding: {e}")

    # Extract meaningful terms for keyword queries
    keywords = build_keyword_queries(query)
    terms = keywords["terms"]

    print(f"  Query parsed: {len(terms)} meaningful terms: {terms}")
    if keywords.get("all_terms_query"):
        print(f"  AND query:    {keywords['all_terms_query']}")
    if keywords.get("any_terms_query"):
        print(f"  OR query:     {keywords['any_terms_query']}")

    # =========================================================================
    # Strategy 1: Vector semantic search on chunks
    # =========================================================================
    print("\n1) Vector semantic search on chunks...")
    try:
        vector_results_raw = None
        if query_embedding:
            try:
                vector_results_raw = notes_table.search(query_embedding).limit(display_limit * 2).to_list()
            except Exception:
                try:
                    vector_results_raw = notes_table.search(query_embedding, vector_column_name="embedding").limit(display_limit * 2).to_list()
                except Exception:
                    vector_results_raw = None

        vector_results = _ensure_list(vector_results_raw)
        if vector_results:
            print(f"Retrieved {len(vector_results)} raw candidates from vector store")
            seen_vector_ids: Set[str] = set()
            for chunk in vector_results:
                distance = _get_field(chunk, "_distance", 0) or 0
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

                    # Convert distance to a similarity score (higher = better)
                    # For LanceDB, _distance is typically squared L2 or cosine distance
                    relevance = round(max(0.0, (1.0 - distance) * 100), 2) if distance < 1.0 else round(max(0.0, (1.0 / (1.0 + distance)) * 100), 2)

                    chunk_results.append({
                        "title": title,
                        "content": _get_field(chunk, "content"),
                        "creation_date": _get_field(chunk, "creation_date"),
                        "modification_date": _get_field(chunk, "modification_date"),
                        "_relevance_score": relevance,
                        "_source": "vector_semantic",
                        "_chunk_index": chunk_index,
                        "_total_chunks": _get_field(chunk, "total_chunks"),
                        "_matching_chunk_content": _get_field(chunk, "chunk_content"),
                        "_chunk_id": chunk_id,
                        "cluster_id": _get_field(chunk, "cluster_id"),
                        "cluster_label": _get_field(chunk, "cluster_label"),
                    })
        else:
            print("No vector results (or vector search not available)")

        vector_chunk_count = len([c for c in chunk_results if c.get('_source') == 'vector_semantic'])
        unique_titles = set(c['title'] for c in chunk_results if c.get('_source') == 'vector_semantic')
        print(f"Vector search: {vector_chunk_count} chunks from {len(unique_titles)} unique notes")
    except Exception as e:
        print(f"Vector Error: {getattr(e, 'message', repr(e))}")
        traceback.print_exc()

    # Track chunk IDs already added
    existing_chunk_ids: Set[str] = set(c.get('_chunk_id', '') for c in chunk_results)

    # =========================================================================
    # Strategy 2: Full-text search with query expansion (AND / OR / phrases)
    # =========================================================================
    print("\n2) Full-text search on chunks...")

    if not terms:
        print("  No meaningful terms extracted from query - skipping FTS")
    else:
        fts_total = 0
        fts_sources = {}

        # --- Phase A: Exact phrase match (highest priority) ---
        # Only try if the query has few terms or was quoted
        if keywords.get("phrase_query") and len(terms) <= 4:
            print(f"  Phase A: Exact phrase search...")
            results = _run_fts_query(notes_table, query.strip(), limit=100)
            phase_count = 0
            for chunk in results:
                title = _get_field(chunk, "title", "<untitled>")
                chunk_index = _get_field(chunk, "chunk_index", 0)
                chunk_id = f"{title}_{chunk_index}"

                if chunk_id not in existing_chunk_ids:
                    score, desc = _score_chunk_for_fts(chunk, query, terms, query_embedding, "exact_phrase")
                    existing_chunk_ids.add(chunk_id)
                    chunk_results.append({
                        "title": title, "content": _get_field(chunk, "content"),
                        "creation_date": _get_field(chunk, "creation_date"),
                        "modification_date": _get_field(chunk, "modification_date"),
                        "_relevance_score": score, "_source": desc,
                        "_chunk_index": chunk_index,
                        "_total_chunks": _get_field(chunk, "total_chunks"),
                        "_matching_chunk_content": _get_field(chunk, "chunk_content"),
                        "_chunk_id": chunk_id,
                        "cluster_id": _get_field(chunk, "cluster_id"),
                        "cluster_label": _get_field(chunk, "cluster_label"),
                    })
                    phase_count += 1
            fts_total += phase_count
            fts_sources["exact_phrase"] = phase_count
            print(f"  Exact phrase: {phase_count} results")

        # --- Phase B: Sub-phrase matches (flexible partial phrases) ---
        for sub_phrase in keywords.get("sub_phrases", []):
            if not sub_phrase:
                continue
            print(f"  Phase B: Sub-phrase '{sub_phrase}'...")
            results = _run_fts_query(notes_table, sub_phrase, limit=200)
            phase_count = 0
            for chunk in results:
                title = _get_field(chunk, "title", "<untitled>")
                chunk_index = _get_field(chunk, "chunk_index", 0)
                chunk_id = f"{title}_{chunk_index}"

                if chunk_id not in existing_chunk_ids:
                    score, desc = _score_chunk_for_fts(chunk, query, terms, query_embedding, f"sub_phrase:{sub_phrase[:20]}")
                    existing_chunk_ids.add(chunk_id)
                    chunk_results.append({
                        "title": title, "content": _get_field(chunk, "content"),
                        "creation_date": _get_field(chunk, "creation_date"),
                        "modification_date": _get_field(chunk, "modification_date"),
                        "_relevance_score": score, "_source": desc,
                        "_chunk_index": chunk_index,
                        "_total_chunks": _get_field(chunk, "total_chunks"),
                        "_matching_chunk_content": _get_field(chunk, "chunk_content"),
                        "_chunk_id": chunk_id,
                        "cluster_id": _get_field(chunk, "cluster_id"),
                        "cluster_label": _get_field(chunk, "cluster_label"),
                    })
                    phase_count += 1
            fts_total += phase_count
            fts_sources[f"sub_phrase:{sub_phrase[:20]}"] = phase_count
            print(f"  Sub-phrase '{sub_phrase}': {phase_count} new results")

        # --- Phase C: AND query (all terms must be present) - strong boost ---
        if keywords.get("all_terms_query"):
            print(f"  Phase C: AND query (all terms)...")
            results = _run_fts_query(notes_table, keywords["all_terms_query"], limit=500)
            phase_count = 0
            for chunk in results:
                title = _get_field(chunk, "title", "<untitled>")
                chunk_index = _get_field(chunk, "chunk_index", 0)
                chunk_id = f"{title}_{chunk_index}"

                if chunk_id not in existing_chunk_ids:
                    score, desc = _score_chunk_for_fts(chunk, query, terms, query_embedding, "all_terms_and")
                    existing_chunk_ids.add(chunk_id)
                    chunk_results.append({
                        "title": title, "content": _get_field(chunk, "content"),
                        "creation_date": _get_field(chunk, "creation_date"),
                        "modification_date": _get_field(chunk, "modification_date"),
                        "_relevance_score": score, "_source": desc,
                        "_chunk_index": chunk_index,
                        "_total_chunks": _get_field(chunk, "total_chunks"),
                        "_matching_chunk_content": _get_field(chunk, "chunk_content"),
                        "_chunk_id": chunk_id,
                        "cluster_id": _get_field(chunk, "cluster_id"),
                        "cluster_label": _get_field(chunk, "cluster_label"),
                    })
                    phase_count += 1
            fts_total += phase_count
            fts_sources["all_terms_AND"] = phase_count
            print(f"  AND query: {phase_count} new results")

        # --- Phase D: OR query (any term present) - broad fallback ---
        if keywords.get("any_terms_query"):
            print(f"  Phase D: OR query (any term)...")
            results = _run_fts_query(notes_table, keywords["any_terms_query"], limit=500)
            phase_count = 0
            for chunk in results:
                title = _get_field(chunk, "title", "<untitled>")
                chunk_index = _get_field(chunk, "chunk_index", 0)
                chunk_id = f"{title}_{chunk_index}"

                if chunk_id not in existing_chunk_ids:
                    score, desc = _score_chunk_for_fts(chunk, query, terms, query_embedding, "any_term_or")
                    existing_chunk_ids.add(chunk_id)
                    chunk_results.append({
                        "title": title, "content": _get_field(chunk, "content"),
                        "creation_date": _get_field(chunk, "creation_date"),
                        "modification_date": _get_field(chunk, "modification_date"),
                        "_relevance_score": score, "_source": desc,
                        "_chunk_index": chunk_index,
                        "_total_chunks": _get_field(chunk, "total_chunks"),
                        "_matching_chunk_content": _get_field(chunk, "chunk_content"),
                        "_chunk_id": chunk_id,
                        "cluster_id": _get_field(chunk, "cluster_id"),
                        "cluster_label": _get_field(chunk, "cluster_label"),
                    })
                    phase_count += 1
            fts_total += phase_count
            fts_sources["any_term_OR"] = phase_count
            print(f"  OR query: {phase_count} new results")

        # --- Summary for FTS ---
        source_summary = ", ".join(f"{k}:{v}" for k, v in fts_sources.items())
        print(f"\nFTS total: {fts_total} new chunks (sources: {source_summary})")

    # =========================================================================
    # Strategy 3: Database-level exact phrase / LIKE matching
    # =========================================================================
    print("\n3) Database-level exact phrase search...")
    try:
        query_words = [w for w in re.split(r"\s+", query.lower()) if len(w) > 2]
        exact_matches = []
        if query_words:
            like_clauses = " AND ".join([f"LOWER(chunk_content) LIKE '%{w}%'" for w in query_words])
            sql_filter = like_clauses

            exact_matches_raw = None
            try:
                builder = notes_table.search("")
                if hasattr(builder, "where") and hasattr(builder, "limit"):
                    exact_matches_raw = builder.where(sql_filter).limit(100)
                exact_matches = _ensure_list(exact_matches_raw)
            except Exception:
                exact_matches = []

            if not exact_matches:
                try:
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
            print(f"Database exact matches: {len(exact_matches)} chunks")
            for chunk in exact_matches:
                title = _get_field(chunk, "title", "<untitled>")
                chunk_index = _get_field(chunk, "chunk_index", 0)
                chunk_id = f"{title}_{chunk_index}"

                if chunk_id in existing_chunk_ids:
                    continue
                existing_chunk_ids.add(chunk_id)

                chunk_content = (_get_field(chunk, "chunk_content") or "").lower()
                title_low = (_get_field(chunk, "title") or "").lower()
                is_exact_match = (query.lower() in chunk_content) or (query.lower() in title_low)

                chunk_results.append({
                    "title": title, "content": _get_field(chunk, "content"),
                    "creation_date": _get_field(chunk, "creation_date"),
                    "modification_date": _get_field(chunk, "modification_date"),
                    "_relevance_score": 50.0 if is_exact_match else 20.0,
                    "_source": "exact_match" if is_exact_match else "partial_match",
                    "_chunk_index": chunk_index,
                    "_total_chunks": _get_field(chunk, "total_chunks"),
                    "_matching_chunk_content": _get_field(chunk, "chunk_content"),
                    "_chunk_id": chunk_id,
                    "cluster_id": _get_field(chunk, "cluster_id"),
                    "cluster_label": _get_field(chunk, "cluster_label"),
                })
        else:
            pass    # no exact matches - fine
    except Exception as e:
        print(f"Database search error: {getattr(e, 'message', repr(e))}")
        traceback.print_exc()
        # Fallback scanning approach
        print("Trying fallback search...")
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
            safe_query = re.escape(query)
            query_regex = re.compile(r"\b" + safe_query + r"\b", flags=re.IGNORECASE)
            matches = []
            for chunk in fallback_results:
                title_text = _get_field(chunk, "title", "") or ""
                content_text = _get_field(chunk, "chunk_content", "") or ""
                if query_regex.search(title_text) or query_regex.search(content_text):
                    matches.append(chunk)
            print(f"Fallback matches: {len(matches)} chunks")
            for chunk in matches:
                title = _get_field(chunk, "title", "<untitled>")
                chunk_index = _get_field(chunk, "chunk_index", 0)
                chunk_id = f"{title}_{chunk_index}"

                if chunk_id in existing_chunk_ids:
                    continue
                existing_chunk_ids.add(chunk_id)

                chunk_results.append({
                    "title": title, "content": _get_field(chunk, "content"),
                    "creation_date": _get_field(chunk, "creation_date"),
                    "modification_date": _get_field(chunk, "modification_date"),
                    "_relevance_score": 0.0,
                    "_source": "fallback_exact",
                    "_chunk_index": chunk_index,
                    "_total_chunks": _get_field(chunk, "total_chunks"),
                    "_matching_chunk_content": _get_field(chunk, "chunk_content"),
                    "_chunk_id": chunk_id,
                    "cluster_id": _get_field(chunk, "cluster_id"),
                    "cluster_label": _get_field(chunk, "cluster_label"),
                })
        except Exception as fallback_error:
            print(f"Fallback also failed: {getattr(fallback_error, 'message', repr(fallback_error))}")
            traceback.print_exc()

    # =========================================================================
    # Combine and rank results (lower _relevance_score = more relevant)
    # =========================================================================
    combined_results = sorted(chunk_results, key=lambda r: r.get("_relevance_score", float('inf')))

    # Count unique notes for summary
    unique_notes = set(c['title'] for c in combined_results)
    print(f"\nFinal results: {len(combined_results)} chunks from {len(unique_notes)} unique notes")

    if combined_results:
        source_breakdown: Dict[str, int] = {}
        for r in combined_results:
            src = r.get("_source", "unknown")
            source_breakdown[src] = source_breakdown.get(src, 0) + 1
        print(f"   Source breakdown: {', '.join(f'{k}:{v}' for k, v in source_breakdown.items())}")

        print(f"\nTop results:")
        for idx, result in enumerate(combined_results[:display_limit]):
            score = result.get("_relevance_score", 0.0)
            source = result.get("_source", "unknown")
            chunk_idx = result.get("_chunk_index", "?")
            total_chunks = result.get("_total_chunks", "?")
            print(f'   {idx + 1}. "{result.get("title")}" (score: {score:.1f}, source: {source}, chunk: {chunk_idx}/{total_chunks})')

    # Map to the final output shape
    final = []
    for result in combined_results:
        final.append({
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
        })

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

    print(f"Connecting to LanceDB at: {DB_PATH}")

    # Initialize database connection
    db = NotesDatabase(db_path=DB_PATH)
    notes_table = db.get_or_create_table()

    # Check if table has data
    try:
        row_count = notes_table.count_rows()
        print(f"Database contains {row_count} chunks")
        if row_count == 0:
            print("No data in database. Run the pipeline first.")
            sys.exit(0)
    except Exception as e:
        print(f"Could not count rows: {e}")

    # Initialize embedding model for query embedding
    embedding_model = EmbeddingModel()

    def compute_query_embedding(q: str):
        """Compute embedding for search query"""
        embeddings = embedding_model.embed_texts([q], show_progress=False)
        return embeddings[0].tolist()

    # Perform search
    print(f"\nSearching for: '{args.query}'")
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
            print(f"   - {label} (ID: {c_id}): {count} notes")

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