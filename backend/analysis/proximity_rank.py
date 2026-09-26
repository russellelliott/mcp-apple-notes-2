"""
proximity_rank.py – Deterministic post-retrieval proximity ranking for multi-word keyword queries.

Designed to work with both search_notes.py (Python backend) and server/index.ts (TypeScript MCP).

Algorithm overview
==================
1. Normalize query and chunk text (lowercase, Unicode-safe tokenization via regex).
2. Deduplicate repeated query terms for coverage scoring.
3. Find the smallest span in the chunk that contains all distinct query terms
   **in the original query order**.  Reverse-order matches do NOT receive the
   ordered-proximity bonus.
4. Score components:
   a. all_terms_coverage_bonus – awarded when ALL distinct query terms appear somewhere
      in the chunk (regardless of order).
   b. ordered_proximity_bonus – decays with the number of extra tokens between the first
      and last matching term in the best (leftmost-smallest) valid span:
          bonus = ORDERED_MAX / (1 + extra_tokens)
          extra_tokens = span_length - distinct_term_count
   c. exact_phrase_bonus     – awarded when all terms appear consecutively in order
      (zero extra tokens).
5. All bonuses are bounded by a hard cap so they cannot overwhelm semantic signals.

Scoring constants (tunable)
============================
ALL_TERMS_COVERAGE_MAX  = 0.08   – when every distinct term is present
ORDERED_MAX             = 0.12   – decays with span width; max at exact adjacency
EXACT_PHRASE_BONUS      = 0.15   – flat bonus for consecutive ordered terms
TOTAL_BONUS_CAP         = 0.30   – absolute maximum proximity bonus

Quoted queries
==============
If is_quoted is True, the query is treated as explicit phrase intent and always
receives the max exact_phrase_bonus (+0.15), in addition to coverage and ordered
proximity bonuses if applicable.
"""

from __future__ import annotations

import re
from typing import List, Optional, Set, Tuple


# ── Tunable constants ────────────────────────────────────────────────────────

ALL_TERMS_COVERAGE_MAX: float = 0.08
ORDERED_MAX: float = 0.12
EXACT_PHRASE_BONUS: float = 0.15
TOTAL_BONUS_CAP: float = 0.30

# Minimum number of distinct terms needed to apply proximity boosting
# (single-word queries get no proximity bonus regardless)
MIN_TERMS_FOR_PROXIMITY: int = 2


# ── Public API ───────────────────────────────────────────────────────────────

def tokenize(text: str) -> List[str]:
    """Unicode-safe tokenization: extract sequences of alphanumeric characters."""
    return re.findall(r"[a-z0-9]+", text.lower())


def compute_proximity_score(
    chunk_text: str,
    query: str,
    is_quoted: bool = False,
) -> float:
    """Compute a bounded proximity bonus for a single (chunk, query) pair.

    Parameters
    ----------
    chunk_text : str
        The text content of the chunk (may be full chunk_content + title).
    query : str
        The raw user search query (with or without surrounding quotes).
    is_quoted : bool
        If True, treat as explicit phrase intent and always award max
        exact_phrase_bonus.

    Returns
    -------
    float
        A value in [0, TOTAL_BONUS_CAP].  Returns 0.0 when:
        - query has fewer than 2 distinct meaningful terms,
        - query is empty / whitespace only.
    """
    # Normalize
    clean_query = _extract_meaningful_terms(query)
    if len(clean_query) < MIN_TERMS_FOR_PROXIMITY:
        return 0.0

    tokens = tokenize(chunk_text)
    query_set: Set[str] = set(clean_query)

    # ── Coverage check ───────────────────────────────────────────────────
    all_present = all(t in tokens for t in query_set)
    if not all_present:
        coverage_bonus = 0.0
    else:
        coverage_bonus = ALL_TERMS_COVERAGE_MAX

    # ── Ordered proximity search ─────────────────────────────────────────
    ordered_bonus, is_phrase = _find_best_ordered_span(tokens, clean_query)

    # Quoted queries always get the phrase bonus (explicit user intent).
    if is_quoted and not is_phrase:
        phrase_bonus = EXACT_PHRASE_BONUS
    else:
        phrase_bonus = EXACT_PHRASE_BONUS if is_phrase else 0.0

    total = min(coverage_bonus + ordered_bonus + phrase_bonus, TOTAL_BONUS_CAP)
    return round(total, 6)


def compute_proximity_scores(
    chunk_texts: List[str],
    query: str,
    is_quoted: bool = False,
) -> List[float]:
    """Vectorized wrapper: compute proximity score for multiple chunks."""
    return [
        compute_proximity_score(text, query, is_quoted=is_quoted)
        for text in chunk_texts
    ]


# ── Internal helpers ─────────────────────────────────────────────────────────

def _extract_meaningful_terms(query: str) -> List[str]:
    """Extract meaningful (non-stopword, >1 char) deduplicated terms."""
    from backend.analysis.search_notes import COMMON_STOP_WORDS

    raw_terms = re.findall(r"[a-z0-9]+", query.lower())
    terms = [t for t in raw_terms if t not in COMMON_STOP_WORDS and len(t) > 1]

    seen: Set[str] = set()
    unique: List[str] = []
    for t in terms:
        if t not in seen:
            seen.add(t)
            unique.append(t)
    return unique


def _find_best_ordered_span(
    tokens: List[str],
    query_terms: List[str],
) -> Tuple[float, bool]:
    """Find the smallest span containing all distinct query terms in order.

    Returns
    -------
    (bonus, is_exact_phrase)
        bonus: the ordered_proximity_bonus value (0.0 if no valid span found).
        is_exact_phrase: True if the best span has zero extra tokens (adjacent).
    """
    distinct_terms = list(dict.fromkeys(query_terms))  # preserve order, deduplicate
    n = len(distinct_terms)

    best_span_length: Optional[int] = None

    # Greedy multi-pass: find all valid ordered spans, track the smallest.
    # For efficiency, use a single-pass algorithm that tracks positions of each
    # query term in order.
    for start_pos in range(len(tokens)):
        if tokens[start_pos] != distinct_terms[0]:
            continue

        # Try to match remaining terms in order
        pos = start_pos
        matched = 1
        for i in range(1, n):
            # Search forward from pos+1 for distinct_terms[i]
            found = False
            for j in range(pos + 1, len(tokens)):
                if tokens[j] == distinct_terms[i]:
                    pos = j
                    matched += 1
                    found = True
                    break
            if not found:
                break

        if matched == n:
            span_length = pos - start_pos + 1
            if best_span_length is None or span_length < best_span_length:
                best_span_length = span_length
                # Early exit if we found exact adjacency
                if span_length == n:
                    return (ORDERED_MAX, True)

    if best_span_length is None:
        return (0.0, False)

    extra_tokens = best_span_length - n
    ordered_bonus = ORDERED_MAX / (1 + extra_tokens)
    is_phrase = (extra_tokens == 0)
    return (round(ordered_bonus, 6), is_phrase)


# ── Convenience / CLI ────────────────────────────────────────────────────────

if __name__ == "__main__":
    import json
    import sys

    demo_cases = [
        ("docker backup", "docker backup", False),
        ("docker backup", "docker postgres backup", False),
        ("docker backup", "backup docker", False),
        ("docker backup", "docker networking", False),
        ("Docker Backup", "docker—postgres, backup", False),
        ("docker postgres backup", "docker postgres backup", False),
    ]

    for query, chunk, quoted in demo_cases:
        score = compute_proximity_score(chunk, query, is_quoted=quoted)
        print(f"Query: {query!r}")
        print(f"  Chunk: {chunk!r}")
        print(f"  Proximity bonus: {score:.4f}")
        print()