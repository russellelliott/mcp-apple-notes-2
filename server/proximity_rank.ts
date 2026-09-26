/**
 * proximity_rank.ts – Deterministic post-retrieval proximity ranking for
 * multi-word keyword queries (TypeScript port of backend/analysis/proximity_rank.py).
 *
 * Algorithm overview
 * ==================
 * 1. Normalize query and chunk text (lowercase, Unicode-safe tokenization via regex).
 * 2. Deduplicate repeated query terms for coverage scoring.
 * 3. Find the smallest span in the chunk that contains all distinct query terms
 *    **in the original query order**. Reverse-order matches do NOT receive the
 *    ordered-proximity bonus.
 * 4. Score components:
 *    a. allTermsCoverageBonus – awarded when ALL distinct query terms appear somewhere
 *       in the chunk (regardless of order).
 *    b. orderedProximityBonus – decays with the number of extra tokens between the first
 *       and last matching term in the best (leftmost-smallest) valid span:
 *          bonus = ORDERED_MAX / (1 + extra_tokens)
 *          extra_tokens = span_length - distinct_term_count
 *    c. exactPhraseBonus      – awarded when all terms appear consecutively in order
 *       (zero extra tokens).
 * 5. All bonuses are bounded by a hard cap so they cannot overwhelm semantic signals.
 *
 * Scoring constants (tunable)
 * ===========================
 * ALL_TERMS_COVERAGE_MAX = 0.08    – when every distinct term is present
 * ORDERED_MAX            = 0.12    – decays with span width; max at exact adjacency
 * EXACT_PHRASE_BONUS     = 0.15    – flat bonus for consecutive ordered terms
 * TOTAL_BONUS_CAP        = 0.30    – absolute maximum proximity bonus
 *
 * Quoted queries
 * ==============
 * If isQuoted is true, the query is treated as explicit phrase intent and always
 * receives the max exactPhraseBonus (+0.15), in addition to coverage and ordered
 * proximity bonuses if applicable.
 */

// ── Tunable constants ────────────────────────────────────────────────────────

const ALL_TERMS_COVERAGE_MAX = 0.08;
const ORDERED_MAX = 0.12;
const EXACT_PHRASE_BONUS = 0.15;
const TOTAL_BONUS_CAP = 0.30;

// Minimum number of distinct terms needed to apply proximity boosting
// (single-word queries get no proximity bonus regardless)
const MIN_TERMS_FOR_PROXIMITY = 2;

// Common English stop words - same set as Python implementation
const COMMON_STOP_WORDS: Readonly<Set<string>> = new Set([
     // articles, prepositions, pronouns, auxiliary verbs, conjunctions
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
]);

// ── Public API ───────────────────────────────────────────────────────────────

/**
 * Unicode-safe tokenization: extract sequences of alphanumeric characters.
 */
export function tokenize(text: string): string[] {
    return text.toLowerCase().match(/[a-z0-9]+/g) ?? [];
}

/**
 * Compute a bounded proximity bonus for a single (chunk, query) pair.
 *
 * @param chunkText - The text content of the chunk (may be full chunk_content + title).
 * @param query - The raw user search query (with or without surrounding quotes).
 * @param isQuoted - If true, treat as explicit phrase intent and always award max
 *                   exactPhraseBonus.
 * @returns A value in [0, TOTAL_BONUS_CAP]. Returns 0.0 when:
 *          - query has fewer than 2 distinct meaningful terms,
 *          - query is empty / whitespace only.
 */
export function computeProximityScore(
    chunkText: string,
    query: string,
    isQuoted = false,
): number {
    // Normalize and extract meaningful terms (deduplicated)
    const cleanQuery = extractMeaningfulTerms(query);
    if (cleanQuery.length < MIN_TERMS_FOR_PROXIMITY) {
        return 0.0;
    }

    const tokens = tokenize(chunkText);
    const querySet = new Set(cleanQuery);

    // ── Coverage check ────────────────────────────────────────────────
    const allPresent = cleanQuery.every(t => tokens.includes(t));
    const coverageBonus = allPresent ? ALL_TERMS_COVERAGE_MAX : 0.0;

    // ── Ordered proximity search ──────────────────────────────────────
    const [orderedBonus, isPhrase] = findBestOrderedSpan(tokens, cleanQuery);

    // Quoted queries always get the phrase bonus (explicit user intent).
    const phraseBonus = isQuoted || isPhrase ? EXACT_PHRASE_BONUS : 0.0;

    const total = Math.min(coverageBonus + orderedBonus + phraseBonus, TOTAL_BONUS_CAP);
    return Math.round(total * 1_000_000) / 1_000_000;
}

/**
 * Compute proximity scores for multiple chunks.
 */
export function computeProximityScores(
    chunkTexts: string[],
    query: string,
    isQuoted = false,
): number[] {
    return chunkTexts.map(c => computeProximityScore(c, query, isQuoted));
}

// ── Internal helpers ─────────────────────────────────────────────────────────

/**
 * Extract meaningful (non-stopword, >1 char) deduplicated terms from a query.
 */
function extractMeaningfulTerms(query: string): string[] {
    const rawTerms = query.toLowerCase().match(/[a-z0-9]+/g) ?? [];
    const seen = new Set<string>();
    const unique: string[] = [];

    for (const t of rawTerms) {
        if (!COMMON_STOP_WORDS.has(t) && t.length > 1 && !seen.has(t)) {
            seen.add(t);
            unique.push(t);
        }
    }
    return unique;
}

/**
 * Find the smallest span containing all distinct query terms in order.
 *
 * @returns [bonus, isExactPhrase]
 *   - bonus: the orderedProximityBonus value (0.0 if no valid span found).
 *   - isExactPhrase: true if the best span has zero extra tokens (adjacent).
 */
function findBestOrderedSpan(
    tokens: string[],
    queryTerms: string[],
): [number, boolean] {
    // Deduplicate while preserving order
    const distinctTerms = Array.from(new Set(queryTerms));
    const n = distinctTerms.length;

    let bestSpanLength: number | null = null;

    // Multi-pass: try each occurrence of the first query term as start,
    // then greedily match remaining terms in order. Track the smallest span.
    for (let startPos = 0; startPos < tokens.length; startPos++) {
        if (tokens[startPos] !== distinctTerms[0]) {
            continue;
        }

        // Try to match remaining terms in order
        let pos = startPos;
        let matched = 1;

        for (let i = 1; i < n; i++) {
            // Search forward from pos+1 for distinctTerms[i]
            let found = false;
            for (let j = pos + 1; j < tokens.length; j++) {
                if (tokens[j] === distinctTerms[i]) {
                    pos = j;
                    matched++;
                    found = true;
                    break;
                }
            }
            if (!found) {
                break;
            }
        }

        if (matched === n) {
            const spanLength = pos - startPos + 1;
            if (bestSpanLength === null || spanLength < bestSpanLength) {
                bestSpanLength = spanLength;
                // Early exit if we found exact adjacency
                if (spanLength === n) {
                    return [ORDERED_MAX, true];
                }
            }
        }
    }

    if (bestSpanLength === null) {
        return [0.0, false];
    }

    const extraTokens = bestSpanLength - n;
    const orderedBonus = ORDERED_MAX / (1 + extraTokens);
    const isPhrase = extraTokens === 0;
    return [Math.round(orderedBonus * 1_000_000) / 1_000_000, isPhrase];
}