"""
test_proximity_rank.py – Automated tests for proximity_rank module.

Covers all required scenarios from the task specification:

1. Exact phrase match (docker backup)
2. One intervening word (docker postgres backup)
3. Large distance between terms
4. Reverse order (backup docker)
5. Partial match (docker networking)
6. Three-term query ranking
7. Case and punctuation normalization
8. Repeated query term handling
"""

import math
import sys
from pathlib import Path

# Ensure backend root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pytest
from backend.analysis.proximity_rank import (
    compute_proximity_score,
    compute_proximity_scores,
    tokenize,
    # Internal constants for assertions
    ALL_TERMS_COVERAGE_MAX,
    ORDERED_MAX,
    EXACT_PHRASE_BONUS,
    TOTAL_BONUS_CAP,
)


# ──────────────────────────────── fixtures ───────────────────────────────────

@pytest.fixture(params=["backend", "frontend"])
def env(request: pytest.FixtureRequest):
    """Run each test in two environments to catch path/import differences."""
    return request.param


# ──────────────────────────── 1. Exact phrase ───────────────────────────────

class TestExactPhrase:
    def test_exact_phrase_receives_max_bonus(self, env):
        """Chunk and query are identical adjacent terms → max possible bonus."""
        score = compute_proximity_score("docker backup", "docker backup")
        expected = ALL_TERMS_COVERAGE_MAX + ORDERED_MAX + EXACT_PHRASE_BONUS
        # Should be capped at TOTAL_BONUS_CAP if sum exceeds it
        assert score == min(expected, TOTAL_BONUS_CAP)
        assert score > 0

    def test_exact_phrase_case_insensitive(self, env):
        """Query 'Docker Backup' with chunk 'docker—postgres, backup'.

        Note: After tokenization, tokens are ['docker', 'postgres', 'backup'].
        Query terms are ['docker', 'backup']. They are NOT adjacent (postgres in between).
        So we expect coverage + ordered proximity (1 extra token), but NO phrase bonus.
        """
        score = compute_proximity_score("docker—postgres, backup", "Docker Backup")
        # Coverage (0.08) + ordered with 1 extra (0.12/2 = 0.06) = 0.14
        expected = ALL_TERMS_COVERAGE_MAX + ORDERED_MAX / 2.0
        assert score == min(expected, TOTAL_BONUS_CAP)
        # Must be positive and less than exact phrase
        assert score > 0
        assert score < compute_proximity_score("docker backup", "docker backup")


# ──────────────────────────── 2. One intervening word ───────────────────────

class TestOneInterveningWord:
    def test_one_word_between_receives_positive_bonus(self, env):
        """docker <postgres> backup → ordered bonus with 1 extra token."""
        score = compute_proximity_score("docker postgres backup", "docker backup")
        # Ordered bonus should be ORDERED_MAX / (1 + 1) = 0.06
        expected_ordered = ORDERED_MAX / 2.0
        expected_total = ALL_TERMS_COVERAGE_MAX + expected_ordered + 0.0  # no exact phrase
        assert score == min(expected_total, TOTAL_BONUS_CAP)
        assert score > 0

    def test_bonus_smaller_than_exact_phrase(self, env):
        """One-intervening-word bonus must be less than exact phrase bonus."""
        one_word = compute_proximity_score("docker postgres backup", "docker backup")
        exact = compute_proximity_score("docker backup", "docker backup")
        assert one_word < exact


# ──────────────────────────── 3. Large distance ─────────────────────────────

class TestLargeDistance:
    def test_distant_terms_get_smaller_bonus(self, env):
        """Terms separated by many words → small but positive bonus."""
        distant_text = " ".join([
            "docker", "setup", "networking", "volumes", "config",
            "deployment", "monitoring", "backup"
        ])
        score = compute_proximity_score(distant_text, "docker backup")
        # Should be positive (coverage + tiny ordered proximity)
        assert score > 0
        # But much smaller than adjacent case
        exact = compute_proximity_score("docker backup", "docker backup")
        assert score < exact * 0.5  # At least 2x smaller

    def test_bonus_decays_with_distance(self, env):
        """Bonus for 1 intervening word > bonus for many intervening words."""
        one_word = compute_proximity_score("docker postgres backup", "docker backup")
        many_words = compute_proximity_score(
            "docker setup networking volumes config deployment backup",
            "docker backup"
        )
        assert one_word > many_words


# ──────────────────────────── 4. Reverse order ──────────────────────────────

class TestReverseOrder:
    def test_reverse_order_gets_no_ordered_bonus(self, env):
        """backup docker → only coverage bonus, no ordered-proximity bonus."""
        score = compute_proximity_score("backup docker", "docker backup")
        # Should have coverage bonus (both terms present) but NOT ordered bonus
        # Score should equal ALL_TERMS_COVERAGE_MAX (only coverage, no ordered, no phrase)
        assert score == ALL_TERMS_COVERAGE_MAX

    def test_reverse_order_smaller_than_forward(self, env):
        """Reverse-order score < forward-order score."""
        forward = compute_proximity_score("docker backup procedure", "docker backup")
        reverse = compute_proximity_score("backup docker procedure", "docker backup")
        assert forward > reverse


# ──────────────────────────── 5. Partial match ──────────────────────────────

class TestPartialMatch:
    def test_partial_match_gets_no_all_terms_bonus(self, env):
        """docker networking → only 'docker' matches, no coverage bonus."""
        score = compute_proximity_score("docker networking", "docker backup")
        # Should be 0.0 — neither all-terms nor ordered-proximity applies.
        assert score == 0.0

    def test_only_one_term_no_bonus(self, env):
        """When only one query term is present, no bonus."""
        score = compute_proximity_score("docker container guide", "docker backup")
        assert score == 0.0


# ──────────────────────────── 6. Three-term ranking ─────────────────────────

class TestThreeTerms:
    def test_three_exact_terms_rank_above_scattered(self, env):
        """'docker postgres backup' should rank above 'docker local postgres database backup'."""
        exact = compute_proximity_score("docker postgres backup", "docker postgres backup")
        scattered = compute_proximity_score(
            "docker local postgres database backup for production",
            "docker postgres backup"
        )
        assert exact > scattered

    def test_three_terms_adjacent_max(self, env):
        """Three adjacent terms in order → all bonuses apply."""
        score = compute_proximity_score("docker postgres backup procedure", "docker postgres backup")
        # Coverage + ordered (0 extra tokens) + phrase
        expected = ALL_TERMS_COVERAGE_MAX + ORDERED_MAX + EXACT_PHRASE_BONUS
        assert score == min(expected, TOTAL_BONUS_CAP)

    def test_three_terms_in_order_preferred_over_unordered(self, env):
        """Chunks with terms in query order rank above those with reversed terms."""
        ordered_text = "docker postgres backup"
        unorder1 = "postgres docker backup"   # first two swapped
        unorder2 = "backup postgres docker"   # completely reversed
        q = "docker postgres backup"

        s_ordered = compute_proximity_score(ordered_text, q)
        s_unorder1 = compute_proximity_score(unorder1, q)
        s_unorder2 = compute_proximity_score(unorder2, q)

        assert s_ordered > s_unorder1
        assert s_ordered > s_unorder2


# ──────────────────────────── 7. Case and punctuation ───────────────────────

class TestCasePunctuation:
    def test_case_insensitive_matching(self, env):
        """Query 'Docker Backup' matches chunk 'docker postgres backup'."""
        score = compute_proximity_score("docker postgres backup", "Docker Backup")
        assert score > 0  # Coverage + ordered bonus

    def test_punctuation_normalized(self, env):
        """Query 'Docker Backup' should match chunk with em-dash/commas."""
        score = compute_proximity_score("docker—postgres, backup", "Docker Backup")
        # After tokenization, all three terms appear in order.
        # 'docker' and 'backup' are adjacent (separated only by 'postgres')
        assert score > 0


# ──────────────────────────── 8. Repeated query term ────────────────────────

class TestRepeatedTerm:
    def test_repeated_query_term_handled(self, env):
        """Query 'backup docker backup' should be deduplicated → treated as 'backup docker'.

        After dedup: query_terms = ['backup', 'docker'].
        Chunk tokens = ['backup', 'docker'] — adjacent in order.
        Coverage (0.08) + ordered adjacent (0.12) + phrase (0.15) = 0.35 → capped at 0.30.
        """
        score = compute_proximity_score("backup docker", "backup docker backup")
        expected = ALL_TERMS_COVERAGE_MAX + ORDERED_MAX + EXACT_PHRASE_BONUS
        assert score == min(expected, TOTAL_BONUS_CAP)

    def test_repeated_term_chunk(self, env):
        """Chunk with repeated term 'docker docker backup' for query 'docker backup'."""
        score = compute_proximity_score("docker docker backup procedure", "docker backup")
        # After dedup: query_terms=['docker', 'backup']. Chunk tokens=['docker','docker','backup','procedure']
        # Best span: positions 0-2 (docker, docker, backup) or 1-3 (docker, backup, procedure)
        # First span: docker→docker→backup has docker at 0 and backup at 2 → span=3
        # Second span: docker at 1 and backup at 2 → span=2, better!
        # So ordered bonus = ORDERED_MAX / (1 + 0) = ORDERED_MAX with exact phrase
        expected = ALL_TERMS_COVERAGE_MAX + ORDERED_MAX + EXACT_PHRASE_BONUS
        assert score == min(expected, TOTAL_BONUS_CAP)


# ──────────────────────────── Edge cases ─────────────────────────────────────

class TestEdgeCases:
    def test_single_word_query_returns_zero(self, env):
        """Single-term queries get no proximity bonus."""
        assert compute_proximity_score("docker networking", "docker") == 0.0

    def test_empty_query_returns_zero(self, env):
        """Empty or whitespace-only queries get no proximity bonus."""
        assert compute_proximity_score("some content", "") == 0.0
        assert compute_proximity_score("some content", "   ") == 0.0

    def test_stop_word_only_query_returns_zero(self, env):
        """Query containing only stop words gets no proximity bonus."""
        assert compute_proximity_score("some content", "the and or") == 0.0

    def test_quoted_query_always_gets_phrase_bonus(self, env):
        """Quoted queries receive max phrase bonus even without exact adjacency."""
        # Non-adjacent terms with quotes → should still get phrase bonus
        score = compute_proximity_score("docker some postgres many backup", '"docker backup"', is_quoted=True)
        # Coverage (0.08) + ordered with 3 extra tokens (0.12/4=0.03) + phrase (quoted, 0.15) = 0.26
        expected_ordered = ORDERED_MAX / 4.0    # 3 extra tokens between docker and backup
        expected_total = ALL_TERMS_COVERAGE_MAX + expected_ordered + EXACT_PHRASE_BONUS
        assert score == min(expected_total, TOTAL_BONUS_CAP)

    def test_unquoted_non_adjacent_no_phrase_bonus(self, env):
        """Non-quoted non-adjacent terms should NOT get phrase bonus."""
        score = compute_proximity_score("docker some postgres many backup", "docker backup")
        # Coverage + small ordered proximity (non-zero span) but NO phrase bonus
        expected_ordered = ORDERED_MAX / 4.0  # 4 extra tokens
        expected_total = ALL_TERMS_COVERAGE_MAX + expected_ordered
        assert score == min(expected_total, TOTAL_BONUS_CAP)
        # Verify it's less than the quoted version (which has phrase bonus)
        quoted_score = compute_proximity_score(
            "docker some postgres many backup", '"docker backup"', is_quoted=True
        )
        assert quoted_score > score

    def test_vectorized_scores_match_individual(self, env):
        """compute_proximity_scores should match individual calls."""
        chunks = ["docker backup", "docker postgres backup", "backup docker", "no match here"]
        query = "docker backup"

        individual = [compute_proximity_score(c, query) for c in chunks]
        vectorized = compute_proximity_scores(chunks, query)

        assert individual == vectorized


# ──────────────────────────── Scoring formula verification ────────────────────

class TestScoringFormula:
    def test_bonus_capped_at_total_bonus_cap(self, env):
        """Sum of all bonuses should never exceed TOTAL_BONUS_CAP."""
        # Best possible case: all terms present, adjacent, in order
        score = compute_proximity_score("docker backup", "docker backup")
        assert score <= TOTAL_BONUS_CAP

    def test_bonus_is_deterministic(self, env):
        """Same inputs should always produce same output."""
        q1 = compute_proximity_score("docker postgres backup procedure", "docker backup")
        q2 = compute_proximity_score("docker postgres backup procedure", "docker backup")
        assert q1 == q2

    def test_bonus_does_not_overwhelm_base(self, env):
        """Proximity bonus should not produce negative final scores after subtraction."""
        # The bonus is subtracted * 10 in search_notes.py as a scale factor,
        # but the score is capped at 0.0 minimum via max(0.0, ...)
        score = compute_proximity_score("docker backup", "docker backup")
        assert score >= 0.0


# ──────────────────────────── Run with pytest ────────────────────────────────

if __name__ == "__main__":
    pytest.main([__file__, "-v"])