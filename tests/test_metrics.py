"""Tests for src/metrics.py — run with: pytest tests/test_metrics.py"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src import metrics


def test_rouge_l_identical_strings():
    assert metrics.rouge_l("the cat sat on the mat", "the cat sat on the mat") == 1.0


def test_rouge_l_no_overlap():
    assert metrics.rouge_l("completely different words here", "totally unrelated content indeed") == 0.0


def test_rouge_l_partial_overlap():
    score = metrics.rouge_l("the cat sat on the mat", "a cat sat on a rug")
    assert 0.0 < score < 1.0


def test_rouge_l_empty_strings():
    assert metrics.rouge_l("", "something") == 0.0
    assert metrics.rouge_l("something", "") == 0.0


def test_semantic_similarity_identical():
    score, backend = metrics.semantic_similarity("Paris is the capital of France", "Paris is the capital of France")
    assert score > 0.9
    assert backend in ("bert-score", "tfidf-cosine", "jaccard")


def test_semantic_similarity_empty():
    score, backend = metrics.semantic_similarity("", "something")
    assert score == 0.0
    assert backend == "none"


def test_distinct_n_repetitive_text():
    repetitive = "the the the the the the"
    diverse = "the quick brown fox jumps over"
    assert metrics.distinct_n(repetitive, n=2) < metrics.distinct_n(diverse, n=2)


def test_distinct_n_short_text():
    # fewer tokens than n -> defined as fully diverse (1.0)
    assert metrics.distinct_n("hi", n=2) == 1.0


def test_response_length():
    assert metrics.response_length("one two three four") == 4


def test_score_response_with_reference():
    result = metrics.score_response("Paris", "Paris")
    assert result.rouge_l == 1.0
    assert result.composite > 0.0


def test_score_response_without_reference():
    result = metrics.score_response(None, "some generated text here")
    assert result.rouge_l == 0.0
    assert result.semantic_sim == 0.0
    # composite falls back to diversity-only signal
    assert result.composite == result.distinct_2


def test_metric_human_agreement_perfect_correlation():
    auto = [0.1, 0.5, 0.9]
    human = [0.2, 0.5, 0.95]
    agreement = metrics.metric_human_agreement(auto, human)
    assert agreement["pearson"] > 0.9
    assert agreement["n"] == 3


def test_metric_human_agreement_mismatched_lengths():
    agreement = metrics.metric_human_agreement([0.1, 0.2], [0.1])
    assert agreement["pearson"] is None
    assert agreement["spearman"] is None
