
import pytest

from llm_eval_benchmark.metrics import (
    ContainsAllScorer,
    KeywordMatchScorer,
)


def test_keyword_match_scorer_returns_full_score():
    scorer = KeywordMatchScorer()

    result = scorer.score(
        {"required_keywords": ["dependency injection", "FastAPI"]},
        "FastAPI supports dependency injection.",
    )

    assert result["keyword_recall"] == 1.0


def test_keyword_match_scorer_returns_partial_score():
    scorer = KeywordMatchScorer()

    result = scorer.score(
        {"required_keywords": ["Python", "FastAPI", "PostgreSQL"]},
        "Python and FastAPI are used here.",
    )

    assert result["keyword_recall"] == pytest.approx(2 / 3)


def test_keyword_match_scorer_rejects_empty_keywords():
    with pytest.raises(ValueError, match="required_keywords"):
        KeywordMatchScorer().score(
            {"required_keywords": []},
            "Some response",
        )


def test_contains_all_scorer_returns_full_score():
    scorer = ContainsAllScorer()

    result = scorer.score(
        {"required_phrases": ["valid JSON", "error handling"]},
        "The output contains valid JSON and error handling.",
    )

    assert result["instruction_compliance"] == 1.0


def test_contains_all_scorer_returns_zero_when_phrase_missing():
    scorer = ContainsAllScorer()

    result = scorer.score(
        {"required_phrases": ["valid JSON", "error handling"]},
        "The output contains valid JSON.",
    )

    assert result["instruction_compliance"] == 0.0


def test_contains_all_scorer_rejects_empty_phrases():
    with pytest.raises(ValueError, match="required_phrases"):
        ContainsAllScorer().score(
            {"required_phrases": []},
            "Some response",
        )
