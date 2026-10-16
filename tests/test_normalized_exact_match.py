
import pytest

from llm_eval_benchmark.metrics import NormalizedExactMatchScorer


@pytest.fixture
def scorer() -> NormalizedExactMatchScorer:
    return NormalizedExactMatchScorer()


def test_matches_case_and_whitespace_differences(scorer):
    result = scorer.score(
        expected="  Hello   WORLD ",
        actual="hello world",
    )

    assert result == {"normalized_exact_match": 1.0}


def test_returns_zero_for_different_text(scorer):
    result = scorer.score(
        expected="Hello world",
        actual="Hello there",
    )

    assert result == {"normalized_exact_match": 0.0}


def test_punctuation_difference_does_not_match(scorer):
    result = scorer.score(
        expected="Hello, world!",
        actual="Hello world",
    )

    assert result == {"normalized_exact_match": 0.0}


def test_rejects_non_string_expected(scorer):
    with pytest.raises(ValueError, match="expected must be a string"):
        scorer.score(expected={"answer": "hello"}, actual="hello")


def test_empty_strings_match(scorer):
    result = scorer.score(expected="", actual="")

    assert result == {"normalized_exact_match": 1.0}
