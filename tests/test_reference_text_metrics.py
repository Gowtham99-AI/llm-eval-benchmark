
import pytest

from llm_eval_benchmark.metrics import ReferenceTextScorer


def test_identical_text_scores_higher_than_different_text():
    scorer = ReferenceTextScorer()
    reference = "Dependency injection improves software testing."

    identical = scorer.score(reference, reference)
    different = scorer.score(
        reference,
        "Bananas grow well in tropical climates.",
    )

    assert identical["rouge_l"] == pytest.approx(1.0)
    assert identical["bleu"] > different["bleu"]
    assert identical["rouge_l"] > different["rouge_l"]


def test_empty_response_returns_zero_scores():
    scorer = ReferenceTextScorer()

    result = scorer.score(
        "Dependency injection improves testing.",
        "",
    )

    assert result["rouge_l"] == 0.0
    assert result["bleu"] == 0.0


def test_rejects_empty_reference():
    scorer = ReferenceTextScorer()

    with pytest.raises(ValueError, match="non-empty reference"):
        scorer.score("", "Generated response")


def test_rejects_non_string_response():
    scorer = ReferenceTextScorer()

    with pytest.raises(ValueError, match="actual must be a string"):
        scorer.score("Reference text", None)
