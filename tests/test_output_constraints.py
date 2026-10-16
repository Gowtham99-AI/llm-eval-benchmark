
import pytest

from llm_eval_benchmark.metrics import OutputConstraintScorer


def test_all_constraints_pass():
    scorer = OutputConstraintScorer()

    result = scorer.score(
        {
            "min_words": 4,
            "max_words": 10,
            "max_characters": 60,
            "require_non_empty": True,
        },
        "Dependency injection provides dependencies externally.",
    )

    assert result["output_constraints_pass_rate"] == 1.0
    assert result["output_constraints_passed"] == 1.0


def test_partial_constraint_failure():
    scorer = OutputConstraintScorer()

    result = scorer.score(
        {"max_words": 2, "require_non_empty": True},
        "This answer contains too many words.",
    )

    assert result["output_constraints_pass_rate"] == 0.5
    assert result["output_constraints_passed"] == 0.0


def test_rejects_invalid_word_range():
    scorer = OutputConstraintScorer()

    with pytest.raises(ValueError, match="cannot exceed"):
        scorer.score(
            {"min_words": 10, "max_words": 2},
            "Example response",
        )


def test_rejects_missing_constraints():
    scorer = OutputConstraintScorer()

    with pytest.raises(ValueError, match="At least one"):
        scorer.score({}, "Example response")


def test_empty_response_fails_non_empty_constraint():
    scorer = OutputConstraintScorer()

    result = scorer.score(
        {"require_non_empty": True},
        "   ",
    )

    assert result["output_constraints_passed"] == 0.0
