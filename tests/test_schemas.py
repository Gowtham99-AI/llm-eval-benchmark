
import pytest
from pydantic import ValidationError

from llm_eval_benchmark.schemas import EvaluationExample


def valid_example() -> dict:
    return {
        "id": "json_001",
        "category": "structured_output",
        "difficulty": "easy",
        "prompt": "Return a JSON object with name set to Ada.",
        "expected": {"name": "Ada"},
        "scoring_method": "json_schema",
        "source": "original",
        "version": "0.1.0",
    }


def test_valid_example_is_accepted():
    example = EvaluationExample(**valid_example())

    assert example.id == "json_001"
    assert example.category == "structured_output"
    assert example.expected == {"name": "Ada"}


def test_invalid_category_is_rejected():
    data = valid_example()
    data["category"] = "unknown"

    with pytest.raises(ValidationError):
        EvaluationExample(**data)


def test_empty_prompt_is_rejected():
    data = valid_example()
    data["prompt"] = ""

    with pytest.raises(ValidationError):
        EvaluationExample(**data)


def test_unexpected_field_is_rejected():
    data = valid_example()
    data["promt"] = data.pop("prompt")

    with pytest.raises(ValidationError):
        EvaluationExample(**data)


def test_missing_id_is_rejected():
    data = valid_example()
    del data["id"]

    with pytest.raises(ValidationError):
        EvaluationExample(**data)
