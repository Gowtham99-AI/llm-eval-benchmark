
import pytest

from llm_eval_benchmark.metrics import JsonSchemaScorer


@pytest.fixture
def scorer():
    return JsonSchemaScorer()


@pytest.fixture
def expected():
    return {"name": "Ada", "active": True}


def test_correct_json_scores_one(scorer, expected):
    actual = '{"name": "Ada", "active": true}'

    result = scorer.score(expected, actual)

    assert result == {
        "json_valid": 1.0,
        "schema_valid": 1.0,
        "exact_match": 1.0,
    }


def test_invalid_json_scores_zero(scorer, expected):
    result = scorer.score(expected, '{"name": "Ada",}')

    assert result == {
        "json_valid": 0.0,
        "schema_valid": 0.0,
        "exact_match": 0.0,
    }


def test_wrong_value_fails_exact_match(scorer, expected):
    actual = '{"name": "Grace", "active": true}'

    result = scorer.score(expected, actual)

    assert result["json_valid"] == 1.0
    assert result["schema_valid"] == 1.0
    assert result["exact_match"] == 0.0


def test_wrong_type_fails_schema(scorer, expected):
    actual = '{"name": "Ada", "active": 1}'

    result = scorer.score(expected, actual)

    assert result["json_valid"] == 1.0
    assert result["schema_valid"] == 0.0
    assert result["exact_match"] == 0.0


def test_missing_field_fails_schema(scorer, expected):
    actual = '{"name": "Ada"}'

    result = scorer.score(expected, actual)

    assert result["schema_valid"] == 0.0
    assert result["exact_match"] == 0.0


def test_extra_field_fails_schema(scorer, expected):
    actual = '{"name": "Ada", "active": true, "age": 30}'

    result = scorer.score(expected, actual)

    assert result["schema_valid"] == 0.0
    assert result["exact_match"] == 0.0


def test_boolean_is_not_integer(scorer):
    result = scorer.score({"count": 1}, '{"count": true}')

    assert result["schema_valid"] == 0.0
    assert result["exact_match"] == 0.0


def test_non_string_response_is_handled(scorer, expected):
    result = scorer.score(expected, None)

    assert result["json_valid"] == 0.0
