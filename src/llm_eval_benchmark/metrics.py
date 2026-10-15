
import json
from typing import Any


def matches_expected_types(
    expected: Any,
    actual: Any,
) -> bool:
    """Check whether actual values have the expected JSON types."""

    # bool is a subclass of int in Python, so use exact type checks.
    if type(expected) is bool:
        return type(actual) is bool

    if type(expected) is int:
        return type(actual) is int

    if type(expected) is float:
        return type(actual) in (int, float) and type(actual) is not bool

    if isinstance(expected, str):
        return isinstance(actual, str)

    if isinstance(expected, list):
        return (
            isinstance(actual, list)
            and len(expected) == len(actual)
            and all(
                matches_expected_types(e, a)
                for e, a in zip(expected, actual)
            )
        )

    if isinstance(expected, dict):
        return (
            isinstance(actual, dict)
            and expected.keys() == actual.keys()
            and all(
                matches_expected_types(expected[key], actual[key])
                for key in expected
            )
        )

    if expected is None:
        return actual is None

    return False


class JsonSchemaScorer:
    """Score a model response against an expected JSON value."""
    supported_methods = {"json_schema"}

    def score(
        self,
        expected: dict[str, Any],
        actual: str,
    ) -> dict[str, float]:
        try:
            parsed = json.loads(actual)
        except (json.JSONDecodeError, TypeError):
            return {
                "json_valid": 0.0,
                "schema_valid": 0.0,
                "exact_match": 0.0,
            }

        schema_valid = matches_expected_types(expected, parsed)
        exact_match = schema_valid and parsed == expected

        return {
            "json_valid": 1.0,
            "schema_valid": float(schema_valid),
            "exact_match": float(exact_match),
        }

class KeywordMatchScorer:
    """Measure the proportion of required keywords found in a response."""

    supported_methods = {"keyword_match"}

    def score(
        self,
        expected: dict,
        actual: str,
    ) -> dict[str, float]:
        keywords = expected.get("required_keywords", [])

        if not isinstance(keywords, list) or not keywords:
            raise ValueError(
                "expected must contain a non-empty "
                "'required_keywords' list"
            )

        normalized_response = actual.casefold()

        matched = sum(
            1
            for keyword in keywords
            if isinstance(keyword, str)
            and keyword.casefold() in normalized_response
        )

        return {
            "keyword_recall": matched / len(keywords),
        }


class ContainsAllScorer:
    """Check whether all required phrases appear in a response."""

    supported_methods = {"contains_all"}

    def score(
        self,
        expected: dict,
        actual: str,
    ) -> dict[str, float]:
        phrases = expected.get("required_phrases", [])

        if not isinstance(phrases, list) or not phrases:
            raise ValueError(
                "expected must contain a non-empty "
                "'required_phrases' list"
            )

        normalized_response = " ".join(actual.casefold().split())

        matched = sum(
            1
            for phrase in phrases
            if isinstance(phrase, str)
            and " ".join(phrase.casefold().split())
            in normalized_response
        )

        return {
            "instruction_compliance": (
                1.0 if matched == len(phrases) else 0.0
            ),
        }