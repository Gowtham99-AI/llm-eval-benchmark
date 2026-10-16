
from typing import Any


def validate_scoring_compatibility(
    scoring_method: str,
    expected: Any,
) -> list[str]:
    """Return validation errors for an incompatible expected value."""
    errors: list[str] = []

    if scoring_method == "json_schema":
        if not isinstance(expected, dict):
            errors.append(
                "json_schema requires expected to be a dictionary"
            )

    elif scoring_method == "keyword_match":
        if not isinstance(expected, dict):
            errors.append(
                "keyword_match requires expected to be a dictionary"
            )
        elif not isinstance(expected.get("required_keywords"), list):
            errors.append(
                "keyword_match requires a required_keywords list"
            )
        elif not expected["required_keywords"]:
            errors.append("required_keywords must not be empty")

    elif scoring_method == "contains_all":
        if not isinstance(expected, dict):
            errors.append(
                "contains_all requires expected to be a dictionary"
            )
        elif not isinstance(expected.get("required_phrases"), list):
            errors.append(
                "contains_all requires a required_phrases list"
            )
        elif not expected["required_phrases"]:
            errors.append("required_phrases must not be empty")

    elif scoring_method in {
        "normalized_exact_match",
        "reference_text",
        "semantic_similarity",
    }:
        if not isinstance(expected, str) or not expected.strip():
            errors.append(
                f"{scoring_method} requires a non-empty string"
            )

    elif scoring_method == "output_constraints":
        if not isinstance(expected, dict):
            errors.append(
                "output_constraints requires a dictionary"
            )

    elif scoring_method == "llm_judge":
        if not isinstance(expected, dict):
            errors.append("llm_judge requires a dictionary")

    else:
        errors.append(f"Unsupported scoring method: {scoring_method}")

    return errors
