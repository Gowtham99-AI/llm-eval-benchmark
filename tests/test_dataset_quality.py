
import pytest

from llm_eval_benchmark.dataset_loader import load_examples
from llm_eval_benchmark.dataset_quality import (
    validate_scoring_compatibility,
)

DATASET_PATH = "data/examples.jsonl"

EXPECTED_CATEGORIES = {
    "structured_output",
    "software_qa",
    "code_understanding",
    "instruction_following",
}

@pytest.mark.parametrize(
    ("method", "expected"),
    [
        ("json_schema", {"name": "Ada"}),
        ("keyword_match", {"required_keywords": ["Python"]}),
        ("contains_all", {"required_phrases": ["unit test"]}),
        ("normalized_exact_match", "Expected output"),
        ("reference_text", "Reference answer"),
        ("semantic_similarity", "Expected meaning"),
        ("output_constraints", {"min_words": 5}),
        ("llm_judge", {"reference_answer": "Expected answer"}),
    ],
)

def test_valid_scoring_combinations(method, expected):
    assert validate_scoring_compatibility(method, expected) == []

def test_dataset_has_unique_example_ids():
    examples = load_examples(DATASET_PATH)
    ids = [example.id for example in examples]

    assert len(ids) == len(set(ids))


def test_dataset_covers_all_required_categories():
    examples = load_examples(DATASET_PATH)
    categories = {example.category for example in examples}

    assert EXPECTED_CATEGORIES.issubset(categories)


def test_dataset_has_all_difficulty_levels():
    examples = load_examples(DATASET_PATH)
    difficulties = {example.difficulty for example in examples}

    assert {"easy", "medium", "hard"}.issubset(difficulties)


def test_prompts_are_not_empty_or_too_short():
    examples = load_examples(DATASET_PATH)

    for example in examples:
        assert example.prompt.strip(), example.id
        assert len(example.prompt.strip()) >= 15, example.id


def test_expected_outputs_are_present():
    examples = load_examples(DATASET_PATH)

    for example in examples:
        assert example.expected is not None, example.id

        if isinstance(example.expected, str):
            assert example.expected.strip(), example.id

@pytest.mark.parametrize(
    ("method", "expected"),
    [
        ("json_schema", "not a dictionary"),
        ("keyword_match", {"required_keywords": "Python"}),
        ("contains_all", {"required_phrases": []}),
        ("normalized_exact_match", ""),
        ("reference_text", None),
        ("semantic_similarity", "   "),
        ("output_constraints", "not a dictionary"),
        ("llm_judge", "not a dictionary"),
        ("unknown_method", "some value"),
    ],
)
def test_invalid_scoring_combinations(method, expected):
    assert validate_scoring_compatibility(method, expected)

