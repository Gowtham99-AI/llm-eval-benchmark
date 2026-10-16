
from llm_eval_benchmark.dataset_loader import load_examples


def test_dataset_has_no_duplicate_ids():
    examples = load_examples("data/examples.jsonl")
    ids = [example.id for example in examples]

    assert len(ids) == len(set(ids))


def test_dataset_contains_expected_edge_case_categories():
    examples = load_examples("data/examples.jsonl")

    categories = {example.category for example in examples}

    assert {
        "structured_output",
        "software_qa",
        "code_understanding",
        "instruction_following",
    }.issubset(categories)


def test_dataset_prompts_and_expected_values_are_present():
    examples = load_examples("data/examples.jsonl")

    for example in examples:
        assert example.prompt.strip(), example.id
        assert example.expected is not None, example.id
