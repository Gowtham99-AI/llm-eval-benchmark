
from llm_eval_benchmark.dataset_audit import audit_dataset


def test_audit_reports_dataset_coverage():
    report = audit_dataset("data/examples.jsonl")

    assert report["total_examples"] == 24
    assert len(report["categories"]) == 4
    assert report["difficulties"] == {
        "easy": 8,
        "medium": 8,
        "hard": 8,
    }

    assert sum(report["scoring_methods"].values()) == 24


def test_each_category_has_examples_at_multiple_difficulties():
    report = audit_dataset("data/examples.jsonl")

    for counts in report["category_difficulties"].values():
        assert len(counts) >= 2
