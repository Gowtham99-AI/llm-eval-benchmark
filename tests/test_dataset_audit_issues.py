
from llm_eval_benchmark.dataset_audit import find_scoring_issues


def test_current_dataset_has_no_scoring_compatibility_issues():
    issues = find_scoring_issues("data/examples.jsonl")

    assert issues == [], f"Scoring compatibility issues: {issues}"
