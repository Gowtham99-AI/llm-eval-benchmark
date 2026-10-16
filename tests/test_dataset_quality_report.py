
import json

from llm_eval_benchmark.dataset_audit import (
    generate_quality_report,
    save_quality_report,
)


def test_quality_report_contains_coverage_and_compatibility(
    tmp_path,
):
    report = generate_quality_report("data/examples.jsonl")

    assert report["coverage"]["total_examples"] == 24
    assert report["scoring_compatibility"]["valid"] is True
    assert report["scoring_compatibility"]["issue_count"] == 0

    output_path = tmp_path / "quality" / "report.json"
    save_quality_report(report, output_path)

    saved_report = json.loads(
        output_path.read_text(encoding="utf-8")
    )

    assert saved_report == report
