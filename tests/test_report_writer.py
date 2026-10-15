
import json

from llm_eval_benchmark.evaluation_runner import (
    EvaluationReport,
    ExampleResult,
)
from llm_eval_benchmark.report_writer import save_report


def test_save_report_writes_valid_json(tmp_path):
    report = EvaluationReport(
        model_id="fake-model",
        total_examples=1,
        results=[
            ExampleResult(
                example_id="json_001",
                model_id="fake-model",
                response='{"name": "Ada", "active": true}',
                scores={
                    "json_valid": 1.0,
                    "schema_valid": 1.0,
                    "exact_match": 1.0,
                },
            )
        ],
        aggregate_scores={
            "json_valid": 1.0,
            "schema_valid": 1.0,
            "exact_match": 1.0,
        },
    )

    output_path = save_report(report, tmp_path)
    saved = json.loads(output_path.read_text(encoding="utf-8"))

    assert saved["run_metadata"]["model_id"] == "fake-model"
    assert saved["run_metadata"]["total_examples"] == 1
    assert saved["results"][0]["example_id"] == "json_001"
    assert saved["aggregate_scores"]["exact_match"] == 1.0
