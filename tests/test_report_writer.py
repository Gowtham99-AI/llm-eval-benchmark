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

    # Ordinary evaluations should have no judge metadata.
    result = saved["results"][0]
    assert result["judge_model_id"] is None
    assert result["judge_justifications"] is None


def test_save_report_preserves_judge_metadata(tmp_path):
    report = EvaluationReport(
        model_id="fake-model",
        total_examples=1,
        results=[
            ExampleResult(
                example_id="judge_001",
                model_id="fake-model",
                response="Dependencies are provided externally.",
                scores={
                    "judge_relevance": 0.75,
                    "judge_correctness": 0.75,
                },
                judge_model_id="test-judge",
                judge_justifications={
                    "judge_relevance": "The answer addresses the question.",
                    "judge_correctness": "The answer is factually accurate.",
                },
            )
        ],
        aggregate_scores={
            "judge_relevance": 0.75,
            "judge_correctness": 0.75,
        },
    )

    output_path = save_report(report, tmp_path)
    saved = json.loads(output_path.read_text(encoding="utf-8"))

    result = saved["results"][0]

    assert result["judge_model_id"] == "test-judge"
    assert result["judge_justifications"] == {
        "judge_relevance": "The answer addresses the question.",
        "judge_correctness": "The answer is factually accurate.",
    }


def test_save_report_includes_reproducibility_metadata(tmp_path):
    report = EvaluationReport(
        model_id="fake-model",
        total_examples=1,
        results=[
            ExampleResult(
                example_id="json_001",
                model_id="fake-model",
                response='{"name": "Ada", "active": true}',
                scores={"exact_match": 1.0},
            )
        ],
        aggregate_scores={"exact_match": 1.0},
    )

    output_path = save_report(
        report,
        output_dir=tmp_path,
        backend="huggingface",
        max_new_tokens=128,
    )
    saved = json.loads(output_path.read_text(encoding="utf-8"))
    metadata = saved["run_metadata"]

    assert metadata["backend"] == "huggingface"
    assert metadata["max_new_tokens"] == 128

    environment = metadata["environment"]
    assert environment["python_version"]
    assert environment["platform"]
    assert isinstance(environment["package_versions"], dict)
