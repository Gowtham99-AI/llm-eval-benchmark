import json

import pytest

from llm_eval_benchmark.report_comparator import (
    compare_reports,
    load_report,
)


def write_report(tmp_path, filename, model_id, fingerprint):
    report = {
        "run_metadata": {
            "model_id": model_id,
            "dataset_fingerprint": fingerprint,
            "aggregate_scores": {
                "json_valid": 1.0,
                "exact_match": 0.8,
            },
        },
        "results": [
            {
                "example_id": "json_001",
                "status": "success",
                "scores": {
                    "json_valid": 1.0,
                    "exact_match": 0.8,
                },
                "latency_seconds": 0.5,
                "tokens_per_second": 20.0,
            },
            {
                "example_id": "json_002",
                "status": "error",
                "scores": {},
                "latency_seconds": 0.1,
                "tokens_per_second": None,
            },
        ],
    }

    path = tmp_path / filename
    path.write_text(json.dumps(report), encoding="utf-8")
    return path


def test_compare_reports_calculates_metric_differences(tmp_path):
    report_a = write_report(
        tmp_path, "a.json", "model-a", "same-fingerprint"
    )
    report_b = write_report(
        tmp_path, "b.json", "model-b", "same-fingerprint"
    )

    result = compare_reports(report_a, report_b)

    assert result["model_a"] == "model-a"
    assert result["model_b"] == "model-b"
    assert result["dataset_comparison"]["same_dataset"] is True
    assert (
        result["metric_comparison"]["exact_match"][
            "difference_b_minus_a"
        ]
        == pytest.approx(0.0)
    )
    assert result["performance"]["model_a"][
        "average_latency_seconds"
    ] == pytest.approx(0.5)
    assert result["example_counts"]["model_a"]["failed"] == 1


def test_compare_reports_flags_different_datasets(tmp_path):
    report_a = write_report(tmp_path, "a.json", "model-a", "hash-a")
    report_b = write_report(tmp_path, "b.json", "model-b", "hash-b")

    result = compare_reports(report_a, report_b)

    assert result["dataset_comparison"]["same_dataset"] is False


def test_compare_reports_marks_missing_fingerprint_unknown(tmp_path):
    report_a = write_report(tmp_path, "a.json", "model-a", None)
    report_b = write_report(tmp_path, "b.json", "model-b", "hash-b")

    result = compare_reports(report_a, report_b)

    assert result["dataset_comparison"]["same_dataset"] is None


def test_load_report_rejects_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_report(tmp_path / "missing.json")


def test_load_report_rejects_invalid_json(tmp_path):
    path = tmp_path / "invalid.json"
    path.write_text("{invalid", encoding="utf-8")

    with pytest.raises(ValueError, match="Invalid JSON"):
        load_report(path)


def test_performance_summary_calculates_latency_statistics(
    tmp_path,
):
    report = {
        "run_metadata": {"model_id": "test-model"},
        "aggregate_scores": {"accuracy": 1.0},
        "results": [
            {
                "status": "success",
                "scores": {"accuracy": 1.0},
                "latency_seconds": 1.0,
                "tokens_per_second": 10.0,
            },
            {
                "status": "success",
                "scores": {"accuracy": 1.0},
                "latency_seconds": 2.0,
                "tokens_per_second": 20.0,
            },
            {
                "status": "success",
                "scores": {"accuracy": 1.0},
                "latency_seconds": 3.0,
                "tokens_per_second": 30.0,
            },
            {
                "status": "error",
                "scores": {},
                "latency_seconds": 10.0,
            },
        ],
    }

    report_a = tmp_path / "report_a.json"
    report_b = tmp_path / "report_b.json"

    report_a.write_text(json.dumps(report), encoding="utf-8")
    report_b.write_text(json.dumps(report), encoding="utf-8")

    result = compare_reports(report_a, report_b)
    performance = result["performance"]["model_a"]

    assert performance["average_latency_seconds"] == 2.0
    assert performance["median_latency_seconds"] == 2.0
    assert performance["p95_latency_seconds"] == 3.0
    assert performance["min_latency_seconds"] == 1.0
    assert performance["max_latency_seconds"] == 3.0
    assert performance["success_rate"] == 0.75
    assert performance["measured_latency_examples"] == 3


def test_performance_summary_handles_no_latency_values(tmp_path):
    report = {
        "run_metadata": {"model_id": "test-model"},
        "aggregate_scores": {},
        "results": [
            {
                "status": "success",
                "scores": {},
                "latency_seconds": None,
            }
        ],
    }

    report_a = tmp_path / "report_a.json"
    report_b = tmp_path / "report_b.json"

    report_a.write_text(json.dumps(report), encoding="utf-8")
    report_b.write_text(json.dumps(report), encoding="utf-8")

    result = compare_reports(report_a, report_b)
    performance = result["performance"]["model_a"]

    assert performance["average_latency_seconds"] is None
    assert performance["median_latency_seconds"] is None
    assert performance["p95_latency_seconds"] is None
    assert performance["latency_stddev_seconds"] is None
    assert performance["success_rate"] == 1.0
    assert performance["measured_latency_examples"] == 0

