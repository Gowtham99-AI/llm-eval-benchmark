
import json
import math
import statistics
from pathlib import Path
from typing import Any


def load_report(report_path: str | Path) -> dict[str, Any]:
    """Load and validate a benchmark report."""
    path = Path(report_path)

    if not path.is_file():
        raise FileNotFoundError(f"Report not found: {path}")

    try:
        with path.open("r", encoding="utf-8") as file:
            report = json.load(file)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON report: {path}") from exc

    if not isinstance(report, dict):
        raise ValueError("Report must contain a JSON object.")

    metadata = report.get("run_metadata")
    results = report.get("results")
    aggregate_scores = report.get("aggregate_scores")

    if aggregate_scores is None:
        aggregate_scores = metadata.get("aggregate_scores")

    if not isinstance(aggregate_scores, dict):
        raise ValueError(
            "Report must contain aggregate_scores at the top level "
            "or inside run_metadata."
        )

    report["aggregate_scores"] = aggregate_scores

    if not isinstance(metadata, dict):
        raise ValueError("Report is missing run_metadata.")

    if not isinstance(metadata.get("model_id"), str):
        raise ValueError("Report is missing a valid model_id.")

    if not isinstance(results, list):
        raise ValueError("Report is missing a results list.")

    if not isinstance(aggregate_scores, dict):
        raise ValueError(
            "Report must contain an aggregate_scores object "
            "at the top level."
        )

    for result in results:
        if not isinstance(result, dict):
            raise ValueError("Each result must be a JSON object.")

        if not isinstance(result.get("scores"), dict):
            raise ValueError("Each result must contain a scores object.")

    return report


def _average_numeric(values: list[Any]) -> float | None:
    """Average finite numeric values, excluding booleans."""
    valid_values = [
        float(value)
        for value in values
        if isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    ]

    if not valid_values:
        return None

    return sum(valid_values) / len(valid_values)


def _successful_results(
    report: dict[str, Any],
) -> list[dict[str, Any]]:
    """Return successful results, supporting older reports."""
    return [
        result
        for result in report["results"]
        if result.get("status", "success") == "success"
    ]


def _performance_summary(
    report: dict[str, Any],
) -> dict[str, float | None]:
    """Summarize performance for successful examples."""
    results = _successful_results(report)

    return {
        "average_latency_seconds": _average_numeric(
            [result.get("latency_seconds") for result in results]
        ),
        "average_tokens_per_second": _average_numeric(
            [result.get("tokens_per_second") for result in results]
        ),
    }

    
def _performance_summary(
    report: dict[str, Any],
) -> dict[str, float | int | None]:
    """Calculate latency statistics for successful examples."""
    results = _successful_results(report)

    latency_values = [
        float(value)
        for value in (result.get("latency_seconds") for result in results)
        if isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and value >= 0
    ]

    throughput_values = [
        result.get("tokens_per_second")
        for result in results
    ]

    if latency_values:
        ordered_latencies = sorted(latency_values)

        # Nearest-rank percentile: rank = ceil(p * n).
        p95_index = math.ceil(0.95 * len(ordered_latencies)) - 1

        latency_mean = statistics.mean(ordered_latencies)
        latency_median = statistics.median(ordered_latencies)
        latency_p95 = ordered_latencies[p95_index]
        latency_stddev = statistics.pstdev(ordered_latencies)
        latency_min = min(ordered_latencies)
        latency_max = max(ordered_latencies)
    else:
        latency_mean = None
        latency_median = None
        latency_p95 = None
        latency_stddev = None
        latency_min = None
        latency_max = None

    total_examples = len(report["results"])
    success_rate = (
        len(results) / total_examples
        if total_examples > 0
        else None
    )

    return {
        # Preserve existing field names.
        "average_latency_seconds": latency_mean,
        "average_tokens_per_second": _average_numeric(
            throughput_values
        ),
        # New statistical metrics.
        "median_latency_seconds": latency_median,
        "p95_latency_seconds": latency_p95,
        "latency_stddev_seconds": latency_stddev,
        "min_latency_seconds": latency_min,
        "max_latency_seconds": latency_max,
        "success_rate": success_rate,
        "measured_latency_examples": len(latency_values),
    }



def compare_reports(
    report_a_path: str | Path,
    report_b_path: str | Path,
) -> dict[str, Any]:
    """Compare saved benchmark reports without rerunning models."""
    report_a = load_report(report_a_path)
    report_b = load_report(report_b_path)

    metadata_a = report_a["run_metadata"]
    metadata_b = report_b["run_metadata"]

    fingerprint_a = metadata_a.get("dataset_fingerprint")
    fingerprint_b = metadata_b.get("dataset_fingerprint")

    if fingerprint_a and fingerprint_b:
        same_dataset: bool | None = fingerprint_a == fingerprint_b
    else:
        same_dataset = None

    # aggregate_scores is at the report's top level.
    scores_a = report_a["aggregate_scores"]
    scores_b = report_b["aggregate_scores"]

    metric_comparison: dict[str, dict[str, float]] = {}

    for metric in sorted(set(scores_a).intersection(scores_b)):
        value_a = scores_a[metric]
        value_b = scores_b[metric]

        if (
            isinstance(value_a, bool)
            or isinstance(value_b, bool)
            or not isinstance(value_a, (int, float))
            or not isinstance(value_b, (int, float))
        ):
            continue

        if not math.isfinite(value_a) or not math.isfinite(value_b):
            continue

        metric_comparison[metric] = {
            "model_a": float(value_a),
            "model_b": float(value_b),
            "difference_b_minus_a": float(value_b - value_a),
        }

    successful_a = len(_successful_results(report_a))
    successful_b = len(_successful_results(report_b))

    failed_a = sum(
        result.get("status") == "error"
        for result in report_a["results"]
    )
    failed_b = sum(
        result.get("status") == "error"
        for result in report_b["results"]
    )

    return {
        "model_a": metadata_a["model_id"],
        "model_b": metadata_b["model_id"],
        "dataset_comparison": {
            "fingerprint_a": fingerprint_a,
            "fingerprint_b": fingerprint_b,
            "same_dataset": same_dataset,
        },
        "metric_comparison": metric_comparison,
        "performance": {
            "model_a": _performance_summary(report_a),
            "model_b": _performance_summary(report_b),
        },
        "example_counts": {
            "model_a": {
                "successful": successful_a,
                "failed": failed_a,
            },
            "model_b": {
                "successful": successful_b,
                "failed": failed_b,
            },
        },
    }
