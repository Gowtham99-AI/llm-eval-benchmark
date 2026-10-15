
import json
from datetime import UTC, datetime
from pathlib import Path

from llm_eval_benchmark.evaluation_runner import EvaluationReport


def save_report(
    report: EvaluationReport,
    output_dir: str | Path = "results",
) -> Path:
    """Save an evaluation report as a timestamped JSON file."""
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(UTC)
    filename = f"report_{timestamp:%Y%m%dT%H%M%S%fZ}.json"
    output_path = directory / filename

    payload = {
        "run_metadata": {
            "model_id": report.model_id,
            "created_at": timestamp.isoformat(),
            "total_examples": report.total_examples,
            "successful_examples": sum(result.status == "success" for result in report.results),
            "failed_examples": sum(result.status == "error" for result in report.results),
        },
        "aggregate_scores": report.aggregate_scores,
        "results": [
            {
                "example_id": result.example_id,
                "model_id": result.model_id,
                "response": result.response,
                "scores": result.scores,
                "latency_seconds": result.latency_seconds,
                "generated_tokens": result.generated_tokens,
                "tokens_per_second": result.tokens_per_second,
                "status": result.status,
                "error": result.error,
            }
            for result in report.results
        ],
    }

    output_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return output_path
