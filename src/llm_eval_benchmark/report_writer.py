import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

from llm_eval_benchmark.evaluation_runner import EvaluationReport
from llm_eval_benchmark.reproducibility import (
    get_reproducibility_metadata,
)


def calculate_dataset_fingerprint(
    dataset_path: str | Path,
) -> str:
    """Return the SHA-256 fingerprint of the dataset file."""
    path = Path(dataset_path)

    if not path.is_file():
        raise FileNotFoundError(
            f"Dataset file not found: {path}"
        )

    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(8192), b""):
            digest.update(chunk)

    return digest.hexdigest()


def save_report(
    report: EvaluationReport,
    output_dir: str = "results",
    dataset_path: str = "data/examples.jsonl",
    backend: str | None = None,
    max_new_tokens: int | None = None,
) -> Path:
    """Save an evaluation report as a timestamped JSON file."""
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(UTC)
    filename = f"report_{timestamp:%Y%m%dT%H%M%S%fZ}.json"

    dataset_file = Path(dataset_path)
    fingerprint = (
        calculate_dataset_fingerprint(dataset_file)
        if dataset_file.is_file()
        else None
    )

    output_path = directory / filename

    payload = {
        "run_metadata": {
            "model_id": report.model_id,
            "created_at": timestamp.isoformat(),
            "total_examples": report.total_examples,
            "environment": get_reproducibility_metadata(),
            "successful_examples": sum(
                result.status == "success"
                for result in report.results
            ),
            "failed_examples": sum(
                result.status == "error"
                for result in report.results
            ),
            "dataset_path": str(dataset_path),
            "dataset_fingerprint": fingerprint,
            "backend": backend,
            "max_new_tokens": max_new_tokens,
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
                "error_type": result.error_type,
                "attempt_count": result.attempt_count,
                "peak_memory_mb": result.peak_memory_mb,
                "judge_model_id": result.judge_model_id,
                "judge_justifications": result.judge_justifications,
            }
            for result in report.results
        ],
    }

    output_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    return output_path
