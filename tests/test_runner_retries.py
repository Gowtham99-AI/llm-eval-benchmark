
import json
from pathlib import Path

from llm_eval_benchmark.evaluation_runner import EvaluationRunner

VALID_RESPONSE = '{"name": "Ada", "active": true}'


class FakeAdapter:
    model_id = "fake-model"

    def __init__(self, failures: list[Exception] | None = None) -> None:
        self.failures = list(failures or [])
        self.calls = 0

    def generate(self, prompt: str) -> str:
        self.calls += 1

        if self.failures:
            raise self.failures.pop(0)

        return VALID_RESPONSE


def create_dataset(path: Path, count: int = 1) -> Path:
    """Create a minimal valid dataset for runner integration tests."""
    examples = [
        {
            "id": f"test-{index}",
            "category": "structured_output",
            "difficulty": "easy",
            "prompt": f"Return the expected JSON object for example {index}.",
            "expected": {"name": "Ada", "active": True},
            "scoring_method": "json_schema",
            "source": "unit-test",
            "version": "1.0",
        }
        for index in range(count)
    ]

    path.write_text(
        "".join(json.dumps(example) + "\n" for example in examples),
        encoding="utf-8",
    )
    return path


def test_transient_failure_is_retried_and_succeeds(
    tmp_path: Path,
) -> None:
    dataset = create_dataset(tmp_path / "dataset.jsonl")
    adapter = FakeAdapter([TimeoutError("Temporary timeout")])

    runner = EvaluationRunner(
        adapter=adapter,
        dataset_path=str(dataset),
        max_retries=2,
    )

    report = runner.run()

    assert adapter.calls == 2
    assert report.results[0].status == "success"
    assert report.results[0].attempt_count == 2


def test_exhausted_retries_record_attempt_count(
    tmp_path: Path,
) -> None:
    dataset = create_dataset(tmp_path / "dataset.jsonl")
    adapter = FakeAdapter(
        [
            TimeoutError("Timeout 1"),
            TimeoutError("Timeout 2"),
            TimeoutError("Timeout 3"),
        ]
    )

    runner = EvaluationRunner(
        adapter=adapter,
        dataset_path=str(dataset),
        max_retries=2,
    )

    report = runner.run()
    result = report.results[0]

    assert adapter.calls == 3
    assert result.status == "error"
    assert result.error_type == "timeout"
    assert result.attempt_count == 3


def test_permanent_error_is_not_retried(
    tmp_path: Path,
) -> None:
    dataset = create_dataset(tmp_path / "dataset.jsonl")
    adapter = FakeAdapter([ValueError("Invalid prompt")])

    runner = EvaluationRunner(
        adapter=adapter,
        dataset_path=str(dataset),
        max_retries=2,
    )

    report = runner.run()
    result = report.results[0]

    assert adapter.calls == 1
    assert result.status == "error"
    assert result.error_type == "validation_error"
    assert result.attempt_count == 1


def test_failed_example_does_not_stop_remaining_examples(
    tmp_path: Path,
) -> None:
    dataset = create_dataset(tmp_path / "dataset.jsonl", count=2)

    class PerPromptAdapter:
        model_id = "fake-model"

        def __init__(self) -> None:
            self.calls: list[str] = []

        def generate(self, prompt: str) -> str:
            self.calls.append(prompt)

            if "example 0" in prompt:
                raise ValueError("Permanent failure")

            return VALID_RESPONSE

    adapter = PerPromptAdapter()

    runner = EvaluationRunner(
        adapter=adapter,
        dataset_path=str(dataset),
        max_retries=2,
    )

    report = runner.run()

    assert len(adapter.calls) == 2
    assert len(report.results) == 2
    assert report.results[0].status == "error"
    assert report.results[1].status == "success"
