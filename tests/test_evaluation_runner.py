
import json

import pytest

from llm_eval_benchmark.evaluation_runner import EvaluationRunner


class FakeModelAdapter:
    """Deterministic model substitute for testing."""

    model_id = "fake-model"

    def __init__(self, response: str) -> None:
        self.response = response
        self.calls = 0

    def generate(self, prompt: str) -> str:
        self.calls += 1
        return self.response


def create_dataset(tmp_path, example):
    path = tmp_path / "examples.jsonl"
    path.write_text(
        json.dumps(example) + "\n",
        encoding="utf-8",
    )
    return path


def valid_example():
    return {
        "id": "json_001",
        "category": "structured_output",
        "difficulty": "easy",
        "prompt": "Return Ada as JSON.",
        "expected": {"name": "Ada", "active": True},
        "scoring_method": "json_schema",
        "source": "original",
        "version": "0.1.0",
    }


def test_runner_scores_model_response(tmp_path):
    dataset = create_dataset(tmp_path, valid_example())
    adapter = FakeModelAdapter(
        '{"name": "Ada", "active": true}'
    )

    report = EvaluationRunner(adapter, dataset).run()

    assert report.total_examples == 1
    assert report.model_id == "fake-model"
    assert report.aggregate_scores["exact_match"] == 1.0
    assert adapter.calls == 1


def test_runner_records_incorrect_response(tmp_path):
    dataset = create_dataset(tmp_path, valid_example())
    adapter = FakeModelAdapter("not valid JSON")

    report = EvaluationRunner(adapter, dataset).run()

    assert report.results[0].scores["json_valid"] == 0.0
    assert report.aggregate_scores["exact_match"] == 0.0


def test_runner_rejects_unsupported_scoring(tmp_path):
    example = valid_example()
    example["scoring_method"] = "unknown"
    dataset = create_dataset(tmp_path, example)
    adapter = FakeModelAdapter("{}")

    with pytest.raises(ValueError, match="Unsupported evaluation"):
        EvaluationRunner(adapter, dataset).run()

    assert adapter.calls == 0


def test_runner_records_performance_metrics(tmp_path):
    class MeasuredFakeAdapter(FakeModelAdapter):
        def count_tokens(self, text: str) -> int:
            return 10

    dataset = create_dataset(tmp_path, valid_example())
    adapter = MeasuredFakeAdapter(
        '{"name": "Ada", "active": true}'
    )

    report = EvaluationRunner(adapter, dataset).run()
    result = report.results[0]

    assert result.latency_seconds >= 0
    assert result.generated_tokens == 10

    if result.latency_seconds > 0:
        assert result.tokens_per_second is not None
        assert result.tokens_per_second > 0


def test_runner_continues_after_generation_failure(tmp_path):
    first = valid_example()
    second = valid_example()
    first["id"] = "json_001"
    second["id"] = "json_002"

    dataset = tmp_path / "examples.jsonl"
    dataset.write_text(
        json.dumps(first) + "\n" + json.dumps(second) + "\n",
        encoding="utf-8",
    )

    class IntermittentAdapter(FakeModelAdapter):
        def generate(self, prompt: str) -> str:
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("Simulated model failure")
            return '{"name": "Ada", "active": true}'

    adapter = IntermittentAdapter("unused")
    report = EvaluationRunner(adapter, dataset).run()

    assert report.total_examples == 2
    assert report.results[0].status == "error"
    assert "Simulated model failure" in report.results[0].error
    assert report.results[1].status == "success"
    assert report.aggregate_scores["exact_match"] == 1.0
    assert adapter.calls == 2


def test_runner_handles_all_generation_failures(tmp_path):
    dataset = create_dataset(tmp_path, valid_example())

    class FailingAdapter(FakeModelAdapter):
        def generate(self, prompt: str) -> str:
            raise RuntimeError("Model unavailable")

    report = EvaluationRunner(
        FailingAdapter("unused"),
        dataset,
    ).run()

    assert report.results[0].status == "error"
    assert report.aggregate_scores == {}
