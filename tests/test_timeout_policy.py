import pytest

from llm_eval_benchmark.evaluation_runner import (
    generate_with_timeout,
)


class FakeAdapter:
    model_id = "fake-model"

    def generate(self, prompt: str) -> str:
        return f"Response: {prompt}"


def test_generation_without_timeout_uses_adapter() -> None:
    adapter = FakeAdapter()

    result = generate_with_timeout(
        adapter,
        "Hello",
        None,
    )

    assert result == "Response: Hello"


def test_timeout_requires_backend_support() -> None:
    adapter = FakeAdapter()

    with pytest.raises(NotImplementedError, match="does not support"):
        generate_with_timeout(
            adapter,
            "Hello",
            5.0,
        )


def test_non_positive_timeout_is_rejected() -> None:
    # Constructor validation is tested in the runner tests.
    with pytest.raises(ValueError):
        from llm_eval_benchmark.evaluation_runner import EvaluationRunner

        EvaluationRunner(
            adapter=FakeAdapter(),
            generation_timeout_seconds=0,
        )