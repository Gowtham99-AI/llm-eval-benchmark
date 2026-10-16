from llm_eval_benchmark.evaluation_runner import (
    is_retryable_error,
)


def test_timeout_is_retryable() -> None:
    assert is_retryable_error(TimeoutError("Timed out")) is True


def test_connection_error_is_retryable() -> None:
    assert is_retryable_error(ConnectionError("Connection lost")) is True


def test_value_error_is_not_retryable() -> None:
    assert is_retryable_error(ValueError("Invalid input")) is False


def test_memory_error_is_not_retryable() -> None:
    assert is_retryable_error(MemoryError("Out of memory")) is False


def test_unclassified_runtime_error_is_not_retryable() -> None:
    assert is_retryable_error(RuntimeError("Model failed")) is False