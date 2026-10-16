from llm_eval_benchmark.evaluation_runner import classify_error


def test_timeout_error_is_classified() -> None:
    assert classify_error(
        TimeoutError("Generation timed out"),
        stage="generation",
    ) == "timeout"


def test_memory_error_is_classified() -> None:
    assert classify_error(
        MemoryError("Insufficient memory"),
        stage="generation",
    ) == "resource_error"


def test_generation_value_error_is_validation_error() -> None:
    assert classify_error(
        ValueError("Invalid prompt"),
        stage="generation",
    ) == "validation_error"


def test_scoring_error_is_classified_by_stage() -> None:
    assert classify_error(
        RuntimeError("Scoring failed"),
        stage="scoring",
    ) == "scoring_error"


def test_generation_runtime_error_is_model_error() -> None:
    assert classify_error(
        RuntimeError("Model execution failed"),
        stage="generation",
    ) == "model_error"