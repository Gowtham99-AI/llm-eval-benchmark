from dataclasses import dataclass
from time import perf_counter
from typing import Literal

from llm_eval_benchmark.dataset_loader import load_examples
from llm_eval_benchmark.judge import LLMJudge
from llm_eval_benchmark.metrics import (
    ContainsAllScorer,
    JsonSchemaScorer,
    KeywordMatchScorer,
    NormalizedExactMatchScorer,
    OutputConstraintScorer,
    ReferenceTextScorer,
    SemanticSimilarityScorer,
)
from llm_eval_benchmark.model_adapter import ModelAdapter

ErrorStage = Literal["generation", "scoring"]


def classify_error(
    exc: Exception,
    *,
    stage: ErrorStage,
) -> str:
    """Classify an evaluation failure using its stage and exception type."""
    if isinstance(exc, TimeoutError):
        return "timeout"

    if isinstance(exc, MemoryError):
        return "resource_error"

    if stage == "scoring":
        return "scoring_error"

    if isinstance(exc, ValueError):
        return "validation_error"

    if stage == "generation":
        return "model_error"

    return "unknown_error"


def is_retryable_error(exc: Exception) -> bool:
    """Return whether a generation failure is considered transient."""
    return isinstance(exc, (TimeoutError, ConnectionError))


def generate_with_timeout(
    adapter: ModelAdapter,
    prompt: str,
    timeout_seconds: float | None,
) -> str:
    """Generate a response using backend timeout support when requested."""
    if timeout_seconds is None:
        return adapter.generate(prompt)

    timeout_method = getattr(adapter, "generate_with_timeout", None)

    if not callable(timeout_method):
        raise NotImplementedError(
            f"Adapter {adapter.model_id!r} does not support "
            "generation timeouts."
        )

    return timeout_method(prompt, timeout_seconds)


@dataclass(frozen=True)
class ExampleResult:
    example_id: str
    model_id: str
    response: str
    scores: dict[str, float]
    latency_seconds: float = 0.0
    generated_tokens: int | None = None
    tokens_per_second: float | None = None
    status: Literal["success", "error"] = "success"
    error: str | None = None
    error_type: str | None = None
    attempt_count: int = 1
    peak_memory_mb: float | None = None
    judge_model_id: str | None = None
    judge_justifications: dict[str, str] | None = None


@dataclass(frozen=True)
class EvaluationReport:
    model_id: str
    total_examples: int
    results: list[ExampleResult]
    aggregate_scores: dict[str, float]


class EvaluationRunner:
    def __init__(
        self,
        adapter: ModelAdapter,
        dataset_path: str = "data/examples.jsonl",
        max_retries: int = 2,
        generation_timeout_seconds: float | None = None,
        judge_adapter: ModelAdapter | None = None,
    ) -> None:
        if max_retries < 0:
            raise ValueError("max_retries cannot be negative.")

        if (
            generation_timeout_seconds is not None
            and generation_timeout_seconds <= 0
        ):
            raise ValueError(
                "generation_timeout_seconds must be greater than zero."
            )

        self.adapter = adapter
        self.dataset_path = dataset_path
        self.max_retries = max_retries
        self.generation_timeout_seconds = generation_timeout_seconds

        self.scorers = {
            "json_schema": JsonSchemaScorer(),
            "keyword_match": KeywordMatchScorer(),
            "contains_all": ContainsAllScorer(),
            "normalized_exact_match": NormalizedExactMatchScorer(),
            "output_constraints": OutputConstraintScorer(),
            "reference_text": ReferenceTextScorer(),
            "semantic_similarity": SemanticSimilarityScorer(),
        }

        self.judge = (
            LLMJudge(judge_adapter)
            if judge_adapter is not None
            else None
        )

    def run(self) -> EvaluationReport:
        examples = load_examples(self.dataset_path)
        results: list[ExampleResult] = []

        # Validate the entire dataset before calling the generation model.
        for example in examples:
            if example.scoring_method == "llm_judge":
                if self.judge is None:
                    raise ValueError(
                        "The llm_judge scoring method requires a judge_adapter."
                    )

                if not isinstance(example.expected, dict):
                    raise ValueError(
                        "llm_judge expected value must be a dictionary."
                    )

                reference_answer = example.expected.get("reference_answer")
                context = example.expected.get("context")

                if reference_answer is not None and not isinstance(
                    reference_answer, str
                ):
                    raise ValueError(
                        "llm_judge reference_answer must be a string."
                    )

                if context is not None and not isinstance(context, str):
                    raise ValueError("llm_judge context must be a string.")

                continue

            scorer = self.scorers.get(example.scoring_method)

            if scorer is None:
                raise ValueError(
                    f"Unsupported evaluation: {example.scoring_method}"
                )

            expected = example.expected
            method = example.scoring_method
            category = example.category

            if method == "json_schema":
                valid_configuration = (
                    category == "structured_output"
                    and isinstance(expected, dict)
                )

            elif method == "keyword_match":
                valid_configuration = (
                    category in {"software_qa", "code_understanding"}
                    and isinstance(expected, dict)
                )

            elif method == "contains_all":
                valid_configuration = (
                    category == "instruction_following"
                    and isinstance(expected, dict)
                )

            elif method == "normalized_exact_match":
                valid_configuration = isinstance(expected, str)

            elif method == "output_constraints":
                valid_configuration = isinstance(expected, dict)

            elif method in {"reference_text", "semantic_similarity"}:
                valid_configuration = (
                    isinstance(expected, str) and bool(expected.strip())
                )

            else:
                valid_configuration = False

            if not valid_configuration:
                raise ValueError(
                    f"Unsupported evaluation configuration for "
                    f"example {example.id!r}: "
                    f"category={category!r}, "
                    f"scoring_method={method!r}, "
                    f"expected_type={type(expected).__name__}"
                )

        # Evaluate each example independently.
        for example in examples:
            reset_memory = getattr(
                self.adapter,
                "reset_peak_memory_stats",
                None,
            )

            if callable(reset_memory):
                try:
                    reset_memory()
                except Exception:
                    # Instrumentation must not interrupt evaluation.
                    pass

            start_time = perf_counter()
            attempt_count = 0
            response: str | None = None
            generation_error: Exception | None = None

            # Retry transient generation failures only.
            while attempt_count <= self.max_retries:
                attempt_count += 1

                try:
                    response = generate_with_timeout(
                        self.adapter,
                        example.prompt,
                        self.generation_timeout_seconds,
                    )
                    generation_error = None
                    break

                except Exception as exc:
                    generation_error = exc

                    if (
                        attempt_count > self.max_retries
                        or not is_retryable_error(exc)
                    ):
                        break

            latency_seconds = perf_counter() - start_time

            if generation_error is not None:
                results.append(
                    ExampleResult(
                        example_id=example.id,
                        model_id=self.adapter.model_id,
                        response="",
                        scores={},
                        latency_seconds=latency_seconds,
                        status="error",
                        error=(
                            f"{type(generation_error).__name__}: "
                            f"{generation_error}"
                        ),
                        error_type=classify_error(
                            generation_error,
                            stage="generation",
                        ),
                        attempt_count=attempt_count,
                    )
                )
                continue

            assert response is not None

            # Read optional peak-memory measurements.
            peak_memory_mb: float | None = None
            get_peak_memory = getattr(
                self.adapter,
                "get_peak_memory_mb",
                None,
            )

            if callable(get_peak_memory):
                try:
                    peak_memory_mb = get_peak_memory()
                except Exception:
                    peak_memory_mb = None

            # Initialize judge metadata for every example.
            judge_model_id: str | None = None
            judge_justifications: dict[str, str] | None = None

            # Scoring failures are not retried.
            try:
                if example.scoring_method == "llm_judge":
                    if self.judge is None:
                        raise RuntimeError(
                            "The llm_judge scoring method requires "
                            "a judge_adapter."
                        )

                    expected_data = example.expected

                    if not isinstance(expected_data, dict):
                        raise ValueError(
                            "llm_judge expected value must be a dictionary."
                        )

                    judge_result = self.judge.evaluate(
                        question=example.prompt,
                        answer=response,
                        reference_answer=expected_data.get(
                            "reference_answer"
                        ),
                        context=expected_data.get("context"),
                    )

                    scores = judge_result.scores
                    judge_model_id = judge_result.judge_model_id
                    judge_justifications = judge_result.justifications

                else:
                    scorer = self.scorers[example.scoring_method]
                    scores = scorer.score(example.expected, response)

            except Exception as exc:
                results.append(
                    ExampleResult(
                        example_id=example.id,
                        model_id=self.adapter.model_id,
                        response=response,
                        scores={},
                        latency_seconds=latency_seconds,
                        status="error",
                        error=f"Scoring error: {type(exc).__name__}: {exc}",
                        error_type=classify_error(
                            exc,
                            stage="scoring",
                        ),
                        attempt_count=attempt_count,
                        peak_memory_mb=peak_memory_mb,
                        judge_model_id=judge_model_id,
                        judge_justifications=judge_justifications,
                    )
                )
                continue

            # Token counting is optional.
            generated_tokens: int | None = None
            count_tokens = getattr(self.adapter, "count_tokens", None)

            if callable(count_tokens):
                try:
                    generated_tokens = count_tokens(response)
                except Exception:
                    generated_tokens = None

            tokens_per_second = (
                generated_tokens / latency_seconds
                if generated_tokens is not None and latency_seconds > 0
                else None
            )

            results.append(
                ExampleResult(
                    example_id=example.id,
                    model_id=self.adapter.model_id,
                    response=response,
                    scores=scores,
                    latency_seconds=latency_seconds,
                    generated_tokens=generated_tokens,
                    tokens_per_second=tokens_per_second,
                    status="success",
                    attempt_count=attempt_count,
                    peak_memory_mb=peak_memory_mb,
                    judge_model_id=judge_model_id,
                    judge_justifications=judge_justifications,
                )
            )

        # Aggregate metrics from successfully scored examples only.
        successful_results = [
            result for result in results if result.status == "success"
        ]

        metric_names = {
            metric
            for result in successful_results
            for metric in result.scores
        }

        aggregate_scores: dict[str, float] = {}

        for metric in metric_names:
            metric_results = [
                result.scores[metric]
                for result in successful_results
                if metric in result.scores
            ]

            if metric_results:
                aggregate_scores[metric] = (
                    sum(metric_results) / len(metric_results)
                )

        return EvaluationReport(
            model_id=self.adapter.model_id,
            total_examples=len(results),
            results=results,
            aggregate_scores=aggregate_scores,
        )
