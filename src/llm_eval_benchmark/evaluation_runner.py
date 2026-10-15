
from dataclasses import dataclass
from time import perf_counter
from typing import Literal

from llm_eval_benchmark.dataset_loader import load_examples
from llm_eval_benchmark.metrics import JsonSchemaScorer
from llm_eval_benchmark.model_adapter import ModelAdapter


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
    ) -> None:
        self.adapter = adapter
        self.dataset_path = dataset_path
        self.scorer = JsonSchemaScorer()

    def run(self) -> EvaluationReport:
        examples = load_examples(self.dataset_path)
        results: list[ExampleResult] = []

        # Validate the entire dataset before running the model.
        # This prevents partial benchmark runs caused by
        # unsupported scoring methods or invalid configurations.
        for example in examples:
            if example.scoring_method not in self.scorer.supported_methods:
                raise ValueError(
                    f"Unsupported evaluation: {example.scoring_method}"
                )

            if (
                example.category != "structured_output"
                or example.scoring_method != "json_schema"
                or not isinstance(example.expected, dict)
            ):
                raise ValueError(
                    f"Unsupported evaluation configuration for "
                    f"example {example.id!r}: "
                    f"category={example.category!r}, "
                    f"scoring_method={example.scoring_method!r}, "
                    f"expected_type={type(example.expected).__name__}"
                )

        # Run each example independently so one model failure
        # does not prevent the remaining examples from being evaluated.
        for example in examples:
            start_time = perf_counter()

            try:
                response = self.adapter.generate(example.prompt)
            except Exception as exc:
                latency_seconds = perf_counter() - start_time

                results.append(
                    ExampleResult(
                        example_id=example.id,
                        model_id=self.adapter.model_id,
                        response="",
                        scores={},
                        latency_seconds=latency_seconds,
                        status="error",
                        error=f"{type(exc).__name__}: {exc}",
                    )
                )
                continue

            latency_seconds = perf_counter() - start_time

            try:
                scores = self.scorer.score(example.expected, response)
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
                    )
                )
                continue

            # Token counting is optional. Adapters that do not
            # implement count_tokens can still be evaluated.
            generated_tokens: int | None = None
            count_tokens = getattr(self.adapter, "count_tokens", None)

            if callable(count_tokens):
                try:
                    generated_tokens = count_tokens(response)
                except Exception:
                    # Token counting must not invalidate a valid score.
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
                )
            )

        # Aggregate scores from successful examples only.
        successful_results = [
            result for result in results
            if result.status == "success"
        ]

        # Calculate each metric independently. This remains correct
        # even if different scoring methods produce different metrics.
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
