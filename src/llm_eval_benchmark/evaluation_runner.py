from dataclasses import dataclass
from time import perf_counter
from typing import Literal

from llm_eval_benchmark.dataset_loader import load_examples
from llm_eval_benchmark.metrics import (
    ContainsAllScorer,
    JsonSchemaScorer,
    KeywordMatchScorer,
)
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

        self.scorers = {
            "json_schema": JsonSchemaScorer(),
            "keyword_match": KeywordMatchScorer(),
            "contains_all": ContainsAllScorer(),
        }

    def run(self) -> EvaluationReport:
        examples = load_examples(self.dataset_path)
        results: list[ExampleResult] = []

        # Validate the entire dataset before calling the model.
        for example in examples:
            scorer = self.scorers.get(example.scoring_method)

            if scorer is None:
                raise ValueError(
                    f"Unsupported evaluation: {example.scoring_method}"
                )

            if example.scoring_method == "json_schema":
                valid_configuration = (
                    example.category == "structured_output"
                    and isinstance(example.expected, dict)
                )

            elif example.scoring_method == "keyword_match":
                valid_configuration = (
                    example.category
                    in {"software_qa", "code_understanding"}
                    and isinstance(example.expected, dict)
                )

            elif example.scoring_method == "contains_all":
                valid_configuration = (
                    example.category == "instruction_following"
                    and isinstance(example.expected, dict)
                )

            else:
                valid_configuration = False

            if not valid_configuration:
                raise ValueError(
                    f"Unsupported evaluation configuration for "
                    f"example {example.id!r}: "
                    f"category={example.category!r}, "
                    f"scoring_method={example.scoring_method!r}, "
                    f"expected_type={type(example.expected).__name__}"
                )

        # Evaluate examples independently so one model failure
        # does not prevent the remaining examples from running.
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

            # Select the scorer based on this example's scoring method.
            scorer = self.scorers[example.scoring_method]

            try:
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
                        error=(
                            f"Scoring error: "
                            f"{type(exc).__name__}: {exc}"
                        ),
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
                    # Token-counting failures must not invalidate scoring.
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

        # Aggregate metrics using successfully scored examples only.
        successful_results = [
            result
            for result in results
            if result.status == "success"
        ]

        # Each metric is averaged independently because different
        # evaluation tasks produce different metrics.
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
