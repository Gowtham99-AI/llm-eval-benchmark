
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

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



@dataclass(frozen=True)
class EvaluationReport:
    model_id: str
    total_examples: int
    results: list[ExampleResult]
    aggregate_scores: dict[str, float]


class EvaluationRunner:
    """Run supported evaluation examples against a model adapter."""

    def __init__(
        self,
        adapter: ModelAdapter,
        dataset_path: str | Path,
    ) -> None:
        self.adapter = adapter
        self.dataset_path = Path(dataset_path)
        self.scorer = JsonSchemaScorer()

    def run(self) -> EvaluationReport:
        examples = load_examples(self.dataset_path)
        results: list[ExampleResult] = []

        # Validate support before making model calls.
        for example in examples:
            if (
                example.category != "structured_output"
                or example.scoring_method != "json_schema"
                or not isinstance(example.expected, dict)
            ):
                raise ValueError(
                    f"Unsupported evaluation example: {example.id}. "
                    "This runner currently supports structured_output "
                    "examples using json_schema scoring with dict "
                    "expected values."
                )

        
        for example in examples:
            start_time = perf_counter()
            response = self.adapter.generate(example.prompt)
            latency_seconds = perf_counter() - start_time

            scores = self.scorer.score(example.expected, response)

            count_tokens = getattr(self.adapter, "count_tokens", None)
            generated_tokens = (
                count_tokens(response)
                if callable(count_tokens)
                else None
            )

            tokens_per_second = (
                generated_tokens / latency_seconds
                if generated_tokens is not None
                and latency_seconds > 0
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
                )
            )

        metric_names = results[0].scores.keys()
        aggregate_scores = {
            metric: sum(result.scores[metric] for result in results)
            / len(results)
            for metric in metric_names
        }

        return EvaluationReport(
            model_id=self.adapter.model_id,
            total_examples=len(results),
            results=results,
            aggregate_scores=aggregate_scores,
        )
