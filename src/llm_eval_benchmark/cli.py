
import json
from pathlib import Path

import typer

from llm_eval_benchmark.evaluation_runner import EvaluationRunner
from llm_eval_benchmark.report_writer import save_report

app = typer.Typer(
    name="benchmark",
    help="Evaluate language models on a reproducible dataset.",
    invoke_without_command=False,
)


class DemoModelAdapter:
    """Deterministic adapter used to verify the benchmark pipeline."""

    model_id = "demo-json-model"

    def generate(self, prompt: str) -> str:
        # Demo behavior only; this does not call a real LLM.
        return json.dumps({"name": "Ada", "active": True})



@app.command()
def evaluate(
    dataset: Path = typer.Option(
        Path("data/examples.jsonl"),
        "--dataset",
        help="Path to the JSONL evaluation dataset.",
        exists=True,
        readable=True,
        dir_okay=False,
    ),
) -> None:
    """Evaluate the demo model and save a JSON report."""
    try:
        report = EvaluationRunner(
            adapter=DemoModelAdapter(),
            dataset_path=dataset,
        ).run()

        output_path = save_report(report, "results")

    except (ValueError, OSError) as exc:
        typer.echo(f"Evaluation failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    typer.echo(f"Model: {report.model_id}")
    typer.echo(f"Examples evaluated: {report.total_examples}")
    typer.echo("Aggregate scores:")

    for metric, score in report.aggregate_scores.items():
        typer.echo(f"  {metric}: {score:.3f}")

    typer.echo(f"Report saved to: {output_path}")



if __name__ == "__main__":
    app()
