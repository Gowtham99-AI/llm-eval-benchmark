import json

from llm_eval_benchmark.evaluation_runner import EvaluationRunner
from llm_eval_benchmark.judge import JUDGE_DIMENSIONS


class FakeAdapter:
    model_id = "fake-target-model"

    def generate(self, prompt: str) -> str:
        return "Dependencies are provided externally."


class FakeJudgeAdapter:
    model_id = "fake-judge-model"

    def generate(self, prompt: str) -> str:
        return json.dumps({
            "scores": {
                dimension: 4 for dimension in JUDGE_DIMENSIONS
            },
            "justifications": {
                dimension: f"Reason for {dimension}"
                for dimension in JUDGE_DIMENSIONS
            },
        })


def test_runner_supports_llm_judge(tmp_path):
    dataset = tmp_path / "judge_examples.jsonl"

    example = {
        "id": "judge-001",
        "category": "software_qa",
        "difficulty": "easy",
        "prompt": "What is dependency injection?",
        "expected": {
            "reference_answer": (
                "Dependencies are provided to a component externally."
            )
        },
        "scoring_method": "llm_judge",
        "source": "unit-test",
        "version": "1.0",
    }

    dataset.write_text(
        json.dumps(example) + "\n",
        encoding="utf-8",
    )

    runner = EvaluationRunner(
        adapter=FakeAdapter(),
        judge_adapter=FakeJudgeAdapter(),
        dataset_path=str(dataset),
    )

    report = runner.run()

    result = report.results[0]
    assert result.status == "success"
    assert result.scores["judge_relevance"] == 0.75
    assert result.scores["judge_correctness"] == 0.75