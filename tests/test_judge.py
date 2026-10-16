
import json

import pytest

from llm_eval_benchmark.judge import JUDGE_DIMENSIONS, LLMJudge


def make_judge_response(rating: int = 4) -> str:
    return json.dumps({
        "scores": {
            dimension: rating for dimension in JUDGE_DIMENSIONS
        },
        "justifications": {
            dimension: f"Test justification for {dimension}"
            for dimension in JUDGE_DIMENSIONS
        },
    })


class FakeJudgeAdapter:
    model_id = "fake-judge"

    def __init__(self, response: str) -> None:
        self.response = response
        self.last_prompt = ""

    def generate(self, prompt: str) -> str:
        self.last_prompt = prompt
        return self.response


def test_valid_response_is_normalized():
    judge = LLMJudge(FakeJudgeAdapter(make_judge_response(5)))

    result = judge.evaluate(
        question="What is dependency injection?",
        answer="Dependencies are provided externally.",
    )

    assert result.judge_model_id == "fake-judge"
    assert result.overall_score == pytest.approx(1.0)
    assert result.scores["judge_relevance"] == 1.0


def test_lowest_rating_normalizes_to_zero():
    judge = LLMJudge(FakeJudgeAdapter(make_judge_response(1)))

    result = judge.evaluate(
        question="Explain dependency injection.",
        answer="I do not know.",
    )

    assert result.overall_score == pytest.approx(0.0)


def test_invalid_json_is_rejected():
    judge = LLMJudge(FakeJudgeAdapter("not JSON"))

    with pytest.raises(ValueError, match="invalid JSON"):
        judge.evaluate("Question", "Answer")


def test_out_of_range_rating_is_rejected():
    judge = LLMJudge(FakeJudgeAdapter(make_judge_response(6)))

    with pytest.raises(ValueError, match="integer from 1 to 5"):
        judge.evaluate("Question", "Answer")


def test_missing_justification_is_rejected():
    payload = json.loads(make_judge_response())
    payload["justifications"].pop("clarity")

    judge = LLMJudge(FakeJudgeAdapter(json.dumps(payload)))

    with pytest.raises(ValueError, match="missing or unexpected"):
        judge.evaluate("Question", "Answer")


def test_empty_question_is_rejected():
    judge = LLMJudge(FakeJudgeAdapter(make_judge_response()))

    with pytest.raises(ValueError, match="question"):
        judge.evaluate(" ", "Answer")


def test_context_and_reference_are_included_in_prompt():
    adapter = FakeJudgeAdapter(make_judge_response())
    judge = LLMJudge(adapter)

    judge.evaluate(
        question="What does the document say?",
        answer="It describes dependency injection.",
        reference_answer="Dependencies are supplied externally.",
        context="The document explains dependency injection.",
    )

    assert "reference_answer" in adapter.last_prompt
    assert "context" in adapter.last_prompt
