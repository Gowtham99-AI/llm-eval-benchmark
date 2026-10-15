from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class EvaluationExample(BaseModel):
    """A single example in the LLM evaluation dataset."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    category: Literal[
        "software_qa",
        "structured_output",
        "instruction_following",
        "code_understanding",
    ]
    difficulty: Literal["easy", "medium", "hard"]
    prompt: str = Field(min_length=1)
    expected: str | dict
    scoring_method: str = Field(min_length=1)
    source: str = Field(min_length=1)
    version: str = Field(min_length=1)
