
import json
from pathlib import Path

from pydantic import ValidationError

from llm_eval_benchmark.schemas import EvaluationExample


def load_examples(path: str | Path) -> list[EvaluationExample]:
    """Load and validate evaluation examples from a JSONL file."""

    dataset_path = Path(path)

    if not dataset_path.is_file():
        raise FileNotFoundError(
            f"Dataset file not found: {dataset_path}"
        )

    examples: list[EvaluationExample] = []
    seen_ids: dict[str, int] = {}

    with dataset_path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON at line {line_number}: {exc.msg}"
                ) from exc

            if not isinstance(record, dict):
                raise ValueError(
                    f"Expected a JSON object at line {line_number}."
                )

            try:
                example = EvaluationExample.model_validate(record)
            except ValidationError as exc:
                raise ValueError(
                    f"Invalid evaluation example at line "
                    f"{line_number}: {exc}"
                ) from exc

            if example.id in seen_ids:
                first_line = seen_ids[example.id]
                raise ValueError(
                    f"Duplicate example ID '{example.id}' at line "
                    f"{line_number}; first seen at line {first_line}."
                )

            seen_ids[example.id] = line_number
            examples.append(example)

    if not examples:
        raise ValueError(
            f"No evaluation examples found in: {dataset_path}"
        )

    return examples
