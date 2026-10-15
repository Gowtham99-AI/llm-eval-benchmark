
import json

import pytest

from llm_eval_benchmark.dataset_loader import load_examples


def valid_record(example_id: str = "json_001") -> dict:
    return {
        "id": example_id,
        "category": "structured_output",
        "difficulty": "easy",
        "prompt": "Return JSON with name set to Ada.",
        "expected": {"name": "Ada"},
        "scoring_method": "json_schema",
        "source": "original",
        "version": "0.1.0",
    }


def write_jsonl(path, records: list[dict]) -> None:
    path.write_text(
        "\n".join(json.dumps(record) for record in records) + "\n",
        encoding="utf-8",
    )


def test_loads_valid_examples(tmp_path):
    path = tmp_path / "examples.jsonl"
    write_jsonl(path, [valid_record(), valid_record("json_002")])

    examples = load_examples(path)

    assert len(examples) == 2
    assert examples[0].id == "json_001"
    assert examples[1].id == "json_002"


def test_rejects_duplicate_ids(tmp_path):
    path = tmp_path / "examples.jsonl"
    write_jsonl(path, [valid_record(), valid_record()])

    with pytest.raises(ValueError, match="Duplicate example ID"):
        load_examples(path)


def test_rejects_invalid_json(tmp_path):
    path = tmp_path / "examples.jsonl"
    path.write_text('{"id": invalid}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="Invalid JSON at line 1"):
        load_examples(path)


def test_rejects_invalid_schema(tmp_path):
    path = tmp_path / "examples.jsonl"
    record = valid_record()
    record["category"] = "unknown"
    write_jsonl(path, [record])

    with pytest.raises(
        ValueError, match="Invalid evaluation example at line 1"
    ):
        load_examples(path)


def test_rejects_empty_dataset(tmp_path):
    path = tmp_path / "examples.jsonl"
    path.write_text("\n  \n", encoding="utf-8")

    with pytest.raises(ValueError, match="No evaluation examples"):
        load_examples(path)


def test_rejects_missing_file(tmp_path):
    path = tmp_path / "missing.jsonl"

    with pytest.raises(FileNotFoundError):
        load_examples(path)
