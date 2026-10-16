import hashlib

import pytest

from llm_eval_benchmark.report_writer import (
    calculate_dataset_fingerprint,
)


def test_fingerprint_matches_sha256(tmp_path):
    dataset = tmp_path / "examples.jsonl"
    content = b'{"id":"example_1"}\n'
    dataset.write_bytes(content)

    expected = hashlib.sha256(content).hexdigest()

    assert calculate_dataset_fingerprint(dataset) == expected


def test_fingerprint_changes_when_dataset_changes(tmp_path):
    dataset = tmp_path / "examples.jsonl"
    dataset.write_text('{"id":"example_1"}\n', encoding="utf-8")

    first = calculate_dataset_fingerprint(dataset)

    dataset.write_text('{"id":"example_2"}\n', encoding="utf-8")

    second = calculate_dataset_fingerprint(dataset)

    assert first != second


def test_fingerprint_rejects_missing_file(tmp_path):
    missing = tmp_path / "missing.jsonl"

    with pytest.raises(FileNotFoundError):
        calculate_dataset_fingerprint(missing)
