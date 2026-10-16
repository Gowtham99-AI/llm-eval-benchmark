import pytest

from llm_eval_benchmark.report_writer import (
    calculate_dataset_fingerprint,
)
from llm_eval_benchmark.reproducibility import (
    get_reproducibility_metadata,
)


def test_reproducibility_metadata_contains_environment():
    metadata = get_reproducibility_metadata()

    assert metadata["python_version"]
    assert metadata["platform"]
    assert isinstance(metadata["package_versions"], dict)


def test_package_versions_are_strings_or_none():
    metadata = get_reproducibility_metadata()

    for package_version in metadata["package_versions"].values():
        assert package_version is None or isinstance(
            package_version, str
        )

def test_dataset_fingerprint_is_stable(tmp_path):
    dataset = tmp_path / "examples.jsonl"
    dataset.write_text('{"id":"ex_001"}\n', encoding="utf-8")

    first = calculate_dataset_fingerprint(dataset)
    second = calculate_dataset_fingerprint(dataset)

    assert first == second
    assert len(first) == 64


def test_dataset_fingerprint_changes_when_content_changes(tmp_path):
    dataset = tmp_path / "examples.jsonl"
    dataset.write_text('{"id":"ex_001"}\n', encoding="utf-8")

    original_fingerprint = calculate_dataset_fingerprint(dataset)

    dataset.write_text('{"id":"ex_002"}\n', encoding="utf-8")
    updated_fingerprint = calculate_dataset_fingerprint(dataset)

    assert original_fingerprint != updated_fingerprint


def test_dataset_fingerprint_raises_for_missing_file(tmp_path):
    dataset = tmp_path / "missing.jsonl"

    with pytest.raises(FileNotFoundError):
        calculate_dataset_fingerprint(dataset)
