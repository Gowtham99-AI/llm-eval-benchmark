import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from llm_eval_benchmark.dataset_loader import load_examples
from llm_eval_benchmark.dataset_quality import (
    validate_scoring_compatibility,
)


def generate_quality_report(
    dataset_path: str | Path,
) -> dict[str, Any]:
    """Combine dataset coverage and scoring compatibility results."""
    coverage = audit_dataset(dataset_path)
    issues = find_scoring_issues(dataset_path)

    return {
        "dataset_path": str(dataset_path),
        "coverage": coverage,
        "scoring_compatibility": {
            "valid": not issues,
            "issue_count": len(issues),
            "issues": issues,
        },
    }


def save_quality_report(
    report: dict[str, Any],
    output_path: str | Path,
) -> None:
    """Persist the quality report as formatted JSON."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

def find_scoring_issues(dataset_path: str | Path) -> list[dict]:
    """Find dataset examples with incompatible scoring configurations."""
    examples = load_examples(dataset_path)
    issues: list[dict] = []

    for example in examples:
        errors = validate_scoring_compatibility(
            example.scoring_method,
            example.expected,
        )

        if errors:
            issues.append(
                {
                    "example_id": example.id,
                    "scoring_method": example.scoring_method,
                    "errors": errors,
                }
            )

    return issues

def audit_dataset(dataset_path: str | Path) -> dict:
    examples = load_examples(dataset_path)

    category_counts = Counter(
        example.category for example in examples
    )
    difficulty_counts = Counter(
        example.difficulty for example in examples
    )
    scoring_counts = Counter(
        example.scoring_method for example in examples
    )

    category_difficulties = defaultdict(Counter)
    category_scoring = defaultdict(Counter)

    for example in examples:
        category_difficulties[example.category][
            example.difficulty
        ] += 1

        category_scoring[example.category][
            example.scoring_method
        ] += 1

    return {
        "total_examples": len(examples),
        "categories": dict(category_counts),
        "difficulties": dict(difficulty_counts),
        "scoring_methods": dict(scoring_counts),
        "category_difficulties": {
            category: dict(counts)
            for category, counts in category_difficulties.items()
        },
        "category_scoring_methods": {
            category: dict(counts)
            for category, counts in category_scoring.items()
        },
    }
