# Contributing to llm-eval-benchmark

Thank you for your interest in improving this benchmark.

## Development setup

Requirements:
- Python 3.12 or compatible version supported by the project.
- `uv`.
- Git.

Install the project and development dependencies:

```powershell
uv sync --all-extras
```

## Contribution workflow

1. Create a branch for your change.
2. Make a focused change.
3. Add or update tests.
4. Update documentation when behavior or configuration changes.
5. Run the test suite and lint checks.
6. Submit a pull request describing the change and its impact.

## Adding benchmark examples

Add examples to the appropriate JSONL dataset or proposed dataset file.

Every example should:
- Have a unique ID.
- Follow the dataset schema.
- Use a supported category, difficulty, and scoring method.
- Have a clear prompt and appropriate expected output.
- Represent a meaningful test case.
- Avoid private information, credentials, and confidential data.

Document new scoring methods and add tests covering successful and invalid inputs.

## Code quality

Run:

```powershell
uv run pytest
uv run ruff check .
```

Do not introduce dependencies without a clear need. Optional integrations should remain optional when practical.

## Reporting issues

When reporting a bug, include:
- A concise description of the issue.
- Reproduction steps.
- Expected and actual behavior.
- Relevant error messages, with sensitive information removed.
- Environment and dependency details when relevant.

## Benchmark result integrity

Do not present demo-backend results as real model performance. Report the dataset, model, configuration, and relevant limitations when publishing comparisons.
