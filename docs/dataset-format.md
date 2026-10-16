# Dataset Format and Quality Guidelines

This document describes the JSONL dataset format used by `llm-eval-benchmark`, including schema fields, supported scoring methods, examples, and dataset quality guidelines.

## 1. Dataset format

Benchmark datasets use **JSONL (JSON Lines)** format.

- Each line contains one complete JSON object.
- Use UTF-8 encoding.
- Every example must have a unique `id`.
- Each example must conform to the project's dataset schema.
- Keep each JSON object on a single physical line.

The default example dataset is `data/examples.jsonl`.

## 2. Required fields

| Field | Type | Description |
|---|---|---|
| `id` | String | Unique identifier for the example |
| `category` | String | Task category |
| `difficulty` | String | Expected task difficulty |
| `prompt` | String | Input provided to the evaluated model |
| `expected` | String or object | Reference answer or scoring configuration |
| `scoring_method` | String | Metric used to evaluate the response |
| `source` | String | Origin of the example |
| `version` | String | Dataset or example version |

### Supported categories

- `software_qa` — Software engineering questions and answers.
- `structured_output` — JSON and other structured response tasks.
- `instruction_following` — Tasks with explicit response requirements.
- `code_understanding` — Code explanation and comprehension tasks.

### Supported difficulty levels

- `easy`
- `medium`
- `hard`

## 3. Scoring methods

The expected value must match the selected scoring method.

### `json_schema`

Checks JSON validity, schema validity, and exact match.

Example:

```json
{"id":"json_001","category":"structured_output","difficulty":"easy","prompt":"Return JSON with name Ada and active true.","expected":{"name":"Ada","active":true},"scoring_method":"json_schema","source":"manual","version":"0.1.0"}
```

### `keyword_match`

Checks whether required keywords appear in the response.

```json
{"id":"keyword_001","category":"software_qa","difficulty":"easy","prompt":"Explain dependency injection and mention two benefits.","expected":{"required_keywords":["dependency injection","testability"]},"scoring_method":"keyword_match","source":"manual","version":"0.1.0"}
```

### `contains_all`

Checks whether every required phrase appears in the response.

```json
{"id":"phrase_001","category":"instruction_following","difficulty":"easy","prompt":"Explain the requirements and include both specified phrases.","expected":{"required_phrases":["input validation","error handling"]},"scoring_method":"contains_all","source":"manual","version":"0.1.0"}
```

### `normalized_exact_match`

Compares the response with an expected string after normalizing case and whitespace.

```json
{"id":"exact_001","category":"software_qa","difficulty":"easy","prompt":"Return only the word SUCCESS.","expected":"SUCCESS","scoring_method":"normalized_exact_match","source":"manual","version":"0.1.0"}
```

### `output_constraints`

Checks configured response constraints. Supported options include minimum words, maximum words, maximum characters, and requiring a non-empty response.

```json
{"id":"constraints_001","category":"instruction_following","difficulty":"easy","prompt":"Summarize the purpose of unit tests in a short response.","expected":{"min_words":3,"max_words":40,"max_characters":300,"require_non_empty":true},"scoring_method":"output_constraints","source":"manual","version":"0.1.0"}
```

### `reference_text`

Evaluates generated text against a reference using ROUGE-L and BLEU when the optional text-metrics dependencies are installed.

```json
{"id":"reference_001","category":"software_qa","difficulty":"medium","prompt":"Explain what a unit test verifies.","expected":"A unit test verifies the behavior of a small, isolated unit of code.","scoring_method":"reference_text","source":"manual","version":"0.1.0"}
```

### `semantic_similarity`

Compares generated text with a reference using embedding-based semantic similarity. Install the optional semantic dependencies to use this metric.

```json
{"id":"semantic_001","category":"software_qa","difficulty":"medium","prompt":"Explain the purpose of caching.","expected":"Caching stores reusable results so repeated requests can be served faster.","scoring_method":"semantic_similarity","source":"manual","version":"0.1.0"}
```

### `llm_judge`

Uses a separately configured judge adapter to evaluate a response. The expected object can include a reference answer and supporting context.

```json
{"id":"judge_001","category":"code_understanding","difficulty":"medium","prompt":"Explain the purpose of this function.","expected":{"reference_answer":"The function validates input before processing it.","context":"The function is part of an input validation service."},"scoring_method":"llm_judge","source":"manual","version":"0.1.0"}
```

**Important:** `llm_judge` requires a separately configured judge adapter in the evaluation runner. Do not assume that the standard CLI automatically configures one.

## 4. Adding a new example

1. Open `data/examples.jsonl`.
2. Add one JSON object on a new line.
3. Assign a unique ID.
4. Select a supported category, difficulty, and scoring method.
5. Provide an `expected` value compatible with the scoring method.
6. Run the test suite.
7. Review the example for ambiguity and correctness before committing it.

Example ID conventions such as `json_001` and `keyword_001` are useful for readability, but uniqueness is the important requirement.

## 5. Dataset quality checklist

Before accepting a new example, verify the following:

- [ ] The prompt is clear and self-contained.
- [ ] The expected answer or scoring configuration is correct.
- [ ] The scoring method measures the intended behavior.
- [ ] The difficulty reflects the reasoning and implementation required.
- [ ] The category matches the task.
- [ ] The example covers a meaningful scenario rather than duplicating an existing case.
- [ ] Edge cases and failure conditions are represented where appropriate.
- [ ] The dataset contains no credentials, private information, or other sensitive data.
- [ ] The JSONL file remains valid and every ID is unique.
- [ ] Tests pass after the change.

## 6. Optional dependencies

Install only the extras needed for your evaluation tasks:

```powershell
uv sync --extra text-metrics
uv sync --extra semantic
uv sync --extra huggingface
```

To install all optional dependencies together:

```powershell
uv sync --all-extras
```

Hugging Face model evaluation may additionally require model downloads, network access, and sufficient memory.

## 7. Validation

Run the project tests:

```powershell
uv run pytest
```

Run lint checks:

```powershell
uv run ruff check .
```

These commands validate the implementation and code quality. They do not, by themselves, guarantee that the dataset's examples are factually correct or appropriately difficult. Dataset review remains a separate quality-control step.

## 8. Evaluation considerations

- Different scoring methods measure different aspects of model behavior. Their scores should not automatically be treated as interchangeable.
- Exact-match metrics can penalize semantically correct answers that use different wording.
- Keyword and phrase metrics can reward superficial matches without verifying deeper correctness.
- Semantic similarity depends on the embedding model and its limitations.
- LLM-judge results can vary with the judge model, prompt, and configuration.
- Results from the demo backend are useful for testing the benchmark pipeline, but they should not be interpreted as evidence of real-world model quality.

For model comparisons, keep the dataset version and evaluation settings consistent. Document any changes that could affect the results.
