# Evaluation Usage Examples

This guide demonstrates how to run a benchmark, evaluate a Hugging Face model, compare reports, and interpret results.

## 1. Run the demo backend

The demo backend is useful for checking the evaluation pipeline without downloading a real model.

```powershell
uv run benchmark evaluate --backend demo --dataset data/examples.jsonl
```

Review the console output and generated report. Use the report path displayed by your CLI or the configured output location in your project.

**Important:** Demo scores are not evidence of real-world model quality. The demo backend is primarily intended for pipeline validation.

## 2. Evaluate a Hugging Face model

Install the optional Hugging Face dependencies:

```powershell
uv sync --extra huggingface
```

Run an evaluation:

```powershell
uv run benchmark evaluate --backend huggingface --model-id Qwen/Qwen2.5-0.5B-Instruct --dataset data/examples.jsonl --max-new-tokens 128
```

The first run may download model files. Runtime and memory requirements depend on the selected model and hardware.

## 3. Compare two evaluation reports

Run evaluations using the same dataset and compatible settings, retaining both generated reports.

Then compare them:

```powershell
uv run benchmark compare results/report-a.json results/report-b.json
```

Replace the example filenames with the actual report paths created by your runs.

A meaningful comparison should use compatible datasets, scoring definitions, and evaluation configurations.

## 4. Interpret benchmark metrics

| Metric | What it tells you | What to watch for |
|---|---|---|
| Exact match | Whether output matches the expected answer after the configured normalization | Valid answers can differ in wording |
| JSON validity | Whether generated output is valid JSON | Valid JSON may still have incorrect values |
| Schema validity | Whether output satisfies the expected structure | Correct structure does not guarantee factual correctness |
| Keyword match | Whether required keywords appear | Keywords can appear in an incorrect explanation |
| Phrase match | Whether all required phrases appear | Phrase presence does not establish correctness |
| ROUGE-L / BLEU | Text overlap with a reference | Different valid wording can lower scores |
| Semantic similarity | Similarity between embeddings of generated and reference text | Similarity is not a guarantee of factual accuracy |
| Latency | Time taken to produce a response, where recorded | Hardware, warm-up, and model loading affect timing |
| Throughput | Generation speed, where recorded | Compare using consistent measurement conditions |

Only interpret metrics actually present in the report for the selected scoring method.

## 5. Understand report differences

When one model scores higher than another:

1. Inspect the aggregate score and individual metric values.
2. Review examples where the models disagree.
3. Check for errors, timeouts, or incomplete responses.
4. Verify that both runs used compatible datasets and settings.
5. Consider latency, throughput, and memory alongside quality.
6. Avoid selecting a model based on one metric alone.

A benchmark score is evidence about the tested dataset and configuration, not a universal ranking of model capability.

## 6. Reproducibility checklist

Before publishing results, record:

- Dataset version and fingerprint, when available.
- Model identifier and relevant generation settings.
- Package and runtime versions.
- Hardware details when relevant and available.
- Metric configuration and optional dependencies.
- Errors or skipped examples.
- The exact commands used to run the benchmark.

Even with matching settings, different hardware, library versions, and nondeterministic generation can produce different results.

## 7. Run quality checks

Run the project's automated tests:

```powershell
uv run pytest
```

Run lint checks:

```powershell
uv run ruff check .
```

Passing tests confirms the tested behaviors; it does not independently validate every model output or guarantee benchmark fairness.
