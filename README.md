# LLM Evaluation Benchmark

A Python-based benchmark framework for evaluating, comparing, and tracking the quality and performance of Large Language Models (LLMs) using a structured evaluation dataset.

The project is designed to make LLM evaluation more systematic, testable, and reproducible through configurable scoring methods, model adapters, structured JSON reports, and automated quality checks.

## Key Features

- **Multiple model backends:** Demo backend for pipeline testing and Hugging Face backend for real model inference.
- **Structured evaluation datasets:** JSONL examples with categories, difficulty levels, expected outputs, and scoring methods.
- **Multiple evaluation metrics:** Structured-output validation, exact matching, keyword matching, instruction constraints, reference-text metrics, and semantic similarity.
- **Performance measurements:** Latency, generated-token counts, throughput, and optional peak-memory measurements where supported.
- **Error handling:** Records evaluation errors and supports retry and timeout policies.
- **LLM-as-a-judge support:** Supports configurable judge-based evaluation through the evaluation runner.
- **Benchmark reports:** Saves model results, aggregate scores, environment information, and dataset fingerprints.
- **Model comparison:** Compares saved JSON reports.
- **Engineering quality:** Automated tests, linting, coverage reporting, and CI-ready tooling.

## Architecture

```text
Evaluation Dataset (JSONL)
          |
          v
    CLI / Typer
          |
          v
   Evaluation Runner
          |
          +------------------+
          |                  |
          v                  v
     Model Adapter      Scoring System
          |                  |
          v                  v
     Model Output       Metric Results
          |                  |
          +--------+---------+
                   |
                   v
          Evaluation Report
                   |
                   v
           JSON Report File
                   |
                   v
          Report Comparison
```

The evaluation runner coordinates model inference, scoring, error handling, and report creation. Model adapters isolate backend-specific inference logic from the evaluation workflow.

## Requirements

- Python 3.12 or newer
- [uv](https://docs.astral.sh/uv/) for Python environment and dependency management
- Internet access and sufficient hardware resources when downloading and running Hugging Face models

## Installation

Clone the repository and open its root directory:

```bash
git clone https://github.com/Gowtham99-AI
cd llm-eval-benchmark
```

Install the project and development dependencies:

```bash
uv sync --locked
```

Install optional dependencies for text metrics and semantic similarity:

```bash
uv sync --locked --extra text-metrics --extra semantic
```

The Hugging Face backend requires additional dependencies:

```bash
uv sync --locked --extra huggingface
```

To install all optional capabilities together:

```bash
uv sync --locked --all-extras
```

The Hugging Face backend may download model weights on the first run. Model size, available RAM, and hardware acceleration affect inference performance.

## Quick Start

### 1. Inspect available commands

```bash
uv run benchmark --help
```

The CLI currently provides two commands:

- `evaluate` — evaluates a model and saves a JSON report.
- `compare` — compares two saved benchmark reports.

### 2. Run the demo backend

```bash
uv run benchmark evaluate --backend demo --dataset data/examples.jsonl
```

The demo backend is intended for validating the benchmark pipeline, dataset compatibility, metric execution, and report generation without requiring a real LLM.

**Important:** Demo scores do not represent the quality or performance of a production LLM.

### 3. Run a Hugging Face model

```bash
uv run benchmark evaluate --backend huggingface --model-id Qwen/Qwen2.5-0.5B-Instruct --dataset data/examples.jsonl --max-new-tokens 128
```

Available options:

| Option | Purpose |
|---|---|
| `--dataset` | Path to the JSONL evaluation dataset |
| `--backend` | Select `demo` or `huggingface` |
| `--model-id` | Hugging Face model identifier |
| `--max-new-tokens` | Maximum generated tokens, from 1 to 4096 |

The default backend is `demo`. The default Hugging Face model identifier is `Qwen/Qwen2.5-0.5B-Instruct`, and the default generation limit is 128 tokens.

Actual results depend on the model, prompt, generation configuration, hardware, and dataset.

## Dataset Format

The evaluation dataset uses JSON Lines: one JSON object per line.

Each example contains these fields:

| Field | Purpose |
|---|---|
| `id` | Unique example identifier |
| `category` | Task category |
| `difficulty` | `easy`, `medium`, or `hard` |
| `prompt` | Input sent to the model |
| `expected` | Expected answer or scoring configuration |
| `scoring_method` | Metric used to evaluate the response |
| `source` | Example provenance |
| `version` | Dataset/example version |

Example:

```json
{
  "id": "json_001",
  "category": "structured_output",
  "difficulty": "easy",
  "prompt": "Return a JSON object containing name Ada and active true.",
  "expected