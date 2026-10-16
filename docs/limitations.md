# Benchmark Limitations

## Dataset coverage

Benchmark results depend on the quality, diversity, and representativeness of the dataset. A small dataset may not capture the range of real-world tasks, edge cases, or domain-specific requirements.

## Metric limitations

- Exact-match metrics can penalize correct answers with different wording.
- Keyword and phrase checks may reward superficial matches.
- Reference-based metrics measure text overlap rather than complete factual correctness.
- Embedding similarity does not guarantee factual accuracy.
- LLM-as-judge evaluations may vary with judge models, prompts, and configuration.
- JSON validity does not guarantee that generated values are semantically correct.

## Model and environment differences

Latency and throughput depend on hardware, model size, quantization, runtime, concurrency, warm-up, and generation settings. Comparisons should use consistent conditions wherever possible.

## Reproducibility

Recording configuration and environment metadata improves reproducibility but cannot guarantee identical results across hardware, library versions, or nondeterministic inference implementations.

## Demo backend

The demo backend exists to test the benchmark pipeline. Its outputs and scores should not be used to rank real language models.

## Interpretation

Use benchmark scores as one input to model selection, alongside representative task testing, reliability, cost, latency, memory usage, and safety requirements.

A high score on this dataset does not establish universal model superiority or guarantee production readiness.
