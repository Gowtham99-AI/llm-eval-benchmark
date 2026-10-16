
# Dataset Content Quality Review

Review every example in `data/examples.jsonl`.

## 1. Prompt clarity
- [ ] The task is understandable without hidden assumptions.
- [ ] Required output format is explicitly stated where necessary.
- [ ] The prompt does not contain conflicting instructions.

## 2. Expected-answer quality
- [ ] The expected answer is factually correct.
- [ ] The answer satisfies all explicit requirements.
- [ ] The answer does not contain unnecessary ambiguity.
- [ ] Multiple valid answers are handled appropriately by the scorer.

## 3. Difficulty validity
- [ ] Easy examples test foundational knowledge or straightforward tasks.
- [ ] Medium examples require applying knowledge or combining a few steps.
- [ ] Hard examples require deeper reasoning, multiple constraints,
      or handling meaningful edge cases.
- [ ] Difficulty is based on task complexity, not prompt length alone.

## 4. Scoring suitability
- [ ] The scoring method measures the intended capability.
- [ ] Exact matching is used only when wording or structure should match.
- [ ] Keyword scoring does not reward irrelevant keyword mentions.
- [ ] Reference-based metrics use a suitable reference answer.
- [ ] Semantic similarity does not replace factual correctness checks.
- [ ] Output constraints match the prompt requirements.

## 5. Final decision

For each example, record:
- Example ID
- Review status: pass / revise / remove
- Issue found
- Proposed correction
- Reason for the correction
