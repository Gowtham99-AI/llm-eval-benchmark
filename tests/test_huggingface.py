
from llm_eval_benchmark.huggingface_adapter import HuggingFaceAdapter

adapter = HuggingFaceAdapter(
    model_id="Qwen/Qwen2.5-0.5B-Instruct",
    max_new_tokens=128,
)

response = adapter.generate(
    'Return only this JSON object: {"name": "Ada", "active": true}'
)

print(f"Model: {adapter.model_id}")
print(f"Response: {response}")
