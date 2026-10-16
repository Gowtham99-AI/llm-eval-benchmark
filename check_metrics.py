
from llm_eval_benchmark.metrics import JsonSchemaScorer

expected = {"name": "Ada", "active": True}
actual = '{"name": "Ada", "active": true}'

result = JsonSchemaScorer().score(expected, actual)
print(result)
