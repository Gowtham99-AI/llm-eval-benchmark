
class HuggingFaceAdapter:
    """Run a local Hugging Face causal language model."""

    def __init__(
        self,
        model_id: str = "Qwen/Qwen2.5-0.5B-Instruct",
        max_new_tokens: int = 128,
    ) -> None:
        if max_new_tokens < 1:
            raise ValueError("max_new_tokens must be at least 1.")

        try:
            import torch
            from transformers import (
                AutoModelForCausalLM,
                AutoTokenizer,
            )
        except ImportError as exc:
            raise RuntimeError(
                "Hugging Face dependencies are missing. "
                "Install them with: "
                "uv sync --extra huggingface"
            ) from exc

        self.model_id = model_id
        self.max_new_tokens = max_new_tokens
        self.torch = torch

        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(model_id)
        self.model.eval()

    
    def generate(self, prompt: str) -> str:
        """Generate text without including the input prompt in the result."""
        if not prompt.strip():
            raise ValueError("Prompt must not be empty.")

        if getattr(self.tokenizer, "chat_template", None):
            messages = [{"role": "user", "content": prompt}]
            inputs = self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
                return_dict=True,
            )
        else:
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
            )

        input_length = inputs["input_ids"].shape[1]

        with self.torch.inference_mode():
            output = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=(
                    self.tokenizer.pad_token_id
                    if self.tokenizer.pad_token_id is not None
                    else self.tokenizer.eos_token_id
                ),
            )

        generated_tokens = output[0][input_length:]

        return self.tokenizer.decode(
            generated_tokens,
            skip_special_tokens=True,
        ).strip()

    
    def count_tokens(self, text: str) -> int:
        """Count generated text tokens using this model's tokenizer."""
        return len(
            self.tokenizer.encode(
                text,
                add_special_tokens=False,
            )
        )
