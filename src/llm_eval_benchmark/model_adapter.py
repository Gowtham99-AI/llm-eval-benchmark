
from typing import Protocol


class ModelAdapter(Protocol):
    """Contract that every benchmark model adapter must implement."""

    @property
    def model_id(self) -> str:
        """Return the model's identifier."""
        ...

    def generate(self, prompt: str) -> str:
        """Generate a response for the supplied prompt."""
        ...
