
import json
from typing import Any, ClassVar


def matches_expected_types(
    expected: Any,
    actual: Any,
) -> bool:
    """Check whether actual values have the expected JSON types."""

    # bool is a subclass of int in Python, so use exact type checks.
    if type(expected) is bool:
        return type(actual) is bool

    if type(expected) is int:
        return type(actual) is int

    if type(expected) is float:
        return type(actual) in (int, float) and type(actual) is not bool

    if isinstance(expected, str):
        return isinstance(actual, str)

    if isinstance(expected, list):
        return (
            isinstance(actual, list)
            and len(expected) == len(actual)
            and all(
                matches_expected_types(e, a)
                for e, a in zip(expected, actual)
            )
        )

    if isinstance(expected, dict):
        return (
            isinstance(actual, dict)
            and expected.keys() == actual.keys()
            and all(
                matches_expected_types(expected[key], actual[key])
                for key in expected
            )
        )

    if expected is None:
        return actual is None

    return False


class JsonSchemaScorer:
    """Score a model response against an expected JSON value."""
    supported_methods: ClassVar[set[str]] = {"json_schema"}

    def score(
        self,
        expected: dict[str, Any],
        actual: str,
    ) -> dict[str, float]:
        try:
            parsed = json.loads(actual)
        except (json.JSONDecodeError, TypeError):
            return {
                "json_valid": 0.0,
                "schema_valid": 0.0,
                "exact_match": 0.0,
            }

        schema_valid = matches_expected_types(expected, parsed)
        exact_match = schema_valid and parsed == expected

        return {
            "json_valid": 1.0,
            "schema_valid": float(schema_valid),
            "exact_match": float(exact_match),
        }

class KeywordMatchScorer:
    """Measure the proportion of required keywords found in a response."""

    supported_methods: ClassVar[set[str]] = {"keyword_match"}

    def score(
        self,
        expected: dict,
        actual: str,
    ) -> dict[str, float]:
        keywords = expected.get("required_keywords", [])

        if not isinstance(keywords, list) or not keywords:
            raise ValueError(
                "expected must contain a non-empty "
                "'required_keywords' list"
            )

        normalized_response = actual.casefold()

        matched = sum(
            1
            for keyword in keywords
            if isinstance(keyword, str)
            and keyword.casefold() in normalized_response
        )

        return {
            "keyword_recall": matched / len(keywords),
        }


class ContainsAllScorer:
    """Check whether all required phrases appear in a response."""

    supported_methods: ClassVar[set[str]] = {"contains_all"}

    def score(
        self,
        expected: dict,
        actual: str,
    ) -> dict[str, float]:
        phrases = expected.get("required_phrases", [])

        if not isinstance(phrases, list) or not phrases:
            raise ValueError(
                "expected must contain a non-empty "
                "'required_phrases' list"
            )

        normalized_response = " ".join(actual.casefold().split())

        matched = sum(
            1
            for phrase in phrases
            if isinstance(phrase, str)
            and " ".join(phrase.casefold().split())
            in normalized_response
        )

        return {
            "instruction_compliance": (
                1.0 if matched == len(phrases) else 0.0
            ),
        }


class NormalizedExactMatchScorer:
    """Compare text after normalizing case and whitespace."""

    supported_methods: ClassVar[set[str]] = {"normalized_exact_match"}

    @staticmethod
    def normalize(text: str) -> str:
        """Normalize case and collapse consecutive whitespace."""
        return " ".join(text.casefold().split())

    def score(
        self,
        expected: str,
        actual: str,
    ) -> dict[str, float]:
        if not isinstance(expected, str):
            raise ValueError(
                "expected must be a string for normalized_exact_match"
            )

        if not isinstance(actual, str):
            raise ValueError("actual must be a string")

        matched = self.normalize(expected) == self.normalize(actual)

        return {
            "normalized_exact_match": float(matched),
        }


class OutputConstraintScorer:
    """Evaluate simple response-length and non-empty constraints."""

    supported_methods: ClassVar[set[str]] = {"output_constraints"}

    def score(
        self,
        expected: dict[str, Any],
        actual: str,
    ) -> dict[str, float]:
        if not isinstance(actual, str):
            raise ValueError("actual must be a string")

        min_words = expected.get("min_words")
        max_words = expected.get("max_words")
        max_characters = expected.get("max_characters")
        require_non_empty = expected.get("require_non_empty", False)

        if not isinstance(require_non_empty, bool):
            raise ValueError("require_non_empty must be a boolean")

        for name, value in (
            ("min_words", min_words),
            ("max_words", max_words),
            ("max_characters", max_characters),
        ):
            if value is not None and (
                type(value) is not int or value < 0
            ):
                raise ValueError(f"{name} must be a non-negative integer")

        if (
            min_words is not None
            and max_words is not None
            and min_words > max_words
        ):
            raise ValueError("min_words cannot exceed max_words")

        word_count = len(actual.split())
        character_count = len(actual)
        non_empty = bool(actual.strip())

        constraints: list[bool] = []

        if min_words is not None:
            constraints.append(word_count >= min_words)

        if max_words is not None:
            constraints.append(word_count <= max_words)

        if max_characters is not None:
            constraints.append(character_count <= max_characters)

        if require_non_empty:
            constraints.append(non_empty)

        # No constraints means there is nothing to evaluate.
        if not constraints:
            raise ValueError("At least one output constraint is required")

        return {
            "output_constraints_pass_rate": (
                sum(constraints) / len(constraints)
            ),
            "output_constraints_passed": float(all(constraints)),
        }


class ReferenceTextScorer:
    """Evaluate generated text against a reference using ROUGE-L and BLEU."""

    supported_methods: ClassVar[set[str]] = {"reference_text"}

    def score(
        self,
        expected: str,
        actual: str,
    ) -> dict[str, float]:
        if not isinstance(expected, str) or not expected.strip():
            raise ValueError("expected must be a non-empty reference string")

        if not isinstance(actual, str):
            raise ValueError("actual must be a string")

        try:
            from nltk.translate.bleu_score import (
                SmoothingFunction,
                sentence_bleu,
            )
            from rouge_score import rouge_scorer
        except ImportError as exc:
            raise ImportError(
                "Reference text metrics require optional dependencies. "
                "Install them with: uv sync --extra text-metrics"
            ) from exc

        reference_tokens = expected.casefold().split()
        actual_tokens = actual.casefold().split()

        rouge = rouge_scorer.RougeScorer(
            ["rougeL"],
            use_stemmer=True,
        )
        rouge_result = rouge.score(expected, actual)

        bleu = (
            sentence_bleu(
                [reference_tokens],
                actual_tokens,
                smoothing_function=SmoothingFunction().method1,
            )
            if actual_tokens
            else 0.0
        )

        return {
            "rouge_l": float(rouge_result["rougeL"].fmeasure),
            "bleu": float(bleu),
        }


class SemanticSimilarityScorer:
    """Compare reference and generated text using embedding cosine similarity."""

    supported_methods: ClassVar[set[str]] = {"semantic_similarity"}

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        model: Any = None,
    ) -> None:
        self.model_name = model_name
        self._model = model

    def _get_model(self) -> Any:
        """Load the embedding model only when it is first needed."""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise ImportError(
                    "Semantic similarity requires sentence-transformers. "
                    "Install it with: uv sync --extra semantic"
                ) from exc

            self._model = SentenceTransformer(self.model_name)

        return self._model

    def score(
        self,
        expected: str,
        actual: str,
    ) -> dict[str, float]:
        if not isinstance(expected, str) or not expected.strip():
            raise ValueError("expected must be a non-empty reference string")

        if not isinstance(actual, str):
            raise ValueError("actual must be a string")

        if not actual.strip():
            return {"semantic_similarity": 0.0}

        model = self._get_model()
        embeddings = model.encode(
            [expected, actual],
            normalize_embeddings=True,
        )

        # Normalized embeddings make their dot product cosine similarity.
        similarity = float(embeddings[0] @ embeddings[1])

        return {"semantic_similarity": max(-1.0, min(1.0, similarity))}
