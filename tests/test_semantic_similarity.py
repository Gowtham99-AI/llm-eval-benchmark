
import numpy as np
import pytest

from llm_eval_benchmark.metrics import SemanticSimilarityScorer


class FakeEmbeddingModel:
    def encode(self, texts, normalize_embeddings=True):
        assert normalize_embeddings is True

        vectors = {
            "reference answer": [1.0, 0.0],
            "similar answer": [0.8, 0.6],
            "different answer": [0.0, 1.0],
        }
        return np.array([vectors[text] for text in texts])


@pytest.fixture
def scorer():
    return SemanticSimilarityScorer(model=FakeEmbeddingModel())


def test_identical_embeddings_have_similarity_one(scorer):
    result = scorer.score("reference answer", "reference answer")

    assert result["semantic_similarity"] == pytest.approx(1.0)


def test_similar_text_has_higher_score_than_different_text(scorer):
    similar = scorer.score("reference answer", "similar answer")
    different = scorer.score("reference answer", "different answer")

    assert similar["semantic_similarity"] > different["semantic_similarity"]


def test_empty_response_returns_zero(scorer):
    result = scorer.score("reference answer", "  ")

    assert result["semantic_similarity"] == 0.0


def test_rejects_empty_reference(scorer):
    with pytest.raises(ValueError, match="non-empty reference"):
        scorer.score("", "similar answer")


def test_rejects_non_string_response(scorer):
    with pytest.raises(ValueError, match="actual must be a string"):
        scorer.score("reference answer", None)
