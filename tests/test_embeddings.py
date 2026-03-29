"""
Unit tests for the CLIP retrieval pipeline.

Tests:
- text embeddings have the expected shape
- normalized embeddings have unit norm
- identical queries have cosine similarity ~ 1.0
- different queries have cosine similarity < 1.0
- retrieval-style top_k scores are sorted in descending order

Run with:
    pytest tests/test_embeddings.py -v
"""

import torch
import pytest
import open_clip


MODEL_NAME = "ViT-B-32"
PRETRAINED = "laion2b_s34b_b79k"
EMBED_DIM = 512


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def clip_model():
    """Load CLIP model once for all tests in this module."""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model, _, preprocess = open_clip.create_model_and_transforms(
        MODEL_NAME,
        pretrained=PRETRAINED,
    )
    model = model.to(device)
    model.eval()

    tokenizer = open_clip.get_tokenizer(MODEL_NAME)
    return model, tokenizer, preprocess, device


# ---------------------------------------------------------------------------
# Embedding shape tests
# ---------------------------------------------------------------------------

class TestEmbeddingShape:
    def test_text_embedding_dim(self, clip_model):
        """A single text query should produce a 512-d embedding."""
        model, tokenizer, _, device = clip_model
        tokens = tokenizer(["a person reflecting on their identity"]).to(device)

        with torch.no_grad():
            emb = model.encode_text(tokens)

        assert emb.shape == (1, EMBED_DIM), f"Expected (1, {EMBED_DIM}), got {tuple(emb.shape)}"

    def test_batch_text_embedding_shape(self, clip_model):
        """A batch of N queries should produce shape (N, 512)."""
        model, tokenizer, _, device = clip_model
        queries = [
            "a person feeling trapped and powerless",
            "uncertainty before transformation",
            "social isolation in a crowd",
        ]
        tokens = tokenizer(queries).to(device)

        with torch.no_grad():
            emb = model.encode_text(tokens)

        assert emb.shape == (3, EMBED_DIM), f"Expected (3, {EMBED_DIM}), got {tuple(emb.shape)}"


# ---------------------------------------------------------------------------
# Normalisation tests
# ---------------------------------------------------------------------------

class TestNormalisation:
    def test_normalized_embeddings_have_unit_norm(self, clip_model):
        """L2-normalized embeddings must have norm ≈ 1.0."""
        model, tokenizer, _, device = clip_model
        tokens = tokenizer(["a person confronting inner conflict"]).to(device)

        with torch.no_grad():
            emb = model.encode_text(tokens)

        emb = torch.nn.functional.normalize(emb, dim=-1)
        norm = emb.norm(dim=-1).item()

        assert abs(norm - 1.0) < 1e-5, f"Expected norm 1.0, got {norm:.8f}"


# ---------------------------------------------------------------------------
# Cosine similarity tests
# ---------------------------------------------------------------------------

class TestCosineSimilarity:
    def test_identical_queries_score_one(self, clip_model):
        """A normalized embedding compared to itself should have cosine similarity ~ 1.0."""
        model, tokenizer, _, device = clip_model
        query = "uncertainty before transformation"
        tokens = tokenizer([query]).to(device)

        with torch.no_grad():
            emb = model.encode_text(tokens)

        emb = torch.nn.functional.normalize(emb, dim=-1)
        score = (emb @ emb.T).item()

        assert abs(score - 1.0) < 1e-5, f"Expected self-similarity 1.0, got {score:.8f}"

    def test_different_queries_score_less_than_one(self, clip_model):
        """Different queries should not have cosine similarity 1.0."""
        model, tokenizer, _, device = clip_model
        queries = [
            "a sunny tropical beach",
            "a dark underground cave",
        ]
        tokens = tokenizer(queries).to(device)

        with torch.no_grad():
            emb = model.encode_text(tokens)

        emb = torch.nn.functional.normalize(emb, dim=-1)
        score = (emb[0] @ emb[1]).item()

        assert score < 1.0, f"Expected similarity < 1.0, got {score:.8f}"


# ---------------------------------------------------------------------------
# Retrieval ordering tests
# ---------------------------------------------------------------------------

class TestRetrievalOrder:
    def test_top_k_results_are_sorted_descending(self, clip_model):
        """Retrieval-style top-k results must be sorted by descending similarity."""
        model, tokenizer, _, device = clip_model

        torch.manual_seed(42)
        image_embeddings = torch.randn(5, EMBED_DIM, device=device)
        image_embeddings = torch.nn.functional.normalize(image_embeddings, dim=-1)

        tokens = tokenizer(["a person questioning their identity"]).to(device)
        with torch.no_grad():
            text_emb = model.encode_text(tokens)

        text_emb = torch.nn.functional.normalize(text_emb, dim=-1)

        scores = (image_embeddings @ text_emb.T).squeeze(-1)
        top_k = 3
        top = scores.topk(top_k)

        top_indices = top.indices.tolist()
        top_scores = [scores[i].item() for i in top_indices]

        assert top_scores == sorted(top_scores, reverse=True), (
            f"Scores are not sorted descending: {top_scores}"
        )

    def test_top_k_size_matches_request(self, clip_model):
        """topk(k) should return exactly k results when enough embeddings exist."""
        model, tokenizer, _, device = clip_model

        torch.manual_seed(0)
        image_embeddings = torch.randn(8, EMBED_DIM, device=device)
        image_embeddings = torch.nn.functional.normalize(image_embeddings, dim=-1)

        tokens = tokenizer(["emotional burden and inner struggle"]).to(device)
        with torch.no_grad():
            text_emb = model.encode_text(tokens)

        text_emb = torch.nn.functional.normalize(text_emb, dim=-1)

        scores = (image_embeddings @ text_emb.T).squeeze(-1)
        top_k = 5
        top_indices = scores.topk(top_k).indices.tolist()

        assert len(top_indices) == top_k, f"Expected {top_k} results, got {len(top_indices)}"