"""
Embedding provider abstraction.

Each provider exposes a single method: embed(text) -> list[float].
All vectors produced by a given provider instance are L2-normalised and
have the same dimension, making them safe to compare with cosine similarity.

Do NOT mix providers within a session — MockEmbeddingProvider outputs
128-dim vectors, LocalEmbeddingProvider outputs 384-dim vectors (MiniLM-L6-v2
native). Mixing dimensions will cause cosine_similarity to raise ValueError.

Drop-in replacement for embedding_store.generate_embedding:
    provider = get_embedding_provider("mock")
    vec = provider.embed(text)          # same contract as generate_embedding(text)
"""

import hashlib
import math


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class EmbeddingProvider:
    def embed(self, text: str) -> list[float]:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Mock provider — deterministic, zero-dependency, 128-dim
# ---------------------------------------------------------------------------

_MOCK_DIM = 128


class MockEmbeddingProvider(EmbeddingProvider):
    """
    Hash-derived unit vectors. Identical logic to embedding_store.generate_embedding;
    extracted here so the rest of the system can depend on this abstraction instead.
    """

    def embed(self, text: str) -> list[float]:
        raw: bytearray = bytearray()
        seed = 0
        target = _MOCK_DIM * 4
        while len(raw) < target:
            digest = hashlib.sha256(f"{seed}\x00{text}".encode("utf-8")).digest()
            raw.extend(digest)
            seed += 1

        vector: list[float] = []
        for i in range(_MOCK_DIM):
            word = int.from_bytes(raw[i * 4 : i * 4 + 4], "big")
            vector.append(word / 0xFFFFFFFF * 2.0 - 1.0)

        return _l2_normalize(vector)


# ---------------------------------------------------------------------------
# Local provider — sentence-transformers, 384-dim (MiniLM-L6-v2 native)
# ---------------------------------------------------------------------------

class LocalEmbeddingProvider(EmbeddingProvider):
    """
    Real semantic embeddings via sentence-transformers.
    Lazy-loads the model on first use so import cost is zero when using mock.
    Requires: pip install sentence-transformers
    """

    _MODEL_NAME = "all-MiniLM-L6-v2"

    def __init__(self) -> None:
        self._model = None  # deferred until first embed() call

    def _load(self):
        if self._model is not None:
            return
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "LocalEmbeddingProvider requires sentence-transformers. "
                "Install it with: pip install sentence-transformers"
            ) from exc
        self._model = SentenceTransformer(self._MODEL_NAME)

    def embed(self, text: str) -> list[float]:
        self._load()
        vector = self._model.encode(text, normalize_embeddings=True).tolist()
        return vector


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def get_embedding_provider(mode: str) -> EmbeddingProvider:
    """
    mode="mock"  → MockEmbeddingProvider  (128-dim, no dependencies)
    mode="local" → LocalEmbeddingProvider (384-dim, requires sentence-transformers)
    """
    if mode == "mock":
        return MockEmbeddingProvider()
    if mode == "local":
        return LocalEmbeddingProvider()
    raise ValueError(
        f"Unknown embedding mode '{mode}'. Supported: 'mock', 'local'"
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _l2_normalize(vector: list[float]) -> list[float]:
    magnitude = math.sqrt(sum(x * x for x in vector))
    if magnitude == 0.0:
        return vector
    return [x / magnitude for x in vector]
