"""
Embedding store: deterministic pseudo-embeddings + cosine similarity search.

Drop-in replacement point: swap generate_embedding() with any real encoder
(OpenAI, Cohere, sentence-transformers) without touching the rest of this module.
"""

import hashlib
import math

EMBEDDING_DIM = 128

# ---------------------------------------------------------------------------
# Embedding generation
# ---------------------------------------------------------------------------

def generate_embedding(text: str) -> list[float]:
    """
    Deterministic, hash-derived unit vector of EMBEDDING_DIM dimensions.

    Strategy:
      1. Iteratively hash (seed + text) with SHA-256 until we have
         EMBEDDING_DIM * 4 raw bytes (4 bytes → one float).
      2. Map each 4-byte word to [-1, 1] via unsigned-int normalisation.
      3. L2-normalise the result so cosine similarity equals dot product.
    """
    raw: bytearray = bytearray()
    seed = 0
    target = EMBEDDING_DIM * 4
    while len(raw) < target:
        digest = hashlib.sha256(f"{seed}\x00{text}".encode("utf-8")).digest()
        raw.extend(digest)
        seed += 1

    vector: list[float] = []
    for i in range(EMBEDDING_DIM):
        word = int.from_bytes(raw[i * 4 : i * 4 + 4], "big")
        # map [0, 0xFFFFFFFF] → [-1.0, 1.0]
        vector.append(word / 0xFFFFFFFF * 2.0 - 1.0)

    return _l2_normalize(vector)


# ---------------------------------------------------------------------------
# Chunk embedding
# ---------------------------------------------------------------------------

def embed_chunks(chunks: list[dict]) -> list[dict]:
    """Attach an embedding vector to every chunk. Input chunks are not mutated."""
    return [
        {
            "chunk_id":  c["chunk_id"],
            "text":      c["text"],
            "tokens":    c["tokens"],
            "embedding": generate_embedding(c["text"]),
        }
        for c in chunks
    ]


# ---------------------------------------------------------------------------
# Similarity search
# ---------------------------------------------------------------------------

def search(
    query: str,
    embedded_chunks: list[dict],
    top_k: int = 3,
) -> list[dict]:
    """
    Return the top_k chunks most similar to query, ranked by cosine similarity.

    Each result dict includes the original chunk fields plus a 'score' key.
    """
    if not embedded_chunks:
        return []

    top_k = min(top_k, len(embedded_chunks))
    query_vec = generate_embedding(query)

    scored = [
        {
            "chunk_id": c["chunk_id"],
            "text":     c["text"],
            "tokens":   c["tokens"],
            "score":    cosine_similarity(query_vec, c["embedding"]),
        }
        for c in embedded_chunks
    ]

    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:top_k]


# ---------------------------------------------------------------------------
# Cosine similarity
# ---------------------------------------------------------------------------

def cosine_similarity(a: list[float], b: list[float]) -> float:
    """
    Cosine similarity in pure Python.
    Returns a value in [-1.0, 1.0]; vectors produced by this module are
    L2-normalised, so this reduces to a dot product.
    """
    if len(a) != len(b):
        raise ValueError(
            f"Vector dimension mismatch: {len(a)} vs {len(b)}"
        )

    dot    = sum(x * y for x, y in zip(a, b))
    mag_a  = math.sqrt(sum(x * x for x in a))
    mag_b  = math.sqrt(sum(x * x for x in b))

    if mag_a == 0.0 or mag_b == 0.0:
        return 0.0

    return dot / (mag_a * mag_b)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _l2_normalize(vector: list[float]) -> list[float]:
    magnitude = math.sqrt(sum(x * x for x in vector))
    if magnitude == 0.0:
        return vector
    return [x / magnitude for x in vector]
