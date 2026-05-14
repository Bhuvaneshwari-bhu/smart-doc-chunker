"""
RAG chat engine: retrieval → context assembly → extraction-based answer synthesis.

Drop-in replacement point: swap generate_answer() with any LLM call
(OpenAI, Anthropic, local model) without touching answer_query().
"""

import re

from .embedding_store import generate_embedding, cosine_similarity

# ---------------------------------------------------------------------------
# Stopwords filtered out before keyword scoring
# ---------------------------------------------------------------------------

_STOPWORDS: frozenset[str] = frozenset({
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "are", "was", "were", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "will", "would",
    "could", "should", "may", "might", "shall", "can", "this", "that",
    "these", "those", "it", "its", "i", "we", "you", "he", "she", "they",
    "what", "which", "who", "how", "when", "where", "why", "not", "no",
    "as", "if", "so", "than", "then", "about", "into", "over", "after",
})

# Sentence boundary: split on ". ", "? ", "! " or end-of-string after punctuation
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")

_NO_MATCH_RESPONSE = "No relevant information found in the document."


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def answer_query(
    query: str,
    embedded_chunks: list[dict],
    top_k: int = 3,
    debug: bool = False,
) -> dict:
    """
    Full RAG pipeline: embed query → cosine retrieval → context assembly →
    extraction-based answer synthesis.

    Returns {query, answer, sources} where each source carries
    chunk_id, score, and text.
    """
    # Step 1 — embed the query with the same function used for chunks
    query_vec = generate_embedding(query)

    # Step 2 — score every chunk by cosine similarity
    scored: list[dict] = []
    for c in embedded_chunks:
        score = cosine_similarity(query_vec, c["embedding"])
        scored.append({
            "chunk_id": c["chunk_id"],
            "text":     c["text"],
            "score":    score,
        })

    scored.sort(key=lambda x: x["score"], reverse=True)
    results = scored[: min(top_k, len(scored))]

    # Step 3 — debug output
    if debug:
        print(f"[rag_chat] query: {query!r}")
        print(f"[rag_chat] top-{top_k} chunks selected:")
        for r in results:
            preview = r["text"][:70] + ("…" if len(r["text"]) > 70 else "")
            print(f"  chunk_id={r['chunk_id']}  score={r['score']:.4f}  {preview!r}")
        print(f"[rag_chat] similarity scores: {[round(r['score'], 4) for r in results]}")

    # Step 4 — build context window
    if results:
        context = "\n\n".join(
            f"[Chunk {r['chunk_id']}]: {r['text']}" for r in results
        )
    else:
        context = ""

    if debug:
        print(f"[rag_chat] final context:\n{context}\n")

    # Step 5 — synthesise answer
    answer = generate_answer(query, context)

    # Step 6 — assemble response
    return {
        "query":  query,
        "answer": answer,
        "sources": [
            {
                "chunk_id": r["chunk_id"],
                "score":    round(r["score"], 4),
                "text":     r["text"],
            }
            for r in results
        ],
    }


# ---------------------------------------------------------------------------
# Answer generation — extraction-based, no LLM
# ---------------------------------------------------------------------------

def generate_answer(query: str, context: str) -> str:
    """
    Extraction-based answer: score every sentence in context against query
    keywords, then return the top-scoring sentences in their original order
    as a coherent paragraph.

    No facts are invented — every word in the answer came from the context.
    """
    if not context or not context.strip():
        return _NO_MATCH_RESPONSE

    keywords = _extract_keywords(query)

    sentences = _split_sentences(context)
    if not sentences:
        return _NO_MATCH_RESPONSE

    # Score each sentence; track original index to preserve reading order
    scored: list[tuple[int, float, str]] = []
    for idx, sentence in enumerate(sentences):
        score = _score_sentence(sentence, keywords)
        if score > 0.0:
            scored.append((idx, score, sentence))

    if not scored:
        # No keyword overlap — fall back to the first two sentences verbatim
        fallback = [s for s in sentences[:2] if s.strip()]
        return " ".join(fallback) if fallback else _NO_MATCH_RESPONSE

    # Determine how many sentences to surface (at most 5, never more than half)
    max_sentences = max(1, min(5, len(scored) // 2 + 1))

    # Pick top-scoring sentences, then re-sort by original position for coherence
    top_scored = sorted(scored, key=lambda x: x[1], reverse=True)[:max_sentences]
    top_scored.sort(key=lambda x: x[0])

    answer = " ".join(item[2].strip() for item in top_scored)
    return _clean_answer(answer)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _extract_keywords(text: str) -> set[str]:
    """Lower-case tokens from text, stopwords removed, length >= 3."""
    tokens = re.findall(r"\b[a-zA-Z][a-zA-Z0-9]*\b", text.lower())
    return {t for t in tokens if t not in _STOPWORDS and len(t) >= 3}


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences; strip chunk header prefixes."""
    cleaned = re.sub(r"\[Chunk \d+\]:\s*", "", text)
    sentences = _SENTENCE_SPLIT.split(cleaned.strip())
    return [s.strip() for s in sentences if s.strip()]


def _score_sentence(sentence: str, keywords: set[str]) -> float:
    """
    Keyword coverage score: fraction of query keywords present in sentence,
    boosted by term frequency of matching keywords.

    score = coverage + tf_bonus * 0.5
    """
    if not keywords:
        return 0.0

    tokens = re.findall(r"\b[a-zA-Z][a-zA-Z0-9]*\b", sentence.lower())
    token_set = set(tokens)

    matched = keywords & token_set
    if not matched:
        return 0.0

    coverage = len(matched) / len(keywords)
    tf_sum   = sum(tokens.count(kw) for kw in matched)
    tf_bonus = tf_sum / max(len(tokens), 1)

    return coverage + tf_bonus * 0.5


def _clean_answer(text: str) -> str:
    """Normalise whitespace and ensure the answer ends with punctuation."""
    text = re.sub(r"[ \t]+", " ", text).strip()
    text = re.sub(r"\s*\n\s*", " ", text)
    if text and text[-1] not in ".!?":
        text += "."
    return text
