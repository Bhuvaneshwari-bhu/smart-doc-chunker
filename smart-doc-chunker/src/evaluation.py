from . import embedding_store


def evaluate_system(
    golden_set: list[dict],
    embedded_chunks: list[dict],
    top_k: int = 3,
) -> dict:
    if not golden_set:
        return {
            "precision_at_k": 0.0,
            "recall_at_k":    0.0,
            "avg_hit_rate":   0.0,
            "per_query_results": [],
        }

    per_query: list[dict] = []

    for item in golden_set:
        query    = item["query"]
        expected = set(item["expected_chunk_ids"])

        results     = embedding_store.search(query, embedded_chunks, top_k=top_k)
        retrieved   = [r["chunk_id"] for r in results]
        retrieved_k = retrieved[:top_k]

        prec = precision_at_k(expected, retrieved_k)
        rec  = recall_at_k(expected, retrieved_k)

        per_query.append({
            "query":     query,
            "precision": prec,
            "recall":    rec,
        })

    n              = len(per_query)
    macro_prec     = round(sum(r["precision"] for r in per_query) / n, 4)
    macro_recall   = round(sum(r["recall"]    for r in per_query) / n, 4)
    avg_hit        = round(hit_rate(golden_set, embedded_chunks, top_k), 4)

    return {
        "precision_at_k":    macro_prec,
        "recall_at_k":       macro_recall,
        "avg_hit_rate":      avg_hit,
        "per_query_results": per_query,
    }


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def precision_at_k(expected: set[int], retrieved: list[int]) -> float:
    """Fraction of retrieved chunks that are relevant."""
    if not retrieved:
        return 0.0
    hits = sum(1 for cid in retrieved if cid in expected)
    return round(hits / len(retrieved), 4)


def recall_at_k(expected: set[int], retrieved: list[int]) -> float:
    """Fraction of relevant chunks that were retrieved."""
    if not expected:
        return 0.0
    hits = sum(1 for cid in retrieved if cid in expected)
    return round(hits / len(expected), 4)


def hit_rate(
    golden_set: list[dict],
    embedded_chunks: list[dict],
    top_k: int,
) -> float:
    """Fraction of queries where at least one relevant chunk was retrieved."""
    if not golden_set:
        return 0.0
    hits = 0
    for item in golden_set:
        expected  = set(item["expected_chunk_ids"])
        results   = embedding_store.search(item["query"], embedded_chunks, top_k=top_k)
        retrieved = {r["chunk_id"] for r in results}
        if expected & retrieved:
            hits += 1
    return hits / len(golden_set)
