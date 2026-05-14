def count_tokens(text: str) -> int:
    return len(text.split())


def analyze_chunks(chunks: list[dict]) -> dict:
    if not chunks:
        return {
            "total_chunks": 0,
            "total_tokens": 0,
            "avg_tokens_per_chunk": 0.0,
            "min_tokens": 0,
            "max_tokens": 0,
            "token_distribution": [],
        }

    distribution = [
        c["tokens"] if "tokens" in c else count_tokens(c.get("text", ""))
        for c in chunks
    ]

    total_tokens = sum(distribution)
    total_chunks = len(distribution)

    return {
        "total_chunks": total_chunks,
        "total_tokens": total_tokens,
        "avg_tokens_per_chunk": round(total_tokens / total_chunks, 4),
        "min_tokens": min(distribution),
        "max_tokens": max(distribution),
        "token_distribution": distribution,
    }
