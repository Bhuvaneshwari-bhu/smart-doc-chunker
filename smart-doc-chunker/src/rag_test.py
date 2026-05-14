"""
Standalone RAG validation script.

Usage:
    python src/rag_test.py --json output/chunks_<timestamp>.json
    python src/rag_test.py --json output/chunks_<timestamp>.json --top_k 5

Loads a chunker output JSON file, embeds every chunk, then runs a fixed
battery of test queries through the retrieval loop and prints ranked results.
"""

import argparse
import json
import math
import os
import sys

from src.embedding_store import generate_embedding, cosine_similarity

# ---------------------------------------------------------------------------
# Hardcoded test queries
# ---------------------------------------------------------------------------

TEST_QUERIES: list[str] = [
    "test document",
    "chunking system",
    "multiple sentences",
    "machine learning",
    "supervised learning labeled data",
]

# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------

def load_chunks(json_path: str) -> list[dict]:
    if not os.path.isfile(json_path):
        raise FileNotFoundError(f"JSON file not found: {json_path}")
    with open(json_path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    # Accept both the full pipeline artifact {chunks: [...]} and a bare list
    if isinstance(data, list):
        chunks = data
    elif isinstance(data, dict) and "chunks" in data:
        chunks = data["chunks"]
    else:
        raise ValueError(
            "Unrecognised JSON structure. Expected a list of chunks or "
            "a pipeline artifact dict with a 'chunks' key."
        )
    if not chunks:
        raise ValueError("JSON file contains no chunks.")
    return chunks


def embed_chunks(chunks: list[dict]) -> list[dict]:
    embedded = []
    for c in chunks:
        vec = generate_embedding(c["text"])
        embedded.append({
            "chunk_id": c["chunk_id"],
            "text":     c["text"],
            "embedding": vec,
        })
    return embedded


def retrieve(
    query: str,
    embedded_chunks: list[dict],
    top_k: int,
) -> list[dict]:
    query_vec = generate_embedding(query)

    print(f"  query embedding (first 6 dims): "
          f"[{', '.join(f'{v:.4f}' for v in query_vec[:6])} ...]")

    scored = [
        {
            "chunk_id": c["chunk_id"],
            "score":    cosine_similarity(query_vec, c["embedding"]),
            "text":     c["text"],
        }
        for c in embedded_chunks
    ]
    scored.sort(key=lambda x: x["score"], reverse=True)
    top = scored[: min(top_k, len(scored))]

    print(f"  top similarity scores: "
          f"{[round(r['score'], 4) for r in top]}")

    return top


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="RAG retrieval validation against a chunker output JSON file."
    )
    parser.add_argument(
        "--json",
        required=True,
        metavar="FILE",
        help="Path to the chunker output JSON file.",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=3,
        metavar="K",
        help="Number of top results to return per query (default: 3).",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    print(f"\n{'='*60}")
    print(f"  RAG Validation Test")
    print(f"  file  : {args.json}")
    print(f"  top_k : {args.top_k}")
    print(f"{'='*60}\n")

    # ── 1. Load chunks ───────────────────────────────────────────────────────
    print(f"[1/3] Loading chunks from {args.json} ...")
    chunks = load_chunks(args.json)
    print(f"      Loaded {len(chunks)} chunk(s).\n")

    # ── 2. Embed chunks ──────────────────────────────────────────────────────
    print("[2/3] Generating embeddings for all chunks ...")
    embedded = embed_chunks(chunks)
    print(f"      Embedded {len(embedded)} chunk(s). "
          f"Dimension: {len(embedded[0]['embedding'])}.\n")

    # ── 3. Query loop ────────────────────────────────────────────────────────
    print("[3/3] Running test queries ...\n")

    all_results: dict[str, list[dict]] = {}

    for query in TEST_QUERIES:
        print(f"  Query : \"{query}\"")
        results = retrieve(query, embedded, args.top_k)
        all_results[query] = results

        for rank, r in enumerate(results, start=1):
            score_bar = "█" * int(r["score"] * 20)
            preview   = r["text"][:80] + ("…" if len(r["text"]) > 80 else "")
            print(f"    [{rank}] chunk_id={r['chunk_id']}  "
                  f"score={r['score']:.4f}  {score_bar}")
            print(f"        {preview}")
        print()

    # ── Summary ──────────────────────────────────────────────────────────────
    print(f"{'='*60}")
    print(f"  Summary")
    print(f"{'='*60}")
    print(f"  Chunks embedded : {len(embedded)}")
    print(f"  Queries tested  : {len(TEST_QUERIES)}")
    print(f"  top_k           : {args.top_k}")

    all_scores = [r["score"] for results in all_results.values() for r in results]
    if all_scores:
        print(f"  Score range     : {min(all_scores):.4f} – {max(all_scores):.4f}")
        print(f"  Mean top score  : "
              f"{sum(r['score'] for results in all_results.values() for r in results[:1]) / len(TEST_QUERIES):.4f}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
