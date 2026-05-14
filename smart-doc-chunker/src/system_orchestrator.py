"""
RAGSystem — facade over the full Smart Document Chunking + RAG stack.

Wires together:
  loader · cleaner · chunker · tokenizer
  embedding_provider · embedding_store.cosine_similarity
  rag_chat.generate_answer
  evaluation.precision_at_k · recall_at_k · hit_rate
  metrics.Metrics

embedding_store.search() and evaluation.evaluate_system() are intentionally
bypassed here because both hardcode the mock generate_embedding (128-dim).
Using them with LocalEmbeddingProvider (384-dim) would raise a dimension
mismatch. The orchestrator implements provider-aware retrieval with
embedding_store.cosine_similarity, then delegates to the leaf functions of
rag_chat and evaluation — no logic is duplicated.
"""

from src import loader
from src import cleaner
from src import chunker
from src import tokenizer
from src import embedding_store
from src import rag_chat
from src import evaluation
from src.metrics import Metrics
from src.embedding_provider import get_embedding_provider, EmbeddingProvider


class RAGSystem:

    def __init__(self, embedding_mode: str = "mock") -> None:
        self._provider:        EmbeddingProvider = get_embedding_provider(embedding_mode)
        self._embedded_chunks: list[dict]        = []
        self._metrics:         Metrics           = Metrics()

    # ── document ingestion ────────────────────────────────────────────────────

    def process_document(
        self,
        file_path:  str,
        method:     str = "recursive",
        chunk_size: int = 500,
        overlap:    int = 50,
    ) -> dict:
        """
        Load → clean → chunk → embed.
        Returns tokenizer stats and stores embedded chunks internally.
        Raises ValueError / FileNotFoundError on bad input.
        """
        self._metrics.start_timer("process_document")

        raw_text     = loader.load_document(file_path)
        cleaned_text = cleaner.clean_text(raw_text)

        if not cleaned_text.strip():
            raise ValueError("Document is empty after cleaning.")

        chunks = chunker.chunk_text(
            text=cleaned_text,
            method=method,
            chunk_size=chunk_size,
            overlap=overlap,
        )

        if not chunks:
            raise ValueError("Chunking produced no output.")

        stats = tokenizer.analyze_chunks(chunks)

        self._metrics.start_timer("embedding")
        embedded = [
            {
                "chunk_id":  c["chunk_id"],
                "text":      c["text"],
                "tokens":    c["tokens"],
                "embedding": self._provider.embed(c["text"]),
            }
            for c in chunks
        ]
        embedding_ms = self._metrics.end_timer("embedding")

        self._embedded_chunks = embedded

        process_ms = self._metrics.end_timer("process_document")
        self._metrics.log_request({
            "event":          "process_document",
            "file_path":      file_path,
            "method":         method,
            "chunk_count":    len(chunks),
            "latency_ms":     round(process_ms, 3),
            "embedding_ms":   round(embedding_ms, 3),
        })

        return stats

    # ── retrieval-augmented QA ────────────────────────────────────────────────

    def ask(self, query: str, top_k: int = 3) -> dict:
        """
        Retrieve the top_k most relevant chunks then synthesise an answer.
        Returns {query, answer, sources}.
        """
        if not query or not query.strip():
            raise ValueError("query must not be empty")
        if not self._embedded_chunks:
            raise RuntimeError("No document loaded. Call process_document() first.")

        self._metrics.start_timer("retrieval")
        results = self._retrieve(query, top_k)
        retrieval_ms = self._metrics.end_timer("retrieval")

        context = "\n\n".join(
            f"[Chunk {c['chunk_id']}]: {c['text']}" for c in results
        )
        answer = rag_chat.generate_answer(query, context)

        response = {
            "query":   query,
            "answer":  answer,
            "sources": [{"chunk_id": c["chunk_id"], "text": c["text"]} for c in results],
        }

        self._metrics.log_request({
            "event":        "ask",
            "top_k":        top_k,
            "retrieval_ms": round(retrieval_ms, 3),
        })

        return response

    # ── evaluation ────────────────────────────────────────────────────────────

    def evaluate(self, golden_set: list[dict], top_k: int = 3) -> dict:
        """
        Score retrieval quality against a golden set of (query, expected_chunk_ids).
        Returns precision@k, recall@k, avg_hit_rate, and per-query breakdown.
        """
        if not golden_set:
            return {
                "precision_at_k": 0.0,
                "recall_at_k":    0.0,
                "avg_hit_rate":   0.0,
                "per_query_results": [],
            }
        if not self._embedded_chunks:
            raise RuntimeError("No document loaded. Call process_document() first.")

        per_query: list[dict] = []
        hit_count = 0

        for item in golden_set:
            expected  = set(item["expected_chunk_ids"])
            results   = self._retrieve(item["query"], top_k)
            retrieved = [r["chunk_id"] for r in results]

            prec = evaluation.precision_at_k(expected, retrieved)
            rec  = evaluation.recall_at_k(expected, retrieved)

            if expected & set(retrieved):
                hit_count += 1

            per_query.append({"query": item["query"], "precision": prec, "recall": rec})

        n = len(per_query)
        report = {
            "precision_at_k":    round(sum(r["precision"] for r in per_query) / n, 4),
            "recall_at_k":       round(sum(r["recall"]    for r in per_query) / n, 4),
            "avg_hit_rate":      round(hit_count / n, 4),
            "per_query_results": per_query,
        }

        self._metrics.log_request({
            "event":          "evaluate",
            "queries_tested": n,
            "precision_at_k": report["precision_at_k"],
            "recall_at_k":    report["recall_at_k"],
            "avg_hit_rate":   report["avg_hit_rate"],
        })

        return report

    # ── observability ─────────────────────────────────────────────────────────

    def get_metrics(self) -> dict:
        return self._metrics.get_summary()

    # ── provider-aware retrieval (internal) ───────────────────────────────────

    def _retrieve(self, query: str, top_k: int) -> list[dict]:
        """
        Embed the query with the configured provider, score all chunks by
        cosine similarity, return the top_k results descending by score.
        """
        query_vec = self._provider.embed(query)
        scored = [
            {
                "chunk_id": c["chunk_id"],
                "text":     c["text"],
                "tokens":   c["tokens"],
                "score":    embedding_store.cosine_similarity(query_vec, c["embedding"]),
            }
            for c in self._embedded_chunks
        ]
        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[: min(top_k, len(scored))]
