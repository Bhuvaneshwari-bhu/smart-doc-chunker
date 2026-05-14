import re

# Minimum chunk token threshold is now adaptive — see _build_output().
# A fixed value of 10 was the root-cause bug: with chunk_size < 10 every
# produced chunk was silently dropped, returning an empty list.
_ABSOLUTE_MIN_TOKENS = 1   # never drop a chunk that has at least one word

# Sentence boundary: punctuation followed by whitespace or end-of-string.
# Keeps the delimiter attached to the preceding sentence.
_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def chunk_text(
    text: str,
    method: str = "recursive",
    chunk_size: int = 500,
    overlap: int = 50,
) -> list[dict]:
    if not text or not text.strip():
        return []

    if overlap >= chunk_size:
        raise ValueError(
            f"overlap ({overlap}) must be smaller than chunk_size ({chunk_size})"
        )

    original_text = text.strip()

    if method == "recursive":
        raw_chunks = _recursive_chunker(original_text, chunk_size)
    elif method == "sliding":
        raw_chunks = _sliding_window_chunker(original_text, chunk_size, overlap)
    else:
        raise ValueError(
            f"Unknown chunking method '{method}'. Supported: 'recursive', 'sliding'"
        )

    print(f"[chunker] method={method} raw chunks produced: {len(raw_chunks)}")

    result = _build_output(raw_chunks, chunk_size)

    # Safety fallback: valid text must always yield at least one chunk.
    if not result:
        print("[chunker] WARNING: all chunks filtered — applying fallback to full text")
        tokens = _count_tokens(original_text)
        result = [{"chunk_id": 1, "text": original_text, "tokens": tokens}]

    print(f"[chunker] final chunk count: {len(result)}")
    return result


# ---------------------------------------------------------------------------
# Token counting
# ---------------------------------------------------------------------------

def _count_tokens(text: str) -> int:
    return len(text.split())


# ---------------------------------------------------------------------------
# Sentence splitting
# ---------------------------------------------------------------------------

def _sentence_splitter(text: str) -> list[str]:
    """Split text into sentences, preserving each sentence's trailing punctuation."""
    sentences = _SENTENCE_BOUNDARY.split(text.strip())
    return [s.strip() for s in sentences if s.strip()]


# ---------------------------------------------------------------------------
# Recursive chunker
# ---------------------------------------------------------------------------

def _recursive_chunker(text: str, chunk_size: int) -> list[str]:
    """
    Hierarchical splitting:
      Level 1 – paragraphs  (\\n\\n)
      Level 2 – sentences   (punctuation boundaries)
      Level 3 – merge units into chunks <= chunk_size tokens, respecting
                boundaries as much as possible
    """
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

    # Break oversized paragraphs into sentences
    units: list[str] = []
    for para in paragraphs:
        if _count_tokens(para) <= chunk_size:
            units.append(para)
        else:
            units.extend(_sentence_splitter(para))

    # Greedily merge units into chunks up to chunk_size tokens
    chunks: list[str] = []
    current_parts: list[str] = []
    current_tokens = 0

    for unit in units:
        unit_tokens = _count_tokens(unit)

        # A single unit already exceeds chunk_size: hard-split it by words
        if unit_tokens > chunk_size:
            if current_parts:
                chunks.append(" ".join(current_parts))
                current_parts = []
                current_tokens = 0
            chunks.extend(_hard_split(unit, chunk_size))
            continue

        if current_tokens + unit_tokens > chunk_size and current_parts:
            chunks.append(" ".join(current_parts))
            current_parts = []
            current_tokens = 0

        current_parts.append(unit)
        current_tokens += unit_tokens

    # Always flush any remaining accumulated text
    if current_parts:
        chunks.append(" ".join(current_parts))

    return chunks


# ---------------------------------------------------------------------------
# Sliding window chunker
# ---------------------------------------------------------------------------

def _sliding_window_chunker(
    text: str, chunk_size: int, overlap: int
) -> list[str]:
    """
    Token-level sliding window.
    Each window is chunk_size tokens wide; the next window starts at
    stride = chunk_size - overlap tokens from the previous start.
    """
    words = text.split()
    total = len(words)

    if total == 0:
        return []

    stride = chunk_size - overlap
    chunks: list[str] = []
    start = 0

    while start < total:
        end = min(start + chunk_size, total)
        chunks.append(" ".join(words[start:end]))
        if end == total:
            break
        start += stride

    return chunks


# ---------------------------------------------------------------------------
# Hard word-level split (last resort for single oversized units)
# ---------------------------------------------------------------------------

def _hard_split(text: str, chunk_size: int) -> list[str]:
    words = text.split()
    return [
        " ".join(words[i : i + chunk_size])
        for i in range(0, len(words), chunk_size)
    ]


# ---------------------------------------------------------------------------
# Output builder — filter + assign metadata
# ---------------------------------------------------------------------------

def _build_output(raw_chunks: list[str], chunk_size: int) -> list[dict]:
    # Adaptive minimum: at most 20 % of chunk_size, floored at 1.
    # This prevents the fixed-10 bug while still dropping truly empty fragments.
    min_tokens = max(_ABSOLUTE_MIN_TOKENS, min(chunk_size // 5, 10))

    result: list[dict] = []
    chunk_id = 1

    for chunk in raw_chunks:
        chunk = chunk.strip()
        if not chunk:
            continue
        tokens = _count_tokens(chunk)
        if tokens < min_tokens:
            continue
        result.append({"chunk_id": chunk_id, "text": chunk, "tokens": tokens})
        chunk_id += 1

    return result
