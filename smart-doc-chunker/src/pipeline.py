import json
import os
import sys
from datetime import datetime

from . import loader
from . import cleaner
from . import chunker
from . import tokenizer


def run(config: dict) -> None:
    input_path = config["input"]
    method     = config["method"]
    chunk_size = config["chunk_size"]
    overlap    = config["overlap"]

    # ── 1. Load ──────────────────────────────────────────────────────────────
    print("Loading document...")
    try:
        raw_text = loader.load_document(input_path)
    except (FileNotFoundError, ValueError) as exc:
        print(f"[Pipeline error] Load failed: {exc}", file=sys.stderr)
        sys.exit(1)

    # ── 2. Clean ─────────────────────────────────────────────────────────────
    print("Cleaning text...")
    cleaned_text = cleaner.clean_text(raw_text)
    if not cleaned_text.strip():
        print("[Pipeline error] Text is empty after cleaning.", file=sys.stderr)
        sys.exit(1)

    # ── 3. Chunk ─────────────────────────────────────────────────────────────
    print(f"Chunking using {method}...")
    chunks = chunker.chunk_text(
        text=cleaned_text,
        method=method,
        chunk_size=chunk_size,
        overlap=overlap,
    )
    if not chunks:
        print("[Pipeline error] Chunking produced no output.", file=sys.stderr)
        sys.exit(1)
    print(f"Generated {len(chunks)} chunks")

    # ── 4. Analyze ───────────────────────────────────────────────────────────
    stats = tokenizer.analyze_chunks(chunks)
    print("Token stats computed")

    # ── 5. Build dataset artifact ────────────────────────────────────────────
    dataset = {
        "source_file": input_path,
        "method":      method,
        "chunk_size":  chunk_size,
        "overlap":     overlap,
        "chunks":      chunks,
        "stats":       stats,
    }

    # ── 6. Save output ───────────────────────────────────────────────────────
    output_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "output",
    )
    os.makedirs(output_dir, exist_ok=True)

    timestamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(output_dir, f"chunks_{timestamp}.json")

    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(dataset, fh, indent=2, ensure_ascii=False)

    print(f"Saved output to {output_path}")
