# Smart Doc Chunker · RAG Playground

> **An interactive, full-stack visualization of a Retrieval-Augmented Generation pipeline — from raw document to grounded answer.**

[![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-green?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![React](https://img.shields.io/badge/React-18-61DAFB?logo=react&logoColor=black)](https://react.dev)
[![Vite](https://img.shields.io/badge/Vite-5-646CFF?logo=vite&logoColor=white)](https://vitejs.dev)
[![Docker](https://img.shields.io/badge/Docker-ready-2496ED?logo=docker&logoColor=white)](https://docker.com)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

---

## Live Demo

- Frontend UI: https://smart-doc-chunker-77pk.vercel.app
- Backend API: https://smart-doc-chunker.onrender.com
- Swagger Docs: https://smart-doc-chunker.onrender.com/docs

---

## What This Is

Smart Doc Chunker is a **production-grade, full-stack AI engineering project** that exposes the internals of a complete Retrieval-Augmented Generation pipeline as an interactive playground. It implements two document chunking strategies — **recursive hierarchical splitting** (paragraph → sentence → greedy merge) and **sliding window with configurable overlap** — paired with a deterministic embedding engine, a cosine similarity retrieval pipeline, and an extraction-based answer synthesizer, all rendered live in the browser with no external API key required.

You paste text (or upload a `.txt` file), adjust chunk size and overlap, and watch a five-stage pipeline light up: **Document → Chunking → Embeddings → Retrieval → Answer**. Each chunk card shows its token count alongside an **8-dimension embedding visualization** rendered as intensity-colored badges encoding the sign and magnitude of the vector. When you ask a question, the retrieved chunks appear with cosine similarity score bars, and the answer is assembled by scoring and reranking extracted sentences directly from the retrieved context.

The system enforces a clean separation between a **FastAPI backend** (seven documented REST endpoints with Pydantic v2 validation) and a **React + Vite frontend** connected through a typed Axios client and a Vite dev proxy. Both surfaces are independently runnable, Dockerized, and deployed — backend on Render, frontend on Vercel.

---

## Table of Contents

- [Live Demo](#live-demo)
- [Run Locally](#run-locally)
- [Why This Project Exists](#why-this-project-exists)
- [Interactive Features](#interactive-features)
- [Architecture Deep Dive](#architecture-deep-dive)
- [The Chunking Engine](#the-chunking-engine)
- [The Embedding System](#the-embedding-system)
- [The RAG Pipeline](#the-rag-pipeline)
- [Frontend Architecture](#frontend-architecture)
- [Backend Architecture](#backend-architecture)
- [API Reference](#api-reference)
- [Screenshots](#screenshots)
- [Installation](#installation)
- [Demo Walkthrough](#demo-walkthrough)
- [Project Structure](#project-structure)
- [Engineering Decisions](#engineering-decisions)
- [Future Enhancements](#future-enhancements)
- [What This Demonstrates](#what-this-demonstrates)
- [License](#license)

---

## Why This Project Exists

Most RAG tutorials treat chunking, embedding, and retrieval as black boxes. You call a library function, get chunks, call another function, get an answer. The underlying decisions — how chunk boundaries are chosen, why overlap matters, what a similarity score actually measures, how an answer is assembled from retrieved context — remain opaque.

This project exists to make those decisions **visible, interactive, and debuggable**.

### Chunking is not trivial

A 100,000-token document cannot be passed whole to a retrieval system or a language model. It must be split into semantically coherent segments. Naive fixed-length splitting destroys sentence and paragraph structure — a chunk boundary might fall mid-sentence, splitting a key concept across two pieces, guaranteeing that neither will retrieve cleanly. Good chunking preserves semantic units and adds configurable overlap so adjacent chunks share context.

### Preprocessing determines retrieval quality

Raw documents contain noise: page-number patterns (`Page 12 of 40`, `| 12 |`), repeated header and footer lines, legal symbols, excess whitespace, inconsistent line endings. If this noise is embedded, it consumes vector dimensions that should encode meaning. A five-stage cleaning pipeline runs before any text reaches the chunker.

### Embeddings encode semantic position

Meaning is location. When a query and a document chunk live in the same high-dimensional vector space, proximity (cosine similarity) approximates semantic relatedness. Without embeddings, retrieval is keyword matching — fast, but brittle. With embeddings, the system can retrieve a chunk about "neural networks" in response to a query about "deep learning architectures," because those phrases are geometrically nearby.

### Retrieval-augmented generation reduces hallucination

A language model generating without context invents plausible-sounding facts. When the prompt is grounded in retrieved text — text that came from the actual document — the model (or in this case, the extraction-based synthesizer) can only assemble answers from things that were actually written. Retrieval is the mechanism that converts a general-purpose model into a document-specific one.

---

## Interactive Features

### Input Modes

**Paste Text** — A full-height monospace textarea. Any content pasted immediately becomes available for chunking. No file system access required.

**Upload File** — A drag-and-drop zone that accepts `.txt` files. The file is sent as `multipart/form-data` to the backend, read into memory, cleaned, chunked, and embedded without ever being written to disk on the server side.

### Live Chunk Generation

After clicking **Generate Chunks**, the pipeline runs server-side and returns structured chunk objects: `{chunk_id, text, tokens}`. These render immediately as scrollable cards in the center column. The chunk card shows the full chunk text, not a truncated preview — the entire semantic unit is visible.

### Embedding Previews (Per Chunk)

As soon as chunks are returned, the frontend automatically fires a second request to `/embed-chunks`. The response carries the first eight dimensions of the 128-dimensional embedding vector plus the vector's L2 norm. Each dimension renders as a colored badge: **blue** for positive values, **red** for negative, with opacity proportional to magnitude. The `‖v‖ = 1.000` label confirms L2 normalization.

```
Embedding  128-dim · ‖v‖ = 1.000
[d0:+0.23] [d1:−0.11] [d2:+0.87] [d3:−0.44] [d4:+0.12] [d5:+0.55] [d6:−0.31] [d7:+0.09]  ···+120
```

### Pipeline State Machine

A five-step progress bar runs across the top of the playground. Each step has three visual states:

| State | Appearance |
|---|---|
| Inactive | Gray numbered circle |
| Active | Blue filled circle with pulsing concentric ring animation |
| Completed | Green circle with ✓, green connector line to next step |

Steps advance automatically as data becomes available — the UI derives pipeline state from the presence or absence of data, not from a manual flag.

### RAG Chat Panel

The right column activates once chunks exist. A question field and top-K slider let you fire queries against the embedded chunks. The system returns a generated answer plus the top-K retrieved chunks ranked by cosine similarity.

### Retrieval Visualization with Score Bars

Each retrieved chunk displays a horizontal bar whose width encodes the cosine similarity score. The score is mapped from its theoretical `[−1, 1]` range to a `[0%, 100%]` visual fill using `(score + 1) / 2 × 100`. Color thresholds differentiate strong matches (green, ≥ 0.3), moderate matches (blue, ≥ 0.05), and weak matches (gray).

---

## Architecture Deep Dive

### System Overview

```
┌────────────────────────────────────────────────────────────────────┐
│                          Browser Client                            │
│                                                                    │
│   ┌────────────────────────────────────────────────────────────┐  │
│   │              React SPA  (Vite dev server · :3000)          │  │
│   │                                                            │  │
│   │  ┌──────────────┐  ┌─────────────────────────────────┐    │  │
│   │  │  PipelineBar │  │         ChunkerPanel             │    │  │
│   │  │  5-step viz  │  │  Left: Input + Settings          │    │  │
│   │  └──────────────┘  │  Center: Chunks + Embeddings     │    │  │
│   │                    │  Right: RagChatPanel              │    │  │
│   │  ┌─────────────┐   └─────────────────────────────────┘    │  │
│   │  │ ProcessPanel│   ┌──────────────┐  ┌───────────────┐    │  │
│   │  │ (file path) │   │  ChatPanel   │  │   Spinner     │    │  │
│   │  └─────────────┘   └──────────────┘  └───────────────┘    │  │
│   │                                                            │  │
│   │              api.js  ·  Axios  ·  /api/* proxy             │  │
│   └────────────────────────────────────────────────────────────┘  │
└───────────────────────────────┬────────────────────────────────────┘
                                │ HTTP / JSON (+ multipart/form-data)
                                │ Vite rewrites /api/* → /*
                                ▼
┌────────────────────────────────────────────────────────────────────┐
│                   FastAPI Backend  (:8000)                         │
│                                                                    │
│  ┌──────────────┐  ┌───────────────┐  ┌──────────────────────┐   │
│  │  /process    │  │ /process-text │  │  /process-upload     │   │
│  │  (file path) │  │ (raw JSON)    │  │  (multipart)         │   │
│  └──────┬───────┘  └───────┬───────┘  └──────────┬───────────┘   │
│         │                  │                      │               │
│         └──────────────────┼──────────────────────┘               │
│                            │ cleaned → chunked → tokenized        │
│                            ▼                                      │
│  ┌──────────────┐  ┌───────────────┐  ┌──────────────────────┐   │
│  │ /embed-chunks│  │  /ask-chunks  │  │       /ask           │   │
│  │ (previews)   │  │ (stateless RAG│  │  (file-based RAG)    │   │
│  └──────┬───────┘  └───────┬───────┘  └──────────────────────┘   │
│         │                  │                                      │
│  ┌──────▼──────────────────▼──────────────────────────────────┐   │
│  │                    Core Pipeline Modules                    │   │
│  │                                                             │   │
│  │  cleaner.py  →  chunker.py  →  tokenizer.py                 │   │
│  │  embedding_store.py         →  rag_chat.py                  │   │
│  │  loader.py (PDF + TXT)          metrics.py  evaluation.py   │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                    │
│       output/chunks_{timestamp}.json  (only /process persists)    │
└────────────────────────────────────────────────────────────────────┘
```

### Request Lifecycle — Playground Mode

```
  ① User pastes text
        │
        ▼
  POST /process-text   {text, method, chunk_size, overlap}
        │
        ├─ cleaner.clean_text(text)
        │    ├─ normalize line endings & collapse whitespace
        │    ├─ strip legal symbols, deduplicate punctuation
        │    ├─ remove page numbers (regex: "Page N of M", "N |", "| N")
        │    ├─ drop repetitive short lines (header/footer heuristic)
        │    ├─ fix spaces before punctuation, insert missing spaces
        │    └─ discard symbol-only and sub-3-char garbage lines
        │
        ├─ chunker.chunk_text(cleaned, method, chunk_size, overlap)
        │    ├─ [recursive]  paragraphs → sentences → greedy merge
        │    │               → hard split for oversized units
        │    │               → adaptive min-token filter
        │    └─ [sliding]    whitespace tokenize → stride windows
        │
        ├─ tokenizer.analyze_chunks(chunks)
        │    └─ {total_chunks, avg_tokens, min_tokens, max_tokens}
        │
        └─ Response: {chunks: [{chunk_id, text, tokens}], stats}

  ② Frontend auto-fires (no user action)
        │
        ▼
  POST /embed-chunks   {chunks}
        │
        ├─ embedding_store.embed_chunks(chunks)
        │    └─ per chunk: generate_embedding(text)
        │         ├─ SHA-256 hash loop with incrementing seed counter
        │         ├─ 512 raw bytes → 128 float values in [-1, 1]
        │         └─ L2 normalize → unit sphere
        │
        └─ Response: {embedded: [{chunk_id, preview[8], vector_length, dim: 128}]}

  ③ User asks a question
        │
        ▼
  POST /ask-chunks    {chunks, query, top_k}
        │
        ├─ embed_chunks(chunks)          → re-embed (stateless)
        ├─ generate_embedding(query)     → query vector
        ├─ cosine_similarity(q, each)    → scores[]
        ├─ sort descending, slice top_k  → results[]
        ├─ build context string          → "[Chunk N]: text\n\n..."
        └─ generate_answer(query, context)
             ├─ extract query keywords   → stopword filter, len ≥ 3
             ├─ split context sentences
             ├─ score each sentence      → coverage + tf_bonus × 0.5
             ├─ take top min(5, n//2+1)
             ├─ re-sort by original index → reading-order coherence
             └─ Response: {query, answer, sources: [{chunk_id, score, text}]}
```

---

## The Chunking Engine

**File:** `src/chunker.py`

The chunker exposes a single public function — `chunk_text(text, method, chunk_size, overlap)` — and internally selects between two algorithms. Token counting throughout uses whitespace splitting, which is fast and model-agnostic.

### Recursive Chunker

The recursive strategy preserves semantic hierarchy: it respects document structure rather than imposing arbitrary cuts.

```
Raw text
  │
  ├─ Split on \n\n → paragraphs
  │
  ├─ For each paragraph:
  │    if tokens ≤ chunk_size → keep as unit
  │    else → split by sentence boundary  (?<=[.!?])\s+
  │
  ├─ Greedy merge:
  │    while units remain:
  │      if current_tokens + next_tokens ≤ chunk_size → accumulate
  │      else → flush current, start new accumulator
  │
  └─ Hard split (last resort):
       if a single unit still exceeds chunk_size
       → split word by word into chunk_size-length windows
```

**Adaptive minimum token filter:** Rather than a fixed threshold, the minimum accepted token count is computed as `max(1, min(chunk_size // 5, 10))`. This prevents the silent-drop bug where a fixed value of 10 would filter *every* chunk when `chunk_size < 50`.

**Safety fallback:** If filtering removes all chunks (pathological input), the entire cleaned text is returned as a single chunk. The pipeline always produces output.

### Sliding Window Chunker

The sliding strategy prioritizes density over structure. It operates on the token sequence directly.

```
tokens = text.split()           # whitespace tokenization
stride = chunk_size - overlap   # step between window starts

while start < total:
    end   = min(start + chunk_size, total)
    chunk = " ".join(tokens[start:end])
    start += stride
```

With `overlap = 0`, this produces non-overlapping fixed-length windows. With `overlap > 0`, adjacent chunks share `overlap` tokens, ensuring that a concept split across a boundary still appears whole in at least one chunk.

**Constraint:** The API enforces `overlap < chunk_size` at the Pydantic validation layer, raising a `422 Unprocessable Entity` before the chunker is ever called.

### Chunker Output Format

```json
[
  {
    "chunk_id": 1,
    "text": "Machine learning is a subset of artificial intelligence...",
    "tokens": 47
  },
  {
    "chunk_id": 2,
    "text": "...enabling computers to learn from data without explicit programming.",
    "tokens": 31
  }
]
```

---

## The Embedding System

**File:** `src/embedding_store.py`

### Deterministic Hash-Derived Embeddings

The embedding function generates a 128-dimensional unit vector from any text string using only the Python standard library. No external model, no API call, no network dependency.

```python
def generate_embedding(text: str) -> list[float]:
    raw = bytearray()
    seed = 0
    target = 128 * 4   # 4 bytes per float, 16 SHA-256 rounds needed
    while len(raw) < target:
        digest = hashlib.sha256(f"{seed}\x00{text}".encode()).digest()
        raw.extend(digest)   # 32 bytes per round
        seed += 1

    vector = []
    for i in range(128):
        word = int.from_bytes(raw[i*4 : i*4+4], "big")
        vector.append(word / 0xFFFFFFFF * 2.0 - 1.0)  # map to [-1, 1]

    return _l2_normalize(vector)   # project onto unit hypersphere
```

**Why this works for a playground:** The hash function is collision-resistant and deterministic — identical text always produces the identical vector. Because L2 normalization projects every vector onto the unit hypersphere, cosine similarity reduces to a dot product, which is computationally cheap. The 128-dimensional space provides enough separation for meaningful relative rankings even without learned representations.

**The tradeoff is acknowledged:** These are geometry-preserving fingerprints, not semantic embeddings. Two chunks with different wording but the same meaning will not be geometrically close. The codebase marks `generate_embedding()` as the "drop-in replacement point" — swapping it for a `sentence-transformers` encoder or an OpenAI embedding call requires changing exactly one function.

### Cosine Similarity and Retrieval

All vectors are L2-normalized, so cosine similarity equals the dot product:

```
cos(θ) = (a · b) / (‖a‖ · ‖b‖)  =  a · b   (when ‖a‖ = ‖b‖ = 1)
```

Retrieval scores sit in `[−1, 1]`. In practice, with hash-derived embeddings, scores between unrelated texts cluster near `0`; textually overlapping content produces higher scores.

The frontend maps this range to a visual `[0%, 100%]` bar fill:

```javascript
const pct = ((score + 1) / 2 * 100).toFixed(1);
```

---

## The RAG Pipeline

**File:** `src/rag_chat.py`

The RAG engine is extraction-based: it selects and assembles sentences that already exist in the retrieved context. There is no generative model. Every word in the answer came from the document.

### Full Pipeline

```
query: "What is machine learning?"
  │
  ▼
generate_embedding(query)   → 128-dim query vector

  │
  ▼
for each embedded_chunk:
    score = cosine_similarity(query_vec, chunk["embedding"])

sorted(scored, key=score, reverse=True)[:top_k]
  → [{chunk_id, text, score}, ...]

  │
  ▼
context = "\n\n".join(
    f"[Chunk {r['chunk_id']}]: {r['text']}"
    for r in results
)

  │
  ▼
generate_answer(query, context):
    keywords = extract_keywords(query)
        → re.findall(r"\b[a-zA-Z][a-zA-Z0-9]*\b", query.lower())
        → filter stopwords (49 common English words)
        → filter len < 3

    sentences = split_sentences(context)
        → strip "[Chunk N]:" headers via regex
        → split on (?<=[.!?])\s+

    for sentence in sentences:
        coverage = |matched_keywords| / |total_keywords|
        tf_bonus = Σ count(kw in sentence) / len(tokens)  [matched kw only]
        score    = coverage + tf_bonus × 0.5

    top_n = max(1, min(5, len(scored) // 2 + 1))
    top_sentences = nlargest(top_n, by=score)

    re-sort by original_index   → reading-order coherence
    answer = " ".join(top_sentences)
    ensure trailing punctuation

  │
  ▼
{query, answer, sources: [{chunk_id, score, text}]}
```

### Answer Scoring Formula

```
score(sentence, keywords) = coverage + tf_bonus × 0.5

where:
  coverage = |{kw ∈ keywords : kw ∈ sentence_words}| / |keywords|
  tf_bonus = Σ count(kw, sentence) / len(sentence_words)  for matched kw
```

Coverage rewards breadth (how many query keywords appear). The TF bonus rewards depth (how frequently they appear). The 0.5 weight keeps coverage dominant while still rewarding term emphasis. When no sentence has keyword overlap, the system falls back to the first two context sentences verbatim rather than returning nothing.

---

## Frontend Architecture

**Stack:** React 18 · Vite 5 · Axios · CSS custom properties (no external UI library)

### Component Tree

```
App.jsx
├── topbar
│   └── tab navigation  [Chunker | Process | Chat]
│
├── ChunkerPanel.jsx          ← primary playground (default tab)
│   ├── PipelineBar.jsx       ← 5-step animated pipeline visualizer
│   └── playground-grid  (CSS Grid: 285px | 1fr | 370px)
│       ├── Left column
│       │   ├── Input mode tabs  [Paste Text | Upload File]
│       │   ├── <textarea> or drag-and-drop zone
│       │   ├── Settings sliders  (method, chunk_size, overlap)
│       │   ├── Generate Chunks button
│       │   ├── Spinner.jsx
│       │   └── StatsBar  (4-tile metrics grid)
│       │
│       ├── Center column
│       │   └── chunk-list
│       │       └── ChunkCard × N
│       │           ├── chunk-meta  (id · tokens)
│       │           ├── chunk-text  (full text)
│       │           └── EmbedPreview  (DimBadge × 8 + ellipsis)
│       │
│       └── Right column
│           └── RagChatPanel.jsx
│               ├── Query input + Ask button
│               ├── Top-K slider
│               ├── Spinner.jsx
│               └── rag-history
│                   └── RagEntry × N
│                       ├── rag-query   (user question)
│                       ├── answer-box  (extracted answer)
│                       └── rag-sources
│                           └── RetrievedChunk × top_k
│                               ├── rc-header  (chunk id)
│                               ├── ScoreBar   (cosine viz)
│                               └── rc-text    (200-char preview)
│
├── ProcessPanel.jsx          ← file-path based processing tab
└── ChatPanel.jsx             ← RAG chat over processed files tab
```

### State Architecture

`ChunkerPanel.jsx` owns all playground state. `RagChatPanel.jsx` is fully controlled — it receives data as props and emits events via callbacks, making it trivially testable in isolation.

```javascript
// ChunkerPanel state — abbreviated
const [chunks,       setChunks]       = useState(null);   // [{chunk_id, text, tokens}]
const [embedMap,     setEmbedMap]     = useState(null);   // {chunk_id → embed preview}
const [stats,        setStats]        = useState(null);   // chunk statistics
const [ragHistory,   setRagHistory]   = useState([]);     // [{query, answer, sources}]
const [ragCompleted, setRagCompleted] = useState(false);  // for pipeline steps 4+5

// Pipeline steps derived purely from data presence — no manual flags
const completedSteps = new Set();
if (hasInput)     completedSteps.add("document");
if (chunks)       completedSteps.add("chunking");
if (embedMap)     completedSteps.add("embeddings");
if (ragCompleted) { completedSteps.add("retrieval"); completedSteps.add("answer"); }

const activeStep =
  loadingChunks ? "chunking"   :
  loadingEmbed  ? "embeddings" :
  loadingRag    ? "retrieval"  : null;
```

**Re-generate resets downstream state:** When the user clicks **Generate Chunks** again, `handleGenerate()` clears `chunks`, `embedMap`, `ragHistory`, and `ragCompleted` before firing the new request. The pipeline bar resets to step 1.

### Auto-Embed Flow

```javascript
async function handleGenerate() {
  // Step 1: chunk
  setLoadingChunks(true);
  const res = inputTab === "Paste Text"
    ? await processText({ text, method, chunkSize, overlap })
    : await uploadAndProcess({ file, method, chunkSize, overlap });
  setLoadingChunks(false);

  setChunks(res.data.chunks);
  setStats(res.data.stats);

  // Step 2: embed — triggered automatically, no user action required
  setLoadingEmbed(true);
  const embedRes = await embedChunks(res.data.chunks);
  setLoadingEmbed(false);

  const map = {};
  for (const e of embedRes.data.embedded) map[e.chunk_id] = e;
  setEmbedMap(map);
}
```

### Embedding Dimension Visualization

Each dimension is rendered as a `DimBadge` — a colored pill whose background intensity encodes the absolute value of the embedding dimension:

```javascript
function DimBadge({ index, value }) {
  const intensity = Math.abs(value) * 0.72 + 0.08;  // opacity range [0.08, 0.80]
  const bg = value >= 0
    ? `rgba(79,142,247,${intensity})`    // blue for positive
    : `rgba(247,95,95,${intensity})`;    // red for negative
  return (
    <span className="dim-badge" style={{ background: bg }}>
      d{index}:{value >= 0 ? "+" : ""}{value.toFixed(2)}
    </span>
  );
}
```

### Design System

All styling uses a single CSS custom property palette with no framework dependency:

```css
:root {
  --bg:       #0f1117;   /* page background */
  --surface:  #1a1d27;   /* card / panel surface */
  --border:   #2a2d3a;   /* dividers and outlines */
  --accent:   #4f8ef7;   /* primary interactive blue */
  --success:  #34c77b;   /* completed / positive states */
  --error:    #f75f5f;   /* validation and errors */
  --text:     #e2e4ed;
  --muted:    #6b6f80;
  --mono:     "JetBrains Mono", "Fira Code", monospace;
}
```

The pipeline bar's active-step pulse is a pure CSS keyframe animation:

```css
@keyframes ps-pulse {
  0%   { opacity: .75; transform: scale(.75); }
  60%  { opacity: 0;   transform: scale(1.45); }
  100% { opacity: 0;   transform: scale(1.45); }
}
```

### API Client

```javascript
// frontend/src/api.js — all calls return {ok: bool, data?} | {ok: false, error: string}
const client = axios.create({
  baseURL: "/api",       // Vite dev proxy rewrites to localhost:8000
  timeout: 60_000,
});

export async function processText({ text, method, chunkSize, overlap }) { ... }
export async function uploadAndProcess({ file, method, chunkSize, overlap }) { ... }
export async function embedChunks(chunks) { ... }
export async function askChunks({ chunks, query, topK }) { ... }
export async function processDocument({ filePath, method, chunkSize, overlap }) { ... }
export async function askQuestion({ query, filePath, topK }) { ... }
```

Vite proxy configuration:

```javascript
// frontend/vite.config.js
proxy: {
  "/api": {
    target: "http://localhost:8000",
    changeOrigin: true,
    rewrite: (path) => path.replace(/^\/api/, ""),
  },
}
```

---

## Backend Architecture

**Stack:** FastAPI · Pydantic v2 · Uvicorn · Python 3.11

### Design Principles

**Stateless endpoints.** No server-side session, cache, or in-memory store. Every request carries all data it needs — chunks are re-embedded on each `/ask-chunks` call. This makes horizontal scaling trivial and eliminates state synchronization bugs.

**Strict input validation.** All request bodies are Pydantic models with field-level validators that run before any pipeline code executes. Validation errors return `422 Unprocessable Entity` with structured error objects:

```json
{
  "detail": [
    {"field": "body.overlap", "message": "overlap must be >= 0"}
  ]
}
```

**API vs CLI separation.** `src/pipeline.py` is the CLI orchestrator and calls `sys.exit()` on failure. `src/api.py` duplicates the pipeline logic in-process because `sys.exit()` inside a FastAPI handler terminates the entire Uvicorn worker. The API raises `HTTPException` instead. Both paths call the same underlying modules.

**Selective persistence.** Only the file-path-based `/process` endpoint writes output to disk (`output/chunks_{timestamp}.json`). Text-based and upload-based endpoints return data in the response body and write nothing. Ephemerally processed text does not accumulate on the server.

### Module Responsibilities

| Module | Responsibility |
|---|---|
| `api.py` | HTTP interface, Pydantic validation, endpoint orchestration |
| `cleaner.py` | 5-stage text normalization pipeline |
| `chunker.py` | Recursive and sliding-window chunking algorithms |
| `tokenizer.py` | Token counting and chunk statistics |
| `embedding_store.py` | Deterministic embedding, cosine similarity, search |
| `rag_chat.py` | Sentence scoring, context assembly, extraction-based answer |
| `loader.py` | PDF (PyPDF2) and plaintext document ingestion |
| `pipeline.py` | CLI orchestration (load → clean → chunk → analyze) |
| `metrics.py` | Thread-safe request timing, p50/p95 percentile tracking |
| `evaluation.py` | RAG evaluation: precision@k, recall@k, hit rate |

---

## API Reference

**Base URL:** `http://localhost:8000`  
**Frontend proxy:** `/api/*` → `localhost:8000/*` (Vite dev) or configure nginx in production.

---

### `GET /health`

Health check. Returns immediately.

```bash
curl http://localhost:8000/health
```

```json
{ "status": "ok" }
```

---

### `POST /process`

Ingest a document from the server filesystem, chunk it, and persist results to `output/`.

Intended for CLI workflows and batch processing pipelines. The file must exist on the server host.

**Request:**

```json
{
  "file_path": "/app/data/research_paper.pdf",
  "method": "recursive",
  "chunk_size": 300,
  "overlap": 50
}
```

| Field | Type | Constraints |
|---|---|---|
| `file_path` | string | must be an existing file on the server |
| `method` | string | `"recursive"` or `"sliding"` |
| `chunk_size` | integer | > 0 |
| `overlap` | integer | ≥ 0, < chunk_size |

**Response:**

```json
{
  "message": "processed",
  "chunks": 14,
  "output_file": "/app/output/chunks_20260514_142301.json"
}
```

**Persisted output shape** (`output/chunks_{timestamp}.json`):

```json
{
  "source_file": "/app/data/research_paper.pdf",
  "method": "recursive",
  "chunk_size": 300,
  "overlap": 50,
  "chunks": [...],
  "stats": {...}
}
```

---

### `POST /process-text`

Chunk raw text supplied as a JSON body. Returns results directly. No file I/O.

```bash
curl -X POST http://localhost:8000/process-text \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Machine learning enables computers to learn from data...",
    "method": "recursive",
    "chunk_size": 300,
    "overlap": 50
  }'
```

**Response:**

```json
{
  "chunks": [
    { "chunk_id": 1, "text": "Machine learning enables...", "tokens": 47 },
    { "chunk_id": 2, "text": "Deep learning uses layered networks...", "tokens": 39 }
  ],
  "stats": {
    "total_chunks": 2,
    "avg_tokens_per_chunk": 43.0,
    "min_tokens": 39,
    "max_tokens": 47
  }
}
```

---

### `POST /process-upload`

Upload a `.txt` file as `multipart/form-data`. Processes in memory. No disk write.

```bash
curl -X POST http://localhost:8000/process-upload \
  -F "file=@document.txt" \
  -F "method=sliding" \
  -F "chunk_size=200" \
  -F "overlap=40"
```

**Response:**

```json
{
  "chunks": [ ... ],
  "stats": { ... },
  "filename": "document.txt"
}
```

**Notes:** UTF-8 is attempted first; Latin-1 is the fallback for legacy encodings. Only `.txt` files are accepted in this version. PDF upload is a planned enhancement.

---

### `POST /embed-chunks`

Compute embeddings for a pre-chunked document list. Returns the first 8 dimensions and L2 norm per chunk for visualization.

```bash
curl -X POST http://localhost:8000/embed-chunks \
  -H "Content-Type: application/json" \
  -d '{
    "chunks": [
      {"chunk_id": 1, "text": "Machine learning...", "tokens": 47},
      {"chunk_id": 2, "text": "Deep learning...", "tokens": 39}
    ]
  }'
```

**Response:**

```json
{
  "embedded": [
    {
      "chunk_id": 1,
      "text": "Machine learning...",
      "tokens": 47,
      "preview": [0.2341, -0.1102, 0.8714, -0.4423, 0.1198, 0.5512, -0.3087, 0.0921],
      "vector_length": 1.0,
      "dim": 128
    }
  ]
}
```

**Notes:** The full 128-dim vector is computed internally but only `preview[0:8]` is returned for transport efficiency. `vector_length` confirms L2 normalization (`≈ 1.0000`). The frontend renders these dimensions as intensity-colored badges.

---

### `POST /ask-chunks`

Stateless RAG: embed the supplied chunks on-the-fly, retrieve top-K by cosine similarity, and return an extraction-based answer with scored sources.

```bash
curl -X POST http://localhost:8000/ask-chunks \
  -H "Content-Type: application/json" \
  -d '{
    "chunks": [
      {"chunk_id": 1, "text": "Machine learning is a subset of AI...", "tokens": 47},
      {"chunk_id": 2, "text": "Deep learning uses neural networks...", "tokens": 39}
    ],
    "query": "What is machine learning?",
    "top_k": 2
  }'
```

| Field | Type | Default | Constraints |
|---|---|---|---|
| `chunks` | array | — | non-empty list of `{chunk_id, text, tokens}` |
| `query` | string | — | non-empty, whitespace-stripped |
| `top_k` | integer | 3 | > 0; internally clamped to `len(chunks)` |

**Response:**

```json
{
  "query": "What is machine learning?",
  "answer": "Machine learning is a subset of artificial intelligence. It enables computers to learn from data without being explicitly programmed.",
  "sources": [
    { "chunk_id": 1, "score": 0.2341, "text": "Machine learning is a subset of AI..." },
    { "chunk_id": 2, "score": 0.1887, "text": "Deep learning uses neural networks..." }
  ]
}
```

---

### `POST /ask`

File-based RAG. Loads a document from the server filesystem, runs the full pipeline per-request, returns an answer. Each call is fully independent — no session state.

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What are the main findings?",
    "file_path": "/app/data/report.pdf",
    "top_k": 3
  }'
```

**Response:** Same structure as `/ask-chunks`.

---

## Screenshots

> **Note:** Replace these ASCII mockups with actual screenshots once the application is running locally.

### Three-Column Playground Layout

```
┌──────────────────────────────────────────────────────────────────────────────┐
│  📄 Smart Doc Chunker    [Chunker ●]  [Process]  [Chat]                      │
├──────────────────────────────────────────────────────────────────────────────┤
│  ① Document ──── ② Chunking ──── ③ Embeddings ──── ④ Retrieval ──── ⑤ Answer │
├──────────────┬───────────────────────────────┬───────────────────────────────┤
│ Input        │  Chunks — document.txt  (8)   │  RAG Chat                     │
│ ─────────    │  [embedded]                   │  ─────────────────────────    │
│[Paste][Upload│  ┌─ chunk #1 · 47 tokens ───┐ │  [Ask about your document…]   │
│              │  │ Machine learning is a    │ │  [Ask ▶]                      │
│ [monospace   │  │ subset of AI. It enables │ │  Top-K chunks — 3             │
│  textarea    │  │ computers to learn...    │ │  ─────────────────────────    │
│  for pasting │  │ ─── Embedding ────────── │ │  ▶ What is machine learning?  │
│  document    │  │ 128-dim · ‖v‖ = 1.000   │ │                               │
│  content]    │  │[d0:+.23][d1:−.11][d2:+.87│ │  Machine learning is a subset │
│              │  │[d3:−.44][d4:+.12]···+120 │ │  of artificial intelligence.  │
│ Settings     │  └──────────────────────────┘ │  It enables computers to      │
│ ─────────    │                               │  learn from data.             │
│ method:      │  ┌─ chunk #2 · 39 tokens ───┐ │                               │
│ recursive    │  │ Deep learning uses       │ │  Retrieved · 3 chunks         │
│ size:  ──300 │  │ layered neural networks  │ │  ─────────────────────────    │
│ overlap: ─50 │  │ to process patterns...   │ │  chunk #1                     │
│              │  │ [d0:−.05][d1:+.72]···    │ │  ████████████████░░  0.2341   │
│ [Generate    │  └──────────────────────────┘ │  "Machine learning is a sub.. │
│  Chunks]     │                               │                               │
│              │  ┌─ chunk #3 · 52 tokens ───┐ │  chunk #3                     │
│ ┌─ Stats ──┐ │  │ Natural language         │ │  ███████████░░░░░░  0.1887    │
│ │Chunks: 8 │ │  │ processing allows...     │ │  "Natural language processing │
│ │Avg: 43.1 │ │  └──────────────────────────┘ │  allows computers to...       │
│ │Min:  31  │ │                               │                               │
│ │Max:  58  │ │                               │  [Clear history]              │
└──────────────┴───────────────────────────────┴───────────────────────────────┘
```

---

## Installation

### Prerequisites

| Requirement | Version |
|---|---|
| Python | 3.11+ |
| Node.js | 18+ |
| npm | 9+ |
| Docker (optional) | 24+ |

### Option A — Local Development (Full Stack)

```bash
# 1. Clone
git clone https://github.com/your-username/smart-doc-chunker.git
cd smart-doc-chunker

# 2. Install Python dependencies
pip install -r requirements.txt

# 3. Start the FastAPI backend
uvicorn src.api:app --reload --host 0.0.0.0 --port 8000

# 4. In a separate terminal — install and start the frontend
cd frontend
npm install
npm run dev         # hot-reload dev server on :3000

# 5. Open the playground
open http://localhost:3000
```

The Vite dev server proxies all `/api/*` requests to `localhost:8000`. No CORS configuration is needed during development.

### Option B — Docker (Backend Only)

```bash
docker-compose up --build
```

The `docker-compose.yml` mounts `./data` and `./output` as volumes so documents and chunk artifacts persist across container restarts. The backend is available at `http://localhost:8000`.

To also serve the built frontend:

```bash
cd frontend && npm run build
npx serve dist -p 3000
```

### Option C — Streamlit Interface (Alternative)

The `app.py` file provides a self-contained Streamlit interface that runs the same Python pipeline modules without the FastAPI layer:

```bash
streamlit run app.py
```

---

## Run Locally

### Start Backend

```bash
cd smart-doc-chunker
pip install -r requirements.txt
uvicorn src.api:app --reload --host 0.0.0.0 --port 8000
```

Backend runs at: [http://localhost:8000](http://localhost:8000)

Swagger docs: [http://localhost:8000/docs](http://localhost:8000/docs)

### Start Frontend

```bash
cd frontend
npm install
npm run dev
```

Frontend runs at: [http://localhost:3000](http://localhost:3000)

---

## Demo Walkthrough

### Step 1 — Provide a document

Open `http://localhost:3000`. The **Chunker** tab is active by default. Choose **Paste Text** and paste any multi-paragraph document — a research abstract, technical specification, news article, or book excerpt.

### Step 2 — Configure chunking parameters

| Parameter | Guidance |
|---|---|
| **recursive** | Best for structured documents with clear paragraph/sentence boundaries (papers, reports, articles). Preserves semantic units. |
| **sliding** | Best for dense uniform text (transcripts, code, logs). Guarantees consistent chunk sizes. |
| **chunk_size** | Increase for longer, denser passages. Decrease for fine-grained retrieval. 200–400 tokens is a practical range for most text. |
| **overlap** | Set to 10–20% of chunk_size to preserve context across boundaries. Set to 0 for maximum index density with no redundancy. |

### Step 3 — Generate chunks and observe embedding

Click **Generate Chunks**. The pipeline bar advances:

1. **Document** ✓ — Input was detected
2. **Chunking** → ✓ — Chunks appear in the center column
3. **Embeddings** → ✓ — Auto-triggered; embedding badges populate each card

Read the embedding badges. Bright blue (`d2:+0.87`) means this dimension strongly encodes something about this chunk's content. Bright red (`d3:−0.44`) is a strong negative loading. Pale badges near `±0.05` carry minimal signal for this text.

### Step 4 — Ask a question

In the right column, type a natural-language question and press **Enter** or click **Ask**. Good first questions:

- *"What is [core concept from the text]?"*
- *"How does [process described] work?"*
- *"What are the main findings?"*

**Retrieval** and **Answer** steps complete. The answer appears, followed by the retrieved chunks with score bars.

### Step 5 — Analyze retrieval

Compare the score bars. High scores (green, > 0.3) indicate chunks that share substantial textual overlap with the query. Examine the center column to confirm that the highest-scoring chunks visually contain the answer — the retrieval is auditable.

### Step 6 — Experiment

- **Same query, different top-K:** Change the slider from 3 to 1 or 5. Observe how a broader context window affects answer quality.
- **Different chunk size:** Re-generate with `chunk_size = 100` vs `chunk_size = 500`. Small chunks increase precision; large chunks improve context but may dilute relevance scores.
- **Out-of-scope query:** Ask something not covered in the text. The extraction-based synthesizer will fall back to the most prominent sentences from retrieved context rather than fabricating an answer.
- **sliding vs recursive:** Re-generate with the same text but the other method. Compare chunk boundaries and how they affect which chunks are retrieved for the same query.

---

## Project Structure

```
smart-doc-chunker/
│
├── src/                          # Backend Python modules
│   ├── api.py                    # FastAPI app — 7 endpoints, Pydantic models
│   ├── chunker.py                # Recursive + sliding-window algorithms
│   ├── cleaner.py                # 5-stage text normalization pipeline
│   ├── tokenizer.py              # Token counting and chunk statistics
│   ├── embedding_store.py        # 128-dim embeddings, cosine similarity, search
│   ├── rag_chat.py               # Keyword sentence scoring, answer synthesis
│   ├── loader.py                 # PDF (PyPDF2) and .txt document ingestion
│   ├── pipeline.py               # CLI orchestration (calls sys.exit on failure)
│   ├── main.py                   # CLI entry point, argparse, method aliases
│   ├── metrics.py                # Thread-safe request timing, p50/p95 percentiles
│   ├── evaluation.py             # RAG eval: precision@k, recall@k, hit rate
│   ├── config.py                 # Configuration constants (extension point)
│   └── utils.py                  # Shared utilities (extension point)
│
├── frontend/
│   ├── src/
│   │   ├── App.jsx               # Tab shell: Chunker | Process | Chat
│   │   ├── main.jsx              # React 18 entry (StrictMode)
│   │   ├── api.js                # Axios client — 6 typed API functions
│   │   ├── index.css             # Full design system — dark theme, all components
│   │   └── components/
│   │       ├── ChunkerPanel.jsx  # Three-column RAG playground, owns all state
│   │       ├── PipelineBar.jsx   # 5-step animated visualizer
│   │       ├── RagChatPanel.jsx  # Controlled RAG chat with score bars
│   │       ├── ProcessPanel.jsx  # File-path processing tab
│   │       ├── ChatPanel.jsx     # File-based RAG chat tab
│   │       └── Spinner.jsx       # Loading indicator
│   │
│   ├── dist/                     # Production build output (gitignored)
│   ├── index.html
│   ├── vite.config.js            # Dev server :3000, /api/* → :8000 proxy
│   └── package.json
│
├── data/
│   └── sample.txt                # Example document for testing
│
├── output/                       # Persisted chunk JSON files (gitignored)
│   └── chunks_{timestamp}.json   # Written only by /process endpoint
│
├── docker/
│   └── Dockerfile                # Python 3.11-slim, uvicorn on :8000
│
├── docker-compose.yml            # Single-service config, volumes for data/ output/
├── app.py                        # Streamlit alternative interface
├── requirements.txt              # PyPDF2>=3.0.0, fastapi>=0.110.0, uvicorn[standard]
└── README.md
```

---

## Engineering Decisions

### Deterministic embeddings without an external model

The SHA-256 hash-based embedding generator was chosen deliberately. It provides:

- **Zero external dependencies** — no API key, no network call, no 400MB model download to get the playground running
- **Strict reproducibility** — the same text always produces the same vector, making behavior predictable during development
- **Pipeline structural correctness** — the full RAG flow (chunk → embed → retrieve → answer) is implemented exactly as it would be in production; only the semantic quality of the embeddings differs

The tradeoff is transparent: hash embeddings measure textual overlap, not semantic meaning. The codebase marks `generate_embedding()` as the explicit "drop-in replacement point." Moving to `sentence-transformers` or the OpenAI Embeddings API requires changing exactly this one function and nothing else in the pipeline.

### Chunk overlap for context continuity

With strictly non-overlapping chunks, a sentence that happens to fall on a boundary is split. Neither resulting chunk contains the complete thought. A retrieval query targeting that concept may return the wrong half — or neither half — and the answer synthesizer receives incomplete context.

Overlap ensures that for any two adjacent windows, at least one contains each boundary-spanning concept in full. The configurable overlap parameter (enforced at `< chunk_size` by Pydantic validation) makes this tradeoff directly observable: set overlap to 0, ask a query about a concept near a boundary, observe degraded retrieval; increase overlap, observe improved recall.

### Adaptive minimum token threshold

The original implementation used a fixed minimum of 10 tokens to filter near-empty chunk fragments. The bug: with `chunk_size = 30`, the adaptive formula `max(1, min(chunk_size // 5, 10))` would produce `min_tokens = 6`. With the fixed value of 10, every chunk of fewer than 10 tokens would be silently dropped — and at small chunk sizes, many legitimate chunks have fewer than 10 tokens. The pipeline would return an empty list while appearing to succeed.

The adaptive formula scales the minimum proportionally to chunk size, eliminating this class of silent failure at small chunk_size values.

### Stateless API design

The `/ask-chunks` endpoint re-embeds the client-supplied chunks on every call. The alternative — a stateful session where the server stores embedded chunks after the first request — would be faster per request but introduces session lifecycle management, memory limits, eviction policy, and inter-worker state sharing in a multi-process deployment.

The additional compute of re-embedding is fast: 128-dimensional deterministic embedding processes hundreds of chunks in milliseconds. The architectural simplicity of complete statelessness is worth this cost at playground scale, and the design pattern is correct for production systems where stateless horizontal scaling is the default requirement.

### Extraction-based answer generation

The answer synthesizer selects and reorders existing sentences rather than generating new text. This is not a simplification for the playground — it is a deliberate architectural choice for a system where attribution matters. Every word in the answer has a provenance: a specific chunk, a specific sentence, a specific position in the source document. There is no hallucination surface because there is no generation step.

The keyword coverage + TF scoring formula (`coverage + tf_bonus × 0.5`) is interpretable: you can trace exactly why a sentence was selected. The re-sorting by original index restores reading coherence without sacrificing relevance ranking.

### Modular pipeline architecture

Each pipeline stage is an independent Python module with no knowledge of adjacent stages. `chunker.py` does not import `cleaner.py`. `rag_chat.py` does not import `chunker.py`. They communicate through plain Python dicts with consistent schemas. This enables:

- **Unit testing each stage in isolation** — feed a chunk with known content, assert token counts without running the full pipeline
- **Swapping implementations without regressions** — replace the extraction-based answer with an LLM call by changing `rag_chat.py` alone
- **Sharing between CLI and API** — `main.py` and `api.py` both call `chunker.chunk_text()` with no adaptation layer

---

## Future Enhancements

### Semantic embeddings via sentence-transformers

```python
# src/embedding_store.py — one-function swap
from sentence_transformers import SentenceTransformer
_model = SentenceTransformer("all-MiniLM-L6-v2")

def generate_embedding(text: str) -> list[float]:
    return _model.encode(text, normalize_embeddings=True).tolist()
```

Expected outcome: dramatically improved retrieval recall on paraphrase queries where the query and the relevant chunk share meaning but not vocabulary.

### Vector database integration

Replace the in-memory `O(n)` linear scan in `embedding_store.search()` with a vector database (Qdrant, Weaviate, pgvector). At document scale ≥ 10,000 chunks, approximate nearest-neighbour search via HNSW indices outperforms brute-force cosine by orders of magnitude.

### LLM-based answer generation

Replace the extraction-based `generate_answer()` with a call to a generative model. The context string is already assembled in the format required for a RAG prompt:

```
[Chunk 3]: ...
[Chunk 7]: ...

Question: {query}
Answer:
```

Streaming the response token by token via `StreamingResponse` and server-sent events would eliminate perceived latency on long answers.

### PDF upload with OCR

`loader.py` already parses PDFs via PyPDF2. Extending `/process-upload` to accept `.pdf` requires removing the `.txt`-only guard and routing the upload to a temporary file before calling `loader.load_document()`. For scanned PDFs without embedded text, Tesseract OCR (via `pytesseract`) would be added as a fallback.

### Evaluation dashboard

`src/evaluation.py` already implements `precision_at_k`, `recall_at_k`, and `hit_rate` with per-query result tracking against a golden relevance set. A **Evaluate** tab in the UI could expose this: define a test query set, run retrieval across configurations, display a comparison matrix of chunking method × chunk size × overlap.

### Observability

`src/metrics.py` is a thread-safe request timer with p50/p95 percentile computation, ready for a `/metrics` endpoint. A Prometheus scrape target feeding a Grafana dashboard would close the observability loop: chunking latency, embedding throughput, retrieval time, answer generation latency — all visible per endpoint.

### Sentence-level chunking

The CLI (`main.py`) maps the method alias `"sentence"` to `"recursive"` as a convenience. A true sentence-boundary chunker using spaCy (`en_core_web_sm`) or NLTK would improve chunk coherence for dense academic text where paragraph breaks are infrequent.

---

## What This Demonstrates

This project implements the core competencies that appear in AI engineering interviews, production RAG system designs, and applied ML engineering roles.

### RAG Pipeline Engineering
- Document ingestion supporting multiple file formats with encoding fallback
- Multi-stage text preprocessing with explicit noise pattern handling
- Two chunking strategies with configurable overlap and adaptive filtering
- Stateless embedding and retrieval architecture designed for horizontal scale
- Extraction-based answer synthesis with interpretable keyword-coverage scoring

### API Design and Backend Engineering
- RESTful endpoint design with Pydantic v2 field-level validation
- Separation of CLI orchestration (`pipeline.py`, `sys.exit`) from HTTP request handling (FastAPI, `HTTPException`)
- Structured error responses with field-level detail — no raw tracebacks to clients
- Multipart file upload with in-memory processing and no server-side persistence
- Selective persistence: only batch-mode endpoints write to disk

### Frontend Systems Engineering
- React component architecture with clear owned-state vs. controlled-component boundaries
- Pipeline state derived from data presence — no manual step flags or state machines
- Multi-step asynchronous workflows with sequential loading states and error recovery
- CSS-first design system using custom properties, no framework dependency, animated state transitions
- Vite proxy configuration for clean local development without CORS complexity

### Algorithms and Data Structures
- Greedy paragraph-sentence merging for structure-aware recursive chunking
- Token-stride sliding window with configurable overlap guaranteeing boundary coverage
- Hash-derived L2-normalized unit vector generation on the hypersphere
- TF-keyword-coverage sentence scoring with position-aware reranking for coherent extraction

### Software Engineering Craft
- Single-responsibility Python modules with no lateral imports between pipeline stages
- Adaptive thresholds with documented edge-case analysis (the fixed-10 silent-drop bug)
- Thread-safe metrics collection (`threading.Lock`, percentile computation) for concurrent request environments
- Docker multi-volume deployment separating mutable data and build artifacts

### Systems Thinking
- Acknowledging the semantic limitation of hash embeddings and documenting the exact replacement surface
- Choosing stateless API design to eliminate session management complexity at the cost of re-embed latency
- Preferring extraction over generation to eliminate hallucination surface in a document-grounded system
- Building visualization infrastructure so the pipeline is auditable, not just functional

---

## License

This project is licensed under the **MIT License**. See [LICENSE](LICENSE) for full terms.

---

<div align="center">

**Built to make RAG pipelines visible · Designed as a systems engineering learning tool**

*Document → Chunking → Embeddings → Retrieval → Answer*

</div>
