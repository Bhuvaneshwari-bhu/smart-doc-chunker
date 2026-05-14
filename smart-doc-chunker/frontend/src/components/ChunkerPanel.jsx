import { useState, useRef } from "react";
import { processText, uploadAndProcess, embedChunks } from "../api.js";
import PipelineBar  from "./PipelineBar.jsx";
import RagChatPanel from "./RagChatPanel.jsx";
import Spinner      from "./Spinner.jsx";

const METHODS     = ["recursive", "sliding"];
const INPUT_TABS  = ["Paste Text", "Upload File"];

// ── Embedding preview ────────────────────────────────────────────────────────

function DimBadge({ index, value }) {
  const intensity = Math.abs(value) * 0.72 + 0.08;
  const bg = value >= 0
    ? `rgba(79,142,247,${intensity.toFixed(2)})`
    : `rgba(247,95,95,${intensity.toFixed(2)})`;
  return (
    <span className="dim-badge" style={{ background: bg }}>
      d{index}:{value >= 0 ? "+" : ""}{value.toFixed(2)}
    </span>
  );
}

function EmbedPreview({ data }) {
  return (
    <div className="embed-preview">
      <div className="embed-preview-header">
        <span className="embed-label">Embedding</span>
        <span className="embed-meta">
          {data.dim}-dim · ‖v‖ = {data.vector_length.toFixed(3)}
        </span>
      </div>
      <div className="embed-dims-row">
        {data.preview.map((v, i) => <DimBadge key={i} index={i} value={v} />)}
        <span className="embed-more">···+{data.dim - 8}</span>
      </div>
    </div>
  );
}

// ── Chunk card ────────────────────────────────────────────────────────────────

function ChunkCard({ chunk, embedInfo }) {
  return (
    <div className="chunk-card">
      <div className="chunk-meta">
        <span className="cid">chunk #{chunk.chunk_id}</span>
        <span className="ctok">{chunk.tokens} tokens</span>
      </div>
      <div className="chunk-text">{chunk.text}</div>
      {embedInfo && <EmbedPreview data={embedInfo} />}
    </div>
  );
}

// ── Stats bar ─────────────────────────────────────────────────────────────────

function StatsBar({ stats }) {
  const tiles = [
    ["Chunks",  stats.total_chunks],
    ["Avg tok", stats.avg_tokens_per_chunk.toFixed(1)],
    ["Min tok", stats.min_tokens],
    ["Max tok", stats.max_tokens],
  ];
  return (
    <div className="stats-bar">
      {tiles.map(([label, value]) => (
        <div key={label} className="stat-tile">
          <span className="stat-tile-label">{label}</span>
          <span className="stat-tile-value">{value}</span>
        </div>
      ))}
    </div>
  );
}

// ── Main playground ───────────────────────────────────────────────────────────

export default function ChunkerPanel() {
  // Input
  const [inputTab,  setInputTab]  = useState("Paste Text");
  const [pasteText, setPasteText] = useState("");
  const [file,      setFile]      = useState(null);
  const [dragOver,  setDragOver]  = useState(false);

  // Settings
  const [method,    setMethod]    = useState("recursive");
  const [chunkSize, setChunkSize] = useState(300);
  const [overlap,   setOverlap]   = useState(50);

  // Pipeline outputs
  const [chunks,   setChunks]   = useState(null);
  const [stats,    setStats]    = useState(null);
  const [embedMap, setEmbedMap] = useState(null);  // chunk_id → embed preview data
  const [filename, setFilename] = useState("");

  // RAG state (owned here so we can reset on re-generate)
  const [ragHistory,   setRagHistory]   = useState([]);
  const [loadingRag,   setLoadingRag]   = useState(false);
  const [ragCompleted, setRagCompleted] = useState(false);

  // Loading & errors
  const [loadingChunks, setLoadingChunks] = useState(false);
  const [loadingEmbed,  setLoadingEmbed]  = useState(false);
  const [chunkError,    setChunkError]    = useState("");
  const [embedError,    setEmbedError]    = useState("");

  const fileInputRef = useRef(null);
  const overlapMax   = Math.max(0, chunkSize - 1);

  // ── Pipeline step derivation ────────────────────────────────────────────────

  const hasInput =
    inputTab === "Paste Text" ? pasteText.trim().length > 0 : !!file;

  const completedSteps = new Set();
  if (hasInput)     completedSteps.add("document");
  if (chunks)       completedSteps.add("chunking");
  if (embedMap)     completedSteps.add("embeddings");
  if (ragCompleted) { completedSteps.add("retrieval"); completedSteps.add("answer"); }

  const activeStep =
    loadingChunks ? "chunking"   :
    loadingEmbed  ? "embeddings" :
    loadingRag    ? "retrieval"  : null;

  // ── File handling ───────────────────────────────────────────────────────────

  function acceptFile(f) {
    if (!f.name.endsWith(".txt")) {
      setChunkError("Only .txt files are supported.");
      return;
    }
    setFile(f);
    setChunkError("");
  }

  // ── Generate chunks → auto-embed ────────────────────────────────────────────

  async function handleGenerate() {
    setChunkError("");
    setEmbedError("");
    setChunks(null);
    setStats(null);
    setEmbedMap(null);
    setFilename("");
    setRagHistory([]);
    setRagCompleted(false);

    if (inputTab === "Paste Text" && !pasteText.trim()) {
      setChunkError("Paste some text first.");
      return;
    }
    if (inputTab === "Upload File" && !file) {
      setChunkError("Select a .txt file first.");
      return;
    }

    setLoadingChunks(true);
    const res = inputTab === "Paste Text"
      ? await processText({ text: pasteText, method, chunkSize, overlap })
      : await uploadAndProcess({ file, method, chunkSize, overlap });
    setLoadingChunks(false);

    if (!res.ok) { setChunkError(res.error); return; }

    const newChunks = res.data.chunks;
    setChunks(newChunks);
    setStats(res.data.stats);
    if (res.data.filename) setFilename(res.data.filename);

    // Auto-embed immediately after chunking
    setLoadingEmbed(true);
    const embedRes = await embedChunks(newChunks);
    setLoadingEmbed(false);

    if (embedRes.ok) {
      const map = {};
      for (const e of embedRes.data.embedded) map[e.chunk_id] = e;
      setEmbedMap(map);
    } else {
      setEmbedError(embedRes.error);
    }
  }

  const canGenerate = !loadingChunks && !loadingEmbed && hasInput;

  // ── Render ──────────────────────────────────────────────────────────────────

  return (
    <div className="playground-wrap">
      <PipelineBar completedSteps={completedSteps} activeStep={activeStep} />

      <div className="playground-grid">

        {/* ── Left: Input + Settings ────────────────────────────────────── */}
        <div className="pg-col">
          <p className="section-label">Input</p>

          <div className="input-tabs">
            {INPUT_TABS.map((t) => (
              <button
                key={t}
                className={`input-tab-btn${inputTab === t ? " active" : ""}`}
                onClick={() => { setInputTab(t); setChunkError(""); }}
              >
                {t}
              </button>
            ))}
          </div>

          {inputTab === "Upload File" ? (
            <div
              className={`drop-zone${dragOver ? " drag-over" : ""}${file ? " has-file" : ""}`}
              onClick={() => fileInputRef.current?.click()}
              onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
              onDragLeave={() => setDragOver(false)}
              onDrop={(e) => {
                e.preventDefault();
                setDragOver(false);
                const f = e.dataTransfer?.files?.[0];
                if (f) acceptFile(f);
              }}
            >
              <input
                ref={fileInputRef}
                type="file"
                accept=".txt"
                style={{ display: "none" }}
                onChange={(e) => { const f = e.target.files?.[0]; if (f) acceptFile(f); }}
              />
              {file ? (
                <>
                  <div className="drop-icon" style={{ color: "var(--success)" }}>✓</div>
                  <div style={{ fontSize: ".78rem", color: "var(--success)", fontFamily: "var(--mono)", wordBreak: "break-all" }}>
                    {file.name}
                  </div>
                  <div className="drop-hint">click to change</div>
                </>
              ) : (
                <>
                  <div className="drop-icon">↑</div>
                  <div className="drop-hint">Drag & drop a .txt file</div>
                  <div className="drop-hint" style={{ opacity: .5 }}>or click to browse</div>
                </>
              )}
            </div>
          ) : (
            <textarea
              className="paste-area"
              placeholder="Paste document content here..."
              value={pasteText}
              onChange={(e) => setPasteText(e.target.value)}
              rows={7}
            />
          )}

          <p className="section-label">Settings</p>

          <div className="field">
            <label>Chunking method</label>
            <select value={method} onChange={(e) => setMethod(e.target.value)}>
              {METHODS.map((m) => <option key={m} value={m}>{m}</option>)}
            </select>
          </div>

          <div className="field">
            <label>Chunk size — {chunkSize} tokens</label>
            <div className="slider-row">
              <input
                type="range" min={50} max={1000} step={50}
                value={chunkSize}
                onChange={(e) => {
                  const v = Number(e.target.value);
                  setChunkSize(v);
                  if (overlap >= v) setOverlap(Math.max(0, v - 50));
                }}
              />
              <span className="slider-val">{chunkSize}</span>
            </div>
          </div>

          <div className="field">
            <label>Overlap — {overlap} tokens</label>
            <div className="slider-row">
              <input
                type="range" min={0} max={overlapMax} step={10}
                value={overlap}
                onChange={(e) => setOverlap(Number(e.target.value))}
              />
              <span className="slider-val">{overlap}</span>
            </div>
          </div>

          <button className="btn-primary" onClick={handleGenerate} disabled={!canGenerate}>
            {loadingChunks ? "Chunking…" : loadingEmbed ? "Embedding…" : "Generate Chunks"}
          </button>

          {(loadingChunks || loadingEmbed) && (
            <Spinner label={loadingChunks ? "Splitting into chunks…" : "Computing embeddings…"} />
          )}
          {chunkError && <div className="msg error">{chunkError}</div>}
          {embedError && (
            <div className="msg error" style={{ fontSize: ".75rem" }}>
              Embed warning: {embedError}
            </div>
          )}

          {stats && <StatsBar stats={stats} />}
        </div>

        {/* ── Center: Chunk Viewer + Embedding Previews ─────────────────── */}
        <div className="pg-col">
          <p className="section-label">
            {chunks
              ? `Chunks${filename ? ` — ${filename}` : ""} (${chunks.length})`
              : "Chunk Viewer"}
            {loadingEmbed && (
              <span className="col-badge loading">embedding…</span>
            )}
            {embedMap && !loadingEmbed && (
              <span className="col-badge ready">embedded</span>
            )}
          </p>

          {!chunks && !loadingChunks && (
            <p className="col-empty">
              Configure input on the left and click <em>Generate Chunks</em>.
            </p>
          )}

          {loadingChunks && <Spinner label="Splitting document into chunks…" />}

          {chunks && (
            <div className="chunk-list">
              {chunks.map((c) => (
                <ChunkCard
                  key={c.chunk_id}
                  chunk={c}
                  embedInfo={embedMap ? (embedMap[c.chunk_id] ?? null) : null}
                />
              ))}
            </div>
          )}
        </div>

        {/* ── Right: RAG Chat ───────────────────────────────────────────── */}
        <div className="pg-col">
          <RagChatPanel
            chunks={chunks}
            history={ragHistory}
            loading={loadingRag}
            onQueryStart={() => setLoadingRag(true)}
            onQueryEnd={()   => setLoadingRag(false)}
            onQuerySuccess={(entry) => {
              setRagHistory((h) => [...h, entry]);
              setRagCompleted(true);
            }}
            onClearHistory={() => {
              setRagHistory([]);
              setRagCompleted(false);
            }}
          />
        </div>

      </div>
    </div>
  );
}
