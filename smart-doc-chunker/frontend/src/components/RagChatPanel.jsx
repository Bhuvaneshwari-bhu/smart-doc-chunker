import { useState, useRef, useEffect } from "react";
import { askChunks } from "../api.js";
import Spinner from "./Spinner.jsx";

function ScoreBar({ score }) {
  // Map [-1, 1] → [0%, 100%] so bar always has width
  const pct = ((score + 1) / 2 * 100).toFixed(1);
  const color =
    score >= 0.3  ? "var(--success)" :
    score >= 0.05 ? "var(--accent)"  : "var(--muted)";
  return (
    <div className="score-bar-wrap">
      <div className="score-bar-track">
        <div className="score-bar-fill" style={{ width: `${pct}%`, background: color }} />
      </div>
      <span className="score-val" style={{ color }}>{score.toFixed(4)}</span>
    </div>
  );
}

function RetrievedChunk({ source }) {
  const preview = source.text.length > 200
    ? source.text.slice(0, 200) + "…"
    : source.text;
  return (
    <div className="retrieved-chunk">
      <div className="rc-header">
        <span className="rc-cid">chunk #{source.chunk_id}</span>
        <ScoreBar score={source.score} />
      </div>
      <div className="rc-text">{preview}</div>
    </div>
  );
}

function RagEntry({ entry }) {
  return (
    <div className="rag-entry">
      <div className="rag-query">{entry.query}</div>
      <div className="answer-box">{entry.answer}</div>
      {entry.sources?.length > 0 && (
        <div className="rag-sources">
          <p className="section-label" style={{ marginBottom: ".4rem" }}>
            Retrieved · {entry.sources.length} chunk{entry.sources.length > 1 ? "s" : ""}
          </p>
          {entry.sources.map((s) => (
            <RetrievedChunk key={s.chunk_id} source={s} />
          ))}
        </div>
      )}
    </div>
  );
}

export default function RagChatPanel({
  chunks,
  history,
  loading,
  onQueryStart,
  onQueryEnd,
  onQuerySuccess,
  onClearHistory,
}) {
  const [query, setQuery] = useState("");
  const [topK,  setTopK]  = useState(3);
  const [error, setError] = useState("");
  const bottomRef = useRef(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [history]);

  async function handleAsk() {
    const q = query.trim();
    if (!q || !chunks?.length) return;
    setError("");
    onQueryStart?.();
    const res = await askChunks({ chunks, query: q, topK });
    onQueryEnd?.();
    if (res.ok) {
      onQuerySuccess?.(res.data);
      setQuery("");
    } else {
      setError(res.error);
    }
  }

  function handleKey(e) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleAsk();
    }
  }

  const ready = !!chunks?.length;
  const maxTopK = Math.min(10, chunks?.length ?? 1);

  return (
    <>
      <p className="section-label">RAG Chat</p>

      {!ready ? (
        <p className="rag-empty">Generate chunks first to enable semantic search.</p>
      ) : (
        <>
          <div className="field">
            <div className="query-row">
              <input
                type="text"
                placeholder="Ask about your document…"
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                onKeyDown={handleKey}
                disabled={loading}
              />
              <button
                className="btn-ask"
                onClick={handleAsk}
                disabled={loading || !query.trim()}
              >
                Ask
              </button>
            </div>
          </div>

          <div className="field">
            <label>Top-K chunks — {topK}</label>
            <div className="slider-row">
              <input
                type="range" min={1} max={maxTopK} step={1}
                value={topK}
                onChange={(e) => setTopK(Number(e.target.value))}
              />
              <span className="slider-val">{topK}</span>
            </div>
          </div>

          {loading && <Spinner label="Searching & generating answer…" />}
          {error   && <div className="msg error">{error}</div>}
        </>
      )}

      {history?.length === 0 && ready && !loading && (
        <p className="rag-empty">
          Ask a question — results will appear here.
        </p>
      )}

      {history?.length > 0 && (
        <>
          <hr className="divider" />
          <div className="rag-history">
            {history.map((entry, i) => (
              <div key={i}>
                <RagEntry entry={entry} />
                {i < history.length - 1 && (
                  <hr className="divider" style={{ margin: ".75rem 0" }} />
                )}
              </div>
            ))}
            <div ref={bottomRef} />
          </div>
          <button
            className="btn-primary"
            style={{ background: "var(--surface)", color: "var(--muted)" }}
            onClick={onClearHistory}
          >
            Clear history
          </button>
        </>
      )}
    </>
  );
}
