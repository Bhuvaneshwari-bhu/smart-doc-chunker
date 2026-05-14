import { useState, useRef, useEffect } from "react";
import { askQuestion } from "../api.js";
import Spinner from "./Spinner.jsx";

function ChunkCard({ source }) {
  const preview = source.text
    ? source.text.slice(0, 200) + (source.text.length > 200 ? "…" : "")
    : "—";
  const scoreColor =
    source.score >= 0.6 ? "var(--success)"
    : source.score >= 0.2 ? "var(--accent)"
    : "var(--muted)";

  return (
    <div className="chunk-card">
      <div className="chunk-meta">
        <span className="cid">chunk #{source.chunk_id}</span>
        <span className="score" style={{ color: scoreColor }}>
          score {source.score?.toFixed(4) ?? "—"}
        </span>
      </div>
      <div className="chunk-text">{preview}</div>
    </div>
  );
}

function ChatEntry({ entry }) {
  return (
    <div>
      <div className="chat-entry-query">{entry.query}</div>
      <div style={{ margin: ".5rem 0" }}>
        <div className="answer-box">{entry.answer}</div>
      </div>
      {entry.sources?.length > 0 && (
        <div>
          <p className="section-label" style={{ marginBottom: ".5rem" }}>
            Retrieved chunks ({entry.sources.length})
          </p>
          <div className="chunk-list">
            {entry.sources.map((s) => (
              <ChunkCard key={s.chunk_id} source={s} />
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

export default function ChatPanel({ session }) {
  const [query,   setQuery]   = useState("");
  const [topK,    setTopK]    = useState(3);
  const [loading, setLoading] = useState(false);
  const [error,   setError]   = useState("");
  const [history, setHistory] = useState([]);
  const bottomRef = useRef(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [history]);

  async function handleAsk() {
    const q = query.trim();
    if (!q) return;
    if (!session?.filePath) {
      setError("Process a document first (use the Process tab).");
      return;
    }
    setError("");
    setLoading(true);
    const res = await askQuestion({ query: q, filePath: session.filePath, topK });
    setLoading(false);
    if (res.ok) {
      setHistory((h) => [...h, res.data]);
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

  return (
    <div className="panel-layout">
      {/* ── Controls ── */}
      <div className="controls-col">
        <p className="section-label">Query settings</p>

        <div className="field">
          <label>Active document</label>
          <div style={{
            padding: ".45rem .7rem",
            background: "var(--bg)",
            border: "1px solid var(--border)",
            borderRadius: "5px",
            fontFamily: "var(--mono)",
            fontSize: ".78rem",
            color: session?.filePath ? "var(--accent)" : "var(--muted)",
            wordBreak: "break-all",
          }}>
            {session?.filePath ?? "none — process a document first"}
          </div>
        </div>

        <div className="field">
          <label>Top-K chunks — {topK}</label>
          <div className="slider-row">
            <input
              type="range" min={1} max={10} step={1}
              value={topK}
              onChange={(e) => setTopK(Number(e.target.value))}
            />
            <span className="slider-val">{topK}</span>
          </div>
        </div>

        <hr className="divider" />

        <div className="field">
          <label>Question</label>
          <div className="query-row">
            <input
              type="text"
              placeholder="Ask something…"
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

        {loading && <Spinner label="Retrieving…" />}
        {error   && <div className="msg error">{error}</div>}

        {history.length > 0 && (
          <button
            className="btn-primary"
            style={{ background: "var(--surface)", color: "var(--muted)", marginTop: ".5rem" }}
            onClick={() => setHistory([])}
          >
            Clear history
          </button>
        )}
      </div>

      {/* ── Results ── */}
      <div className="results-col">
        <p className="section-label">Chat history</p>

        {history.length === 0 && !loading && (
          <p style={{ color: "var(--muted)", fontSize: ".82rem" }}>
            {session?.filePath
              ? "Ask a question about your document."
              : "Process a document first, then ask questions here."}
          </p>
        )}

        <div className="chat-thread">
          {history.map((entry, i) => (
            <div key={i}>
              <ChatEntry entry={entry} />
              {i < history.length - 1 && <hr className="divider" style={{ marginTop: "1rem" }} />}
            </div>
          ))}
        </div>
        <div ref={bottomRef} />
      </div>
    </div>
  );
}
