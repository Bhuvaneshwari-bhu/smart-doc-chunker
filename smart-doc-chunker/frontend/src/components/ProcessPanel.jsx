import { useState } from "react";
import { processDocument } from "../api.js";
import Spinner from "./Spinner.jsx";

const METHODS = ["recursive", "sliding", "fixed", "sentence"];

export default function ProcessPanel({ onProcessed }) {
  const [filePath,  setFilePath]  = useState("");
  const [method,    setMethod]    = useState("recursive");
  const [chunkSize, setChunkSize] = useState(300);
  const [overlap,   setOverlap]   = useState(50);
  const [loading,   setLoading]   = useState(false);
  const [result,    setResult]    = useState(null);   // { ok, data?, error? }

  async function handleProcess() {
    if (!filePath.trim()) {
      setResult({ ok: false, error: "File path is required." });
      return;
    }
    setLoading(true);
    setResult(null);
    const res = await processDocument({ filePath, method, chunkSize, overlap });
    setResult(res);
    setLoading(false);
    if (res.ok) onProcessed({ filePath, outputFile: res.data.output_file });
  }

  const overlapMax = Math.max(0, chunkSize - 1);

  return (
    <div className="panel-layout">
      {/* ── Controls ── */}
      <div className="controls-col">
        <p className="section-label">Document</p>

        <div className="field">
          <label>File path (.txt or .pdf)</label>
          <input
            type="text"
            placeholder="/path/to/document.pdf"
            value={filePath}
            onChange={(e) => setFilePath(e.target.value)}
          />
        </div>

        <div className="field">
          <label>Chunking method</label>
          <select value={method} onChange={(e) => setMethod(e.target.value)}>
            {METHODS.map((m) => (
              <option key={m} value={m}>{m}</option>
            ))}
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

        <button className="btn-primary" onClick={handleProcess} disabled={loading}>
          {loading ? "Processing…" : "⚙  Process Document"}
        </button>

        {loading && <Spinner label="Chunking document…" />}
      </div>

      {/* ── Results ── */}
      <div className="results-col">
        <p className="section-label">Result</p>

        {!result && !loading && (
          <p style={{ color: "var(--muted)", fontSize: ".82rem" }}>
            Configure settings and click <em>Process Document</em>.
          </p>
        )}

        {result && result.ok && (
          <>
            <div className="msg success">
              Document processed successfully.
            </div>
            <div className="stat-row">
              <div className="stat-badge">
                chunks<span>{result.data.chunks}</span>
              </div>
              <div className="stat-badge" style={{ flex: 1, wordBreak: "break-all" }}>
                output<span style={{ color: "var(--muted)" }}>
                  {result.data.output_file.split("/").pop()}
                </span>
              </div>
            </div>
            <div className="msg info" style={{ fontSize: ".78rem", wordBreak: "break-all" }}>
              {result.data.output_file}
            </div>
          </>
        )}

        {result && !result.ok && (
          <div className="msg error">{result.error}</div>
        )}
      </div>
    </div>
  );
}
