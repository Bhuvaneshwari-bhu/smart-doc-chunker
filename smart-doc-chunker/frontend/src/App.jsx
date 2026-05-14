import { useState } from "react";
import ProcessPanel from "./components/ProcessPanel.jsx";
import ChatPanel    from "./components/ChatPanel.jsx";
import ChunkerPanel from "./components/ChunkerPanel.jsx";

const TABS = ["Chunker", "Process", "Chat"];

export default function App() {
  const [tab,     setTab]     = useState("Chunker");
  const [session, setSession] = useState(null);  // { filePath, outputFile }

  function handleProcessed(info) {
    setSession(info);
    setTab("Chat");
  }

  return (
    <div className="app-shell">
      <header className="topbar">
        <span className="topbar-title">📄 Smart Doc Chunker</span>
        <nav className="topbar-tabs">
          {TABS.map((t) => (
            <button
              key={t}
              className={`tab-btn${tab === t ? " active" : ""}`}
              onClick={() => setTab(t)}
            >
              {t}
              {t === "Chat" && session && (
                <span style={{
                  marginLeft: ".4rem",
                  background: "rgba(255,255,255,.15)",
                  borderRadius: "3px",
                  padding: "0 .3rem",
                  fontSize: ".7rem",
                }}>●</span>
              )}
            </button>
          ))}
        </nav>
        {session && tab !== "Chunker" && (
          <span style={{
            marginLeft: "auto",
            fontSize: ".74rem",
            color: "var(--muted)",
            fontFamily: "var(--mono)",
            maxWidth: "360px",
            overflow: "hidden",
            textOverflow: "ellipsis",
            whiteSpace: "nowrap",
          }}>
            {session.filePath}
          </span>
        )}
      </header>

      <main className="main-content">
        {tab === "Chunker" && <ChunkerPanel />}
        {tab === "Process" && <ProcessPanel onProcessed={handleProcessed} />}
        {tab === "Chat"    && <ChatPanel    session={session} />}
      </main>
    </div>
  );
}
