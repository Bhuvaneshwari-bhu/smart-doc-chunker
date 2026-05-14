import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import streamlit as st

import loader
import cleaner
import chunker
import tokenizer
import embedding_store
import rag_chat

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Smart Doc Chunker",
    page_icon="📄",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------------------

if "embedded_chunks" not in st.session_state:
    st.session_state.embedded_chunks = []
if "stats" not in st.session_state:
    st.session_state.stats = None
if "history" not in st.session_state:
    st.session_state.history = []   # list of {query, answer, sources}
if "doc_processed" not in st.session_state:
    st.session_state.doc_processed = False

# ---------------------------------------------------------------------------
# Sidebar — document configuration
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("📄 Document Settings")
    st.divider()

    file_path = st.text_input(
        "File path",
        placeholder="/path/to/document.pdf",
        help="Absolute or relative path to a .pdf or .txt file.",
    )

    method = st.selectbox(
        "Chunking method",
        options=["recursive", "sliding"],
        index=0,
        help="recursive: paragraph → sentence hierarchy  |  sliding: overlapping token windows",
    )

    chunk_size = st.slider(
        "Chunk size (tokens)",
        min_value=50,
        max_value=1000,
        value=300,
        step=50,
    )

    overlap = st.slider(
        "Overlap (tokens)",
        min_value=0,
        max_value=200,
        value=50,
        step=10,
        help="Only used by the sliding window method.",
    )

    st.divider()
    process_btn = st.button("⚙️ Process Document", use_container_width=True, type="primary")

    if st.session_state.doc_processed and st.session_state.stats:
        st.divider()
        st.caption("Last processed document")
        s = st.session_state.stats
        col1, col2 = st.columns(2)
        col1.metric("Chunks",       s["total_chunks"])
        col2.metric("Total tokens", s["total_tokens"])
        col1.metric("Avg tokens",   f"{s['avg_tokens_per_chunk']:.0f}")
        col2.metric("Max tokens",   s["max_tokens"])

# ---------------------------------------------------------------------------
# Document processing
# ---------------------------------------------------------------------------

if process_btn:
    if not file_path or not file_path.strip():
        st.error("Enter a file path in the sidebar before processing.")
    elif not os.path.isfile(file_path):
        st.error(f"File not found: `{file_path}`")
    elif overlap >= chunk_size:
        st.error(f"Overlap ({overlap}) must be less than chunk size ({chunk_size}).")
    else:
        with st.spinner("Loading and processing document…"):
            try:
                raw_text     = loader.load_document(file_path)
                cleaned_text = cleaner.clean_text(raw_text)

                if not cleaned_text.strip():
                    st.error("Document is empty after cleaning.")
                else:
                    chunks = chunker.chunk_text(
                        text=cleaned_text,
                        method=method,
                        chunk_size=chunk_size,
                        overlap=overlap,
                    )

                    if not chunks:
                        st.error("No chunks were produced. Try a smaller chunk size.")
                    else:
                        stats    = tokenizer.analyze_chunks(chunks)
                        embedded = embedding_store.embed_chunks(chunks)

                        st.session_state.embedded_chunks = embedded
                        st.session_state.stats           = stats
                        st.session_state.doc_processed   = True
                        st.session_state.history         = []

                        st.success(
                            f"Processed **{os.path.basename(file_path)}** — "
                            f"{stats['total_chunks']} chunks, "
                            f"{stats['total_tokens']} tokens"
                        )

            except (FileNotFoundError, ValueError) as exc:
                st.error(f"Processing failed: {exc}")

# ---------------------------------------------------------------------------
# Main UI — header
# ---------------------------------------------------------------------------

st.title("💬 Chat with your Document")
st.caption("Upload a document in the sidebar, then ask questions below.")

if not st.session_state.doc_processed:
    st.info("👈 Configure and process a document using the sidebar to get started.")
    st.stop()

st.divider()

# ---------------------------------------------------------------------------
# Chat history
# ---------------------------------------------------------------------------

for entry in st.session_state.history:
    with st.chat_message("user"):
        st.write(entry["query"])

    with st.chat_message("assistant"):
        st.write(entry["answer"])

        with st.expander(f"📎 Retrieved chunks ({len(entry['sources'])} sources)"):
            for src in entry["sources"]:
                cid    = src["chunk_id"]
                tokens = next(
                    (c["tokens"] for c in st.session_state.embedded_chunks
                     if c["chunk_id"] == cid),
                    "—",
                )
                st.markdown(
                    f"**Chunk {cid}** &nbsp;·&nbsp; "
                    f"<span style='color:grey;font-size:0.85em'>{tokens} tokens</span>",
                    unsafe_allow_html=True,
                )
                st.caption(src["text"])
                st.divider()

# ---------------------------------------------------------------------------
# Chat input
# ---------------------------------------------------------------------------

query = st.chat_input("Ask a question about your document…")

if query:
    with st.chat_message("user"):
        st.write(query)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving…"):
            response = rag_chat.answer_query(
                query=query,
                embedded_chunks=st.session_state.embedded_chunks,
                top_k=3,
            )

        st.write(response["answer"])

        with st.expander(f"📎 Retrieved chunks ({len(response['sources'])} sources)"):
            for src in response["sources"]:
                cid    = src["chunk_id"]
                tokens = next(
                    (c["tokens"] for c in st.session_state.embedded_chunks
                     if c["chunk_id"] == cid),
                    "—",
                )
                st.markdown(
                    f"**Chunk {cid}** &nbsp;·&nbsp; "
                    f"<span style='color:grey;font-size:0.85em'>{tokens} tokens</span>",
                    unsafe_allow_html=True,
                )
                st.caption(src["text"])
                st.divider()

    st.session_state.history.append(response)
    st.rerun()
