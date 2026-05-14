import axios from "axios";

const client = axios.create({
  baseURL: "/api",
  headers: { "Content-Type": "application/json" },
  timeout: 60_000,
});

function extractError(err) {
  const data = err.response?.data;
  if (!data) return err.message || "Network error — is the backend running?";
  if (typeof data.detail === "string") return data.detail;
  if (Array.isArray(data.detail))
    return data.detail.map((e) => `${e.field}: ${e.message}`).join(" | ");
  return JSON.stringify(data);
}

export async function processDocument({ filePath, method, chunkSize, overlap }) {
  try {
    const { data } = await client.post("/process", {
      file_path: filePath,
      method,
      chunk_size: chunkSize,
      overlap,
    });
    return { ok: true, data };
  } catch (err) {
    return { ok: false, error: extractError(err) };
  }
}

export async function processText({ text, method, chunkSize, overlap }) {
  try {
    const { data } = await client.post("/process-text", {
      text,
      method,
      chunk_size: chunkSize,
      overlap,
    });
    return { ok: true, data };
  } catch (err) {
    return { ok: false, error: extractError(err) };
  }
}

export async function uploadAndProcess({ file, method, chunkSize, overlap }) {
  try {
    const form = new FormData();
    form.append("file", file);
    form.append("method", method);
    form.append("chunk_size", String(chunkSize));
    form.append("overlap", String(overlap));
    const { data } = await client.post("/process-upload", form, {
      headers: { "Content-Type": "multipart/form-data" },
    });
    return { ok: true, data };
  } catch (err) {
    return { ok: false, error: extractError(err) };
  }
}

export async function embedChunks(chunks) {
  try {
    const { data } = await client.post("/embed-chunks", { chunks });
    return { ok: true, data };
  } catch (err) {
    return { ok: false, error: extractError(err) };
  }
}

export async function askChunks({ chunks, query, topK = 3 }) {
  try {
    const { data } = await client.post("/ask-chunks", {
      chunks,
      query,
      top_k: topK,
    });
    return { ok: true, data };
  } catch (err) {
    return { ok: false, error: extractError(err) };
  }
}

export async function askQuestion({ query, filePath, topK = 3 }) {
  try {
    const { data } = await client.post("/ask", {
      query,
      file_path: filePath,
      top_k: topK,
    });
    return { ok: true, data };
  } catch (err) {
    return { ok: false, error: extractError(err) };
  }
}
