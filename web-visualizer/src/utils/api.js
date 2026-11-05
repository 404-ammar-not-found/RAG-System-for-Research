export async function uploadPdf(file) {
  const form = new FormData();
  form.append("file", file);

  const response = await fetch("/api/upload", {
    method: "POST",
    body: form,
  });

  const data = await response.json().catch(() => ({}));
  if (!response.ok) {
    const detail = data?.detail || `Upload failed (${response.status})`;
    throw new Error(detail);
  }
  return data;
}

export async function addArxiv(url) {
  return postJson("/api/arxiv", { url }, "Import");
}

export async function getGraph() {
  const response = await fetch("/api/graph");
  const data = await response.json().catch(() => ({}));
  return { papers: data.papers || [], edges: data.edges || [] };
}

export async function deletePaper(file) {
  const response = await fetch(`/api/papers/${encodeURIComponent(file)}`, {
    method: "DELETE",
  });
  const data = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(data?.detail || `Delete failed (${response.status})`);
  }
  return data;
}

export async function getPapers() {
  const response = await fetch("/api/papers");
  const data = await response.json().catch(() => ({}));
  return Array.isArray(data.papers) ? data.papers : [];
}

/** POST the conversation and dispatch each NDJSON event to onEvent:
 * {type:"stage", name}, {type:"token", text}, then
 * {type:"done", answer, matches, suggestions, ...}, or {type:"error", detail}. */
export async function streamChat(messages, onEvent, signal) {
  const response = await fetch("/api/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ messages }),
    signal,
  });
  if (!response.ok) {
    const data = await response.json().catch(() => ({}));
    throw new Error(data?.detail || `Chat failed (${response.status})`);
  }

  const emit = (line) => {
    if (!line.trim()) return;
    try {
      onEvent(JSON.parse(line));
    } catch {
      /* malformed line — skip it rather than kill the stream */
    }
  };

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split("\n");
    buffer = lines.pop();
    lines.forEach(emit);
  }
  emit(buffer);
}

async function postJson(path, body, what) {
  const response = await fetch(path, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(body),
  });

  const data = await response.json().catch(() => ({}));
  if (!response.ok) {
    const detail = data?.detail || `${what} failed (${response.status})`;
    throw new Error(detail);
  }
  return data;
}
