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

export async function askQuestion(question) {
  return postJson("/api/ask", { question }, "Query");
}

export async function addArxiv(url) {
  return postJson("/api/arxiv", { url }, "Import");
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
