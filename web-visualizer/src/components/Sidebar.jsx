import React, { useState } from "react";

function Sidebar({
  conversations,
  activeId,
  onSelect,
  onNew,
  onDelete,
  onRename,
  papers,
  onDeletePaper,
  onUploadFiles,
  onAddArxiv,
  uploading,
  uploadStatus,
  jobs = [],
  error,
}) {
  const [files, setFiles] = useState([]);
  const [arxivUrl, setArxivUrl] = useState("");
  const [editingId, setEditingId] = useState(null);
  const [draft, setDraft] = useState("");

  const commitRename = () => {
    if (editingId) onRename(editingId, draft);
    setEditingId(null);
  };

  return (
    <aside className="rail">
      <section className="block">
        <button className="primary new-chat" type="button" onClick={onNew}>
          New chat
        </button>
        {conversations.length > 0 && (
          <ul className="convos">
            {conversations.map((c) => (
              <li key={c.id} className={c.id === activeId ? "active" : ""}>
                {editingId === c.id ? (
                  <input
                    type="text"
                    className="rename"
                    value={draft}
                    autoFocus
                    onChange={(e) => setDraft(e.target.value)}
                    onBlur={commitRename}
                    onKeyDown={(e) => {
                      if (e.key === "Enter") commitRename();
                      if (e.key === "Escape") setEditingId(null);
                    }}
                  />
                ) : (
                  <button
                    type="button"
                    className="convo"
                    title="Double-click to rename"
                    onClick={() => onSelect(c.id)}
                    onDoubleClick={() => {
                      setEditingId(c.id);
                      setDraft(c.title || "");
                    }}
                  >
                    {c.title || "New chat"}
                  </button>
                )}
                <button
                  type="button"
                  className="delete"
                  title="Delete conversation"
                  aria-label={`Delete conversation ${c.title || "New chat"}`}
                  onClick={() => onDelete(c.id)}
                >
                  ×
                </button>
              </li>
            ))}
          </ul>
        )}
      </section>

      <section className="block">
        <h2 className="eyebrow">Add a paper</h2>
        <form
          onSubmit={async (e) => {
            e.preventDefault();
            const url = arxivUrl.trim();
            if (!url) return;
            if (await onAddArxiv(url)) setArxivUrl("");
          }}
        >
          <input
            type="text"
            value={arxivUrl}
            onChange={(e) => setArxivUrl(e.target.value)}
            placeholder="https://arxiv.org/abs/1706.03762"
          />
          <button type="submit" disabled={!arxivUrl.trim() || uploading}>
            {uploading ? "Reading paper…" : "Fetch from arXiv"}
          </button>
        </form>

        <p className="or">or</p>

        <form
          onSubmit={async (e) => {
            e.preventDefault();
            if (files.length && (await onUploadFiles(files))) setFiles([]);
          }}
        >
          <label className="file">
            <input
              type="file"
              accept="application/pdf"
              multiple
              onChange={(e) => setFiles(Array.from(e.target.files || []))}
            />
            <span>
              {files.length === 0
                ? "Choose PDFs, or drop them anywhere…"
                : files.length === 1
                  ? files[0].name
                  : `${files.length} PDFs selected`}
            </span>
          </label>
          <button type="submit" disabled={!files.length || uploading}>
            {uploading ? "Reading…" : "Add to corpus"}
          </button>
        </form>
        {jobs.length > 0 && (
          <ul className="jobs">
            {jobs.map((j, i) => (
              <li key={i} className={j.status}>
                <span className="job-name">{j.name}</span>
                <span className="job-status">
                  {j.status}
                  {j.detail ? ` — ${j.detail}` : ""}
                </span>
              </li>
            ))}
          </ul>
        )}
        {uploadStatus && <p className="status">{uploadStatus}</p>}
        {error && <p className="error-inline">{error}</p>}
      </section>

      {papers.length > 0 && (
        <section className="block">
          <h2 className="eyebrow">
            Library <span className="count">{papers.length}</span>
          </h2>
          <ul className="library">
            {papers.map((p) => (
              <li key={p.file} title={p.file}>
                <span className="lib-title">{p.title}</span>
                <button
                  type="button"
                  className="delete"
                  title="Remove from corpus"
                  aria-label={`Remove ${p.title || p.file} from corpus`}
                  onClick={() => onDeletePaper(p.file)}
                >
                  ×
                </button>
              </li>
            ))}
          </ul>
          <a className="export-bib" href="/api/bibtex" download="library.bib">
            Export BibTeX
          </a>
        </section>
      )}
    </aside>
  );
}

export default Sidebar;
