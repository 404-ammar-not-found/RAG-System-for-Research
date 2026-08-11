import React, { useMemo, useState } from "react";
import { parseCitations } from "../utils/graphData.js";

const shortSource = (source = "") => source.split("/").pop().replace(/\.pdf$/i, "");

function Answer({ answer, matches, onFocusNode }) {
  const { parts, order } = useMemo(
    () => parseCitations(answer, matches),
    [answer, matches]
  );

  if (!answer) {
    return (
      <p className="empty">
        Ask a question to see which papers answer it. Every claim comes back with a
        numbered citation you can open.
      </p>
    );
  }

  return (
    <div className="answer">
      {parts.map((part, i) =>
        part.cite ? (
          <button
            key={i}
            type="button"
            className="cite"
            title={part.id}
            onClick={() => onFocusNode(part.id)}
          >
            {part.cite}
          </button>
        ) : (
          <span key={i}>{part.text}</span>
        )
      )}
      {order.length === 0 && (
        <p className="caveat">No passage citations in this answer — treat it carefully.</p>
      )}
    </div>
  );
}

function ControlPanel({
  onAsk,
  asking,
  answer,
  qaError,
  matches,
  facts,
  onFocusNode,
  onUploadFile,
  onAddArxiv,
  uploading,
  uploadStatus,
}) {
  const [queryInput, setQueryInput] = useState("");
  const [file, setFile] = useState(null);
  const [arxivUrl, setArxivUrl] = useState("");

  const cited = useMemo(() => {
    const { order } = parseCitations(answer || "", matches || []);
    return order;
  }, [answer, matches]);

  // Cited passages first and numbered to match the answer; the rest are the
  // retrieved-but-unused context, which is useful but secondary.
  const sources = useMemo(() => {
    const list = matches || [];
    const byId = new Map(list.map((m) => [(m.metadata || {}).id, m]));
    const head = cited.map((id, i) => ({ match: byId.get(id), n: i + 1 })).filter((x) => x.match);
    const rest = list
      .filter((m) => !cited.includes((m.metadata || {}).id))
      .map((match) => ({ match, n: null }));
    return [...head, ...rest];
  }, [matches, cited]);

  return (
    <aside className="rail">
      <section className="block">
        <h2 className="eyebrow">Ask</h2>
        <form
          onSubmit={(e) => {
            e.preventDefault();
            onAsk(queryInput.trim());
          }}
        >
          <textarea
            value={queryInput}
            onChange={(e) => setQueryInput(e.target.value)}
            placeholder="Which papers use attention, and what did they evaluate on?"
            rows={3}
          />
          <button className="primary" type="submit" disabled={asking || !queryInput.trim()}>
            {asking ? "Reading…" : "Ask"}
          </button>
        </form>
        {qaError && <p className="error-inline">{qaError}</p>}
      </section>

      <section className="block">
        <h2 className="eyebrow">Answer</h2>
        <Answer answer={answer} matches={matches || []} onFocusNode={onFocusNode} />
      </section>

      {(facts || []).length > 0 && (
        <section className="block">
          <h2 className="eyebrow">
            Relations <span className="count">{facts.length}</span>
          </h2>
          <ul className="facts">
            {facts.map((fact, i) => (
              <li key={i}>
                <button type="button" onClick={() => onFocusNode(fact.source)}>
                  <span className="rel">{fact.name || "related"}</span>
                  <span className="fact">{fact.fact}</span>
                </button>
              </li>
            ))}
          </ul>
        </section>
      )}

      {sources.length > 0 && (
        <section className="block">
          <h2 className="eyebrow">
            Passages <span className="count">{sources.length}</span>
          </h2>
          <ol className="sources">
            {sources.map(({ match, n }, i) => {
              const meta = match.metadata || {};
              const text = (match.document || "").replace(/\s+/g, " ").trim();
              return (
                <li key={`${meta.id || i}`} className={n ? "used" : ""}>
                  <button type="button" onClick={() => onFocusNode(meta.id)}>
                    <span className="src-head">
                      <span className="marker">{n || "·"}</span>
                      <span className="src-title">{shortSource(meta.source)}</span>
                      <span className="src-section">{meta.section || ""}</span>
                    </span>
                    <span className="snippet">{text.slice(0, 150)}…</span>
                  </button>
                </li>
              );
            })}
          </ol>
        </section>
      )}

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
            if (file) await onUploadFile(file);
          }}
        >
          <label className="file">
            <input
              type="file"
              accept="application/pdf"
              onChange={(e) => setFile(e.target.files?.[0] || null)}
            />
            <span>{file ? file.name : "Choose a PDF…"}</span>
          </label>
          <button type="submit" disabled={!file || uploading}>
            {uploading ? "Reading paper…" : "Add to corpus"}
          </button>
        </form>
        {uploadStatus && <p className="status">{uploadStatus}</p>}
      </section>
    </aside>
  );
}

export default ControlPanel;
