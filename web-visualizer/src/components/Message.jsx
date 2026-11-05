import React, { useMemo, useState } from "react";
import {
  disagreements,
  highlightSegments,
  paperName,
  preprintNote,
  renderAnswerHtml,
  shortSource,
} from "../utils/citations.js";

const VISUAL = new Set(["figure", "table", "algorithm", "listing"]);
const STAGE_LABEL = {
  retrieving: "Searching the papers…",
  writing: "Writing…",
  verifying: "Verifying claims…",
};
const CAPTION_RE = /(table|figure|fig\.?|algorithm|listing)\s*\d+[.:]?/i;

function Sources({ matches, cited, openId, spanMap, stances }) {
  // Cited passages first and numbered to match the answer; the rest are the
  // retrieved-but-unused context, which is useful but secondary.
  const sources = useMemo(() => {
    const byId = new Map(matches.map((m) => [(m.metadata || {}).id, m]));
    const head = cited.map((id, i) => ({ match: byId.get(id), n: i + 1 })).filter((x) => x.match);
    const rest = matches
      .filter((m) => !cited.includes((m.metadata || {}).id))
      .map((match) => ({ match, n: null }));
    return [...head, ...rest];
  }, [matches, cited]);

  if (!sources.length) return null;

  return (
    <details className="msg-sources" open={Boolean(openId)}>
      <summary className="eyebrow">
        Sources <span className="count">{sources.length}</span>
      </summary>
      <ol className="sources">
        {sources.map(({ match, n }, i) => {
          const meta = match.metadata || {};
          const doc = match.document || "";
          const collapsed = doc.replace(/\s+/g, " ").trim();
          const isOpen = openId && meta.id === openId;
          const note = preprintNote(meta);
          const visual = VISUAL.has(meta.content_type);
          const caption = visual ? (doc.match(CAPTION_RE) || [""])[0] : "";
          return (
            <li key={meta.id || i} className={n ? "used" : ""} id={meta.id}>
              <div className={`source ${isOpen ? "open" : ""}`}>
                <span className="src-head">
                  <span className="marker">{n || "·"}</span>
                  <span className="src-title" title={shortSource(meta.source)}>
                    {paperName(meta)}
                  </span>
                  <span className="src-section">{meta.section || ""}</span>
                  {stances[meta.id]?.stance === "contradicts" && (
                    <span className="src-tag">pushes back</span>
                  )}
                  {note && <span className="src-tag stale">{note}</span>}
                </span>
                <span className="snippet">
                  {isOpen
                    ? // The open view keeps the raw text: claim-span offsets
                      // index into it, so it must not be whitespace-collapsed.
                      highlightSegments(doc, spanMap.get(meta.id) || []).map((s, j) =>
                        s.hit ? (
                          <mark key={j} className="hl">{s.text}</mark>
                        ) : (
                          <React.Fragment key={j}>{s.text}</React.Fragment>
                        )
                      )
                    : `${collapsed.slice(0, 150)}…`}
                </span>
                {isOpen && visual && (
                  <img
                    className="page-image"
                    loading="lazy"
                    alt={`Page ${(meta.page ?? 0) + 1} of ${paperName(meta)}`}
                    src={`/api/page-image?source=${encodeURIComponent(
                      shortSource(meta.source)
                    )}.pdf&page=${meta.page ?? 0}&caption=${encodeURIComponent(caption)}`}
                  />
                )}
              </div>
            </li>
          );
        })}
      </ol>
    </details>
  );
}

function Message({ message, isLast, streaming: busy, onSend, onRegenerate }) {
  const [openId, setOpenId] = useState(null);
  const [copied, setCopied] = useState(false);

  if (message.role === "user") {
    return <div className="msg user">{message.content}</div>;
  }

  const {
    content,
    matches = [],
    claims = [],
    stances = {},
    verification,
    warnings = [],
    relations = [],
    suggestions = [],
    streaming,
    stage,
    interrupted,
    note,
    error,
  } = message;
  const { html, order } = useMemo(
    () => renderAnswerHtml(content || "", matches, claims),
    [content, matches, claims]
  );
  const spanMap = useMemo(() => {
    const map = new Map();
    for (const c of claims)
      for (const s of c.spans || []) {
        if (!map.has(s.chunkId)) map.set(s.chunkId, []);
        map.get(s.chunkId).push(s);
      }
    return map;
  }, [claims]);
  const split = useMemo(() => disagreements(stances, matches), [stances, matches]);

  if (error) {
    return <div className="msg assistant error-inline">{error}</div>;
  }

  // Interrupted mid-stream: the partial text was never verified, so render it
  // raw (citation markup untrusted) with a caveat instead of discarding it.
  if (interrupted) {
    return (
      <div className="msg assistant">
        <div className="answer streaming">{content}</div>
        <p className="caveat">
          {note || "Stopped."} Answer is incomplete and its citations are unverified.
        </p>
      </div>
    );
  }

  const copy = () => {
    navigator.clipboard
      ?.writeText(content || "")
      .then(() => {
        setCopied(true);
        setTimeout(() => setCopied(false), 1500);
      })
      .catch(() => {});
  };

  const onAnswerClick = (e) => {
    const id = e.target?.dataset?.cite;
    if (id) setOpenId((prev) => (prev === id ? null : id));
  };

  const counts = verification?.counts || {};
  const countLine = ["supported", "partial", "unsupported", "fabricated", "unchecked"]
    .filter((v) => counts[v])
    .map((v) => `${counts[v]} ${v}`)
    .join(" · ");

  return (
    <div className="msg assistant">
      {warnings.map((w, i) => (
        <p key={i} className={`msg-warning ${w.level}`}>
          <strong>{w.paper}</strong>: {w.message}
        </p>
      ))}
      {streaming ? (
        <div className="answer streaming" aria-live="polite" aria-busy="true">
          {content || (
            <span className="thinking">{STAGE_LABEL[stage] || "Reading the papers…"}</span>
          )}
          {content && stage === "verifying" && (
            <p className="thinking">{STAGE_LABEL.verifying}</p>
          )}
        </div>
      ) : (
        <div
          className="answer"
          onClick={onAnswerClick}
          dangerouslySetInnerHTML={{ __html: html }}
        />
      )}
      {!streaming && countLine && <p className="verify-line">{countLine}</p>}
      {!streaming && content && order.length === 0 && (
        <p className="caveat">No passage citations in this answer — treat it carefully.</p>
      )}
      {!streaming && verification?.flagged?.length > 0 && (
        <div className="flagged">
          <p className="eyebrow">Unverified claims</p>
          <ul>
            {verification.flagged.map((c, i) => (
              <li key={i}>
                <span className="verdict">{c.verdict}</span> {c.text}
              </li>
            ))}
          </ul>
        </div>
      )}
      {!streaming && split.length > 0 && (
        <div className="flagged">
          <p className="eyebrow">
            Disagreement <span className="count">{split.length}</span>
          </p>
          <ul>
            {split.map((d) => (
              <li key={d.paper}>
                <span className="verdict">pushes back</span>{" "}
                <strong>{d.paper}</strong>
                {d.year ? ` (${d.year})` : ""} — {d.claim}
              </li>
            ))}
          </ul>
        </div>
      )}
      {!streaming && (
        <Sources
          matches={matches}
          cited={order}
          openId={openId}
          spanMap={spanMap}
          stances={stances}
        />
      )}
      {!streaming && relations.length > 0 && (
        <details className="relations">
          <summary>
            Relationships considered <span className="count">{relations.length}</span>
          </summary>
          <p className="muted">Extracted by an LLM to guide search. Only cited passages are evidence.</p>
          <ul>
            {relations.map((r, i) => (
              <li key={i}>
                {r.source} <span className="relation">{r.relation.replaceAll("_", " ")}</span> {r.target}
              </li>
            ))}
          </ul>
        </details>
      )}
      {!streaming && content && (
        <div className="msg-actions">
          <button type="button" className="action" onClick={copy}>
            {copied ? "Copied" : "Copy"}
          </button>
          {isLast && !busy && onRegenerate && (
            <button type="button" className="action" onClick={onRegenerate}>
              Regenerate
            </button>
          )}
        </div>
      )}
      {!streaming && isLast && !busy && suggestions.length > 0 && (
        <div className="chips followups">
          {suggestions.map((s) => (
            <button key={s} type="button" className="chip" onClick={() => onSend?.(s)}>
              {s}
            </button>
          ))}
        </div>
      )}
    </div>
  );
}

export default Message;
