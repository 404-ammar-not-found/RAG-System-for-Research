import React, { useEffect, useMemo, useState } from "react";
import { getGraph } from "../utils/api.js";

const ROW_H = 44;
const GUTTER = 140;

/** Which library papers cite which, and what a selected paper connects to.
 * ponytail: arc diagram over a year-ordered list — fine to ~40 papers; past
 * that the arcs spaghetti and a force layout earns its keep. */
function LiteratureMap({ paperCount }) {
  const [graph, setGraph] = useState({ papers: [], edges: [], shared: [] });
  const [selected, setSelected] = useState(null);

  useEffect(() => {
    getGraph().then(setGraph).catch(() => {});
  }, [paperCount]);

  const papers = useMemo(
    () => [...graph.papers].sort((a, b) => (a.year || 9999) - (b.year || 9999)),
    [graph.papers]
  );
  const index = useMemo(() => new Map(papers.map((p, i) => [p.file, i])), [papers]);
  const edges = useMemo(
    () => graph.edges.filter(([a, b]) => index.has(a) && index.has(b)),
    [graph.edges, index]
  );

  const detail = useMemo(() => {
    if (!selected) return null;
    return {
      paper: papers.find((p) => p.file === selected),
      cites: edges.filter(([a]) => a === selected).map(([, b]) => b),
      citedBy: edges.filter(([, b]) => b === selected).map(([a]) => a),
      shared: graph.shared
        .filter((s) => s.a === selected || s.b === selected)
        .map((s) => ({ file: s.a === selected ? s.b : s.a, concepts: s.concepts })),
    };
  }, [selected, edges, papers, graph.shared]);

  if (!papers.length) {
    return (
      <div className="litmap empty">
        <p>No bibliographic records yet — add papers, then come back.</p>
      </div>
    );
  }

  const rowY = (i) => i * ROW_H + ROW_H / 2;
  const titleOf = (file) => papers.find((p) => p.file === file)?.title || file;
  const height = papers.length * ROW_H;

  return (
    <div className="litmap">
      <div className="litmap-rows">
        <svg className="litmap-arcs" width={GUTTER} height={height} aria-hidden="true">
          {edges.map(([a, b], i) => {
            const y1 = rowY(index.get(a));
            const y2 = rowY(index.get(b));
            const bend = Math.min(Math.abs(y2 - y1) / 2 + 12, GUTTER - 8);
            const active = selected && (a === selected || b === selected);
            return (
              <path
                key={i}
                className={`arc ${active ? "active" : ""}`}
                d={`M ${GUTTER} ${y1} Q ${GUTTER - bend} ${(y1 + y2) / 2} ${GUTTER} ${y2}`}
              />
            );
          })}
        </svg>
        <ol className="litmap-list">
          {papers.map((p) => (
            <li
              key={p.file}
              style={{ height: ROW_H }}
              className={selected === p.file ? "selected" : ""}
            >
              <button
                type="button"
                onClick={() => setSelected(selected === p.file ? null : p.file)}
              >
                <span className="litmap-year">{p.year || "—"}</span>
                <span className="litmap-title">{p.title || p.file}</span>
                <span className="litmap-count">
                  {p.citationCount ? `${p.citationCount.toLocaleString()} cites` : ""}
                </span>
              </button>
            </li>
          ))}
        </ol>
      </div>
      {detail && (
        <aside className="litmap-detail">
          <p className="eyebrow">{detail.paper?.title || selected}</p>
          <h3>Cites in library</h3>
          {detail.cites.length ? (
            <ul>{detail.cites.map((f) => <li key={f}>{titleOf(f)}</li>)}</ul>
          ) : (
            <p className="muted">none</p>
          )}
          <h3>Cited by in library</h3>
          {detail.citedBy.length ? (
            <ul>{detail.citedBy.map((f) => <li key={f}>{titleOf(f)}</li>)}</ul>
          ) : (
            <p className="muted">none</p>
          )}
          <h3>Shares concepts with</h3>
          {detail.shared.length ? (
            <ul>
              {detail.shared.map((s) => (
                <li key={s.file}>
                  {titleOf(s.file)}: {s.concepts.slice(0, 4).join(", ")}
                </li>
              ))}
            </ul>
          ) : (
            <p className="muted">none</p>
          )}
          {detail.paper?.citers?.length > 0 && (
            <>
              <h3>Later work</h3>
              <ul>
                {detail.paper.citers.slice(0, 3).map((c, i) => (
                  <li key={i}>
                    {c.title}
                    {c.year ? ` (${c.year})` : ""}
                  </li>
                ))}
              </ul>
            </>
          )}
        </aside>
      )}
    </div>
  );
}

export default LiteratureMap;
