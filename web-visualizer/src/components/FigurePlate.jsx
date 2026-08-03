import React from "react";
import { TYPE_COLORS } from "../utils/graphData.js";

const LAYERS = [
  ["entities", "Entities"],
  ["chunks", "Passages"],
  ["both", "Both"],
];

/**
 * The graph presented the way a paper presents a figure: a plate, then a
 * caption that says what you are looking at, then the controls. The caption is
 * live — it is the honest answer to "what is on screen right now".
 */
function FigurePlate({
  children,
  layer,
  setLayer,
  groupFilter,
  setGroupFilter,
  groups,
  showEdges,
  setShowEdges,
  showLabels,
  setShowLabels,
  hasEntities,
  nodeCount,
  linkCount,
  loading,
}) {
  const subject = layer === "chunks" ? "Passage map" : layer === "both" ? "Corpus map" : "Entity graph";
  const legend = layer === "chunks" ? [] : Object.entries(TYPE_COLORS);

  return (
    <div className="plate-wrap">
      <figure className="plate">
        <div className="canvas">{children}</div>
        <figcaption>
          <span className="fig-no">Figure 1</span>
          <span className="fig-text">
            {loading
              ? "Loading corpus…"
              : `${subject} — ${nodeCount} nodes, ${linkCount} links.` +
                (layer === "chunks" ? " Coloured by paper." : " Coloured by type.")}
          </span>
        </figcaption>
      </figure>

      {legend.length > 0 && (
        <ul className="legend">
          {legend.map(([type, color]) => (
            <li key={type}>
              <span className="swatch" style={{ background: color }} />
              {type}
            </li>
          ))}
        </ul>
      )}

      <div className="figure-controls">
        <div className="segmented" role="group" aria-label="Graph layer">
          {LAYERS.map(([value, label]) => (
            <button
              key={value}
              type="button"
              className={layer === value ? "on" : ""}
              aria-pressed={layer === value}
              onClick={() => setLayer(value)}
            >
              {label}
            </button>
          ))}
        </div>

        <label className="inline-field">
          <span>{layer === "chunks" ? "Paper" : "Type"}</span>
          <select value={groupFilter} onChange={(e) => setGroupFilter(e.target.value)}>
            {groups.map((g) => (
              <option key={g} value={g}>
                {g === "all" ? "All" : g}
              </option>
            ))}
          </select>
        </label>

        <label className="check">
          <input
            type="checkbox"
            checked={showEdges}
            onChange={(e) => setShowEdges(e.target.checked)}
          />
          <span>Links</span>
        </label>

        <label className="check">
          <input
            type="checkbox"
            checked={showLabels}
            onChange={(e) => setShowLabels(e.target.checked)}
          />
          <span>Names</span>
        </label>
      </div>

      {!hasEntities && !loading && (
        <p className="notice">
          No entities yet — showing passages. Add papers and run an ingest to build the
          knowledge graph.
        </p>
      )}
    </div>
  );
}

export default FigurePlate;
