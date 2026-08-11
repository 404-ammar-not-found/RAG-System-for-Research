import React from "react";
import { TYPE_COLORS } from "../utils/graphData.js";

const LAYERS = [
  ["overview", "Overview"],
  ["passages", "Passages"],
  ["all", "Everything"],
];

/**
 * The graph presented the way a paper presents a figure: a plate, then a
 * caption that says what you are looking at, then the controls. The caption is
 * live — it is the honest answer to "what is on screen right now", which means
 * it reports an empty entity layer as empty instead of captioning a cloud of
 * passages as a knowledge graph.
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
  census,
  shown,
  openCount,
  onCollapseAll,
  onExpandAll,
  loading,
}) {
  const { papers = 0, entities = 0, passages = 0 } = census || {};

  const subject =
    layer === "passages" ? "Passage map" : layer === "all" ? "Corpus map" : "Corpus overview";

  const caption = () => {
    if (loading) return "Loading corpus…";
    const parts = [`${papers} papers`];
    parts.push(entities ? `${entities} entities` : "no entities yet");
    if (layer === "overview") {
      parts.push(
        openCount ? `${openCount} opened, ${shown.nodes} nodes drawn` : `${passages} passages folded in`
      );
    } else {
      parts.push(`${passages} passages`);
    }
    return `${subject} — ${parts.join(", ")}.`;
  };

  // Only advertise the types actually present. A legend for eight entity types
  // above a plate containing none of them is the figure lying about its data.
  const legend =
    layer === "passages"
      ? []
      : Object.entries(TYPE_COLORS).filter(([type]) => (census?.types || {})[type]);

  return (
    <div className="plate-wrap">
      <figure className="plate">
        <div className="canvas">{children}</div>
        <figcaption>
          <span className="fig-no">Figure 1</span>
          <span className="fig-text">{caption()}</span>
          {layer === "overview" && (
            <span className="fig-hint">
              Click a paper to open it, or{" "}
              <button
                type="button"
                className="linkish"
                onClick={openCount ? onCollapseAll : onExpandAll}
              >
                {openCount ? "close all" : "open all"}
              </button>
            </span>
          )}
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
          <span>{layer === "passages" ? "Paper" : "Type"}</span>
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
          The entity layer is empty — this shows papers and their passages only. Entities and the
          relations between them appear once an ingest runs graph extraction.
        </p>
      )}
    </div>
  );
}

export default FigurePlate;
