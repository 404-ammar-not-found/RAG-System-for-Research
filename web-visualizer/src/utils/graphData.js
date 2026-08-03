export const STORAGE_KEY = "knowledge-graph-cache";
export const EMPTY_GRAPH = { nodes: [], links: [] };

// Highlighter inks — the colours you'd actually mark up a printed paper with.
// Muted enough to sit on white without vibrating, distinct enough to tell a
// Dataset from a Metric at a glance. Fixed per type so a Dataset is the same
// colour every session; anything unrecognised falls back to the hash.
export const TYPE_COLORS = {
  Model: "#1F6F4A",
  Dataset: "#C2603F",
  Method: "#5251C4",
  Metric: "#C2325F",
  Task: "#0F7B8A",
  Paper: "#3F5265",
  Author: "#8A6A2F",
  Institution: "#6B7A3A",
};

export const CHUNK_COLOR = "#A9B6AF";

export const colorForGroup = (group) => {
  if (TYPE_COLORS[group]) return TYPE_COLORS[group];
  const g = group || "ungrouped";
  let hash = 0;
  for (let i = 0; i < g.length; i++) {
    hash = (hash * 31 + g.charCodeAt(i)) >>> 0;
  }
  // Papers (the chunk layer) get muted inks in the same register as the type
  // palette, rather than saturated hues that fight the entity colours.
  return `hsl(${hash % 360}, 32%, 46%)`;
};

/** Split an answer on [chunk-id] citations so they can render as numbered marks. */
export const parseCitations = (answer, matches) => {
  const order = [];
  const indexOf = (id) => {
    const known = matches.findIndex((m) => (m.metadata || {}).id === id);
    if (known === -1) return null;
    if (!order.includes(id)) order.push(id);
    return order.indexOf(id) + 1;
  };

  const parts = [];
  const re = /\[([A-Za-z0-9._-]+chunk-\d+)\]/g;
  let last = 0;
  let match;
  while ((match = re.exec(answer)) !== null) {
    const n = indexOf(match[1]);
    if (n === null) continue;
    if (match.index > last) parts.push({ text: answer.slice(last, match.index) });
    parts.push({ cite: n, id: match[1] });
    last = match.index + match[0].length;
  }
  if (last < answer.length) parts.push({ text: answer.slice(last) });
  return { parts, order };
};

export const LAYERS = ["entities", "chunks", "both"];

export const filterByLayer = (nodes, layer) => {
  if (layer === "both") return nodes;
  const want = layer === "chunks" ? "chunk" : "entity";
  // Nodes without `kind` predate the v2 schema; treat them as chunks.
  const picked = nodes.filter((n) => (n.kind || "chunk") === want);
  // An empty layer would render a blank screen and look like a broken app.
  // Before the graph has been built there are no entities at all, so fall
  // back rather than show nothing.
  return picked.length || !nodes.length ? picked : nodes;
};

/** True when the graph has no entity nodes yet (graph store not yet built). */
export const hasEntities = (nodes) => nodes.some((n) => n.kind === "entity");

export const buildGraphUrl = (base = import.meta.env.BASE_URL || "/") => {
  const normalized = base.endsWith("/") ? base : `${base}/`;
  return `${normalized}data/knowledge-graph.json`;
};

export const isGraphShape = (data) =>
  Boolean(data && Array.isArray(data.nodes) && Array.isArray(data.links));

export const loadCachedGraph = () => {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return null;
    const parsed = JSON.parse(raw);
    return isGraphShape(parsed) ? parsed : null;
  } catch (err) {
    console.warn("Failed to read cached graph", err);
    return null;
  }
};

export const saveCachedGraph = (data) => {
  try {
    if (!isGraphShape(data)) return;
    localStorage.setItem(STORAGE_KEY, JSON.stringify(data));
  } catch (err) {
    console.warn("Failed to cache graph", err);
  }
};
