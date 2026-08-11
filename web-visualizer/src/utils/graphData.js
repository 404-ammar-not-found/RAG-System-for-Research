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

export const LAYERS = ["overview", "passages", "all"];
export const PAPER_PREFIX = "paper::";

const endpointId = (endpoint) =>
  endpoint && typeof endpoint === "object" ? endpoint.id : endpoint;

const stem = (path = "") => path.split("/").pop().replace(/\.pdf$/i, "");

/** Long titles wrap badly as sprite labels; a paper is recognisable by its head. */
const shorten = (text, max = 32) =>
  text.length <= max ? text : `${text.slice(0, max - 1).trimEnd()}…`;

/**
 * Fold the flat export into a hierarchy.
 *
 * The exporter emits one node per passage, which is 700+ equally-weighted dots
 * with no structure to read — the corpus rendered as a swarm. Nothing in the
 * data says "this is a paper", even though every passage carries the metadata
 * to derive one. So: synthesise a node per paper, parent its passages to it,
 * and roll entity→passage provenance up to entity→paper.
 *
 * That last edge is the one worth looking at. An entity linked to three papers
 * is the cross-paper resolution this whole system exists to produce, and it is
 * invisible while provenance points at individual passages.
 */
export function withPaperHubs(graph) {
  const nodes = [];
  const links = [];
  const papers = new Map();
  const paperOfChunk = new Map();

  // Node objects are created here and never re-created.
  //
  // The force simulation stores x/y/vx/vy *on the objects it is given*. Cloning
  // them per render (to attach a colour, say) hands the graph a fresh set with
  // no positions every time, so the layout re-initialises on every hover and
  // the whole figure collapses to the origin. Colour is therefore resolved once,
  // right here, and identity is stable for the lifetime of the data.
  for (const node of graph.nodes || []) {
    const kind = node.kind || "chunk"; // pre-v2 nodes are passages
    if (kind !== "chunk") {
      nodes.push({ ...node, kind, color: node.color || colorForGroup(node.type) });
      continue;
    }
    const group = node.group || "unknown";
    const id = PAPER_PREFIX + group;
    let paper = papers.get(id);
    if (!paper) {
      const title = node.paper_title || stem(group);
      paper = {
        id,
        kind: "paper",
        group,
        name: shorten(title),
        title,
        label: title,
        color: colorForGroup(group),
        chunkCount: 0,
        entityCount: 0,
        sections: new Set(),
      };
      papers.set(id, paper);
    }
    paper.chunkCount += 1;
    if (node.section) paper.sections.add(node.section);
    paperOfChunk.set(node.id, id);
    nodes.push({ ...node, kind, paper: id, group, color: CHUNK_COLOR });
    links.push({ source: id, target: node.id, kind: "contains" });
  }

  // Entity → paper, aggregated from the per-passage `mentions` edges. One edge
  // per (entity, paper) pair carrying how many passages back it.
  //
  // `from_paper` edges cover the entities no passage can reach: a section kept
  // out of the passage store — references above all — still produced entities,
  // and they belong to their paper even though there is no chunk to cite. They
  // add zero to the passage count, so a real citation always outweighs them.
  const bridges = new Map(); // entity id -> paper id -> passages backing it
  for (const link of graph.links || []) {
    let paperId = null;
    if (link.kind === "mentions") paperId = paperOfChunk.get(endpointId(link.target));
    else if (link.kind === "from_paper") paperId = PAPER_PREFIX + endpointId(link.target);
    else continue;
    if (!paperId || !papers.has(paperId)) continue;

    const entityId = endpointId(link.source);
    // A map per entity rather than one keyed by a joined string: the joined
    // form silently produced links whose `source` was the whole key and whose
    // `target` was undefined the moment the join and the split disagreed about
    // their separator, and every bridge vanished from the figure.
    let perPaper = bridges.get(entityId);
    if (!perPaper) bridges.set(entityId, (perPaper = new Map()));
    perPaper.set(paperId, (perPaper.get(paperId) || 0) + (link.kind === "mentions" ? 1 : 0));
  }

  for (const [entityId, perPaper] of bridges) {
    for (const [paperId, passages] of perPaper) {
      links.push({ source: entityId, target: paperId, kind: "bridges", passages });
      const paper = papers.get(paperId);
      if (paper) paper.entityCount += 1;
    }
  }

  // How many papers each entity spans — the number that decides whether it is
  // a finding shared across the corpus or a detail of one paper.
  const spanOf = new Map();
  for (const [entityId, perPaper] of bridges) spanOf.set(entityId, perPaper.size);
  // Pin the papers to a ring, in title order.
  //
  // Papers share no edge with each other, so a force layout has nothing to
  // solve for them — it just scatters them differently on every load, at a
  // different scale, and the camera frames whatever it lands on. Fixing the
  // positions makes the figure identical every time, guarantees the spacing
  // labels need, and leaves the simulation to do the one job it is good at:
  // arranging a paper's passages around it once opened.
  // An ellipse, not a circle: the plate is about twice as wide as it is tall,
  // so a circle would waste the horizontal room and shrink every label to fit
  // the vertical. The empty middle is deliberate — it is where entities land,
  // pulled between the papers that mention them.
  const ordered = [...papers.values()].sort((a, b) => a.title.localeCompare(b.title));
  const rx = 150 + 34 * ordered.length;
  const ry = 90 + 16 * ordered.length;
  ordered.forEach((paper, index) => {
    const angle = (index / ordered.length) * Math.PI * 2 - Math.PI / 2;
    paper.sectionCount = paper.sections.size;
    delete paper.sections; // a Set would not survive the graph's own cloning
    paper.fx = Math.cos(angle) * rx;
    paper.fy = Math.sin(angle) * ry;
    paper.fz = 0;
    nodes.push(paper);
  });

  // Weights are normalised here so the renderer can stay dumb about scale.
  const maxChunks = Math.max(1, ...[...papers.values()].map((p) => p.chunkCount));
  const maxSpan = Math.max(1, ...spanOf.values());
  for (const node of nodes) {
    if (node.kind === "paper") node.weight = Math.sqrt(node.chunkCount / maxChunks);
    if (node.kind === "entity") {
      node.papers = spanOf.get(node.id) || 0;
      node.weight = Math.sqrt(node.papers / maxSpan);
    }
  }

  return { nodes, links: [...links, ...(graph.links || [])] };
}

// Which edge kinds each layer draws. `mentions` is never drawn: it is the raw
// entity→passage provenance, thousands of edges that produced the hairball, and
// `bridges` is the same information aggregated to something readable.
const LAYER_LINKS = {
  overview: new Set(["relation", "bridges", "contains", "sequence"]),
  passages: new Set(["contains", "sequence"]),
  all: new Set(["relation", "bridges", "contains", "sequence"]),
};

/**
 * The subgraph a layer shows. Passages are hidden in `overview` until their
 * paper is opened — 12 named hubs is a figure, 700 dots is a swarm.
 */
export function viewFor(hub, layer, expanded = new Set()) {
  const keep = new Set();
  for (const node of hub.nodes) {
    if (node.kind === "paper") keep.add(node.id);
    else if (node.kind === "entity") {
      if (layer !== "passages") keep.add(node.id);
    } else if (layer === "passages" || layer === "all" || expanded.has(node.paper)) {
      keep.add(node.id);
    }
  }

  const allowed = LAYER_LINKS[layer] || LAYER_LINKS.overview;
  const links = hub.links.filter(
    (link) =>
      allowed.has(link.kind) &&
      keep.has(endpointId(link.source)) &&
      keep.has(endpointId(link.target))
  );
  return { nodes: hub.nodes.filter((n) => keep.has(n.id)), links };
}

/** Counts for the caption — the honest answer to "what is on screen". */
export const censusOf = (graph) => {
  const census = { papers: 0, entities: 0, passages: 0, types: {} };
  for (const node of graph.nodes) {
    if (node.kind === "paper") census.papers += 1;
    else if (node.kind === "entity") {
      census.entities += 1;
      const type = node.type || "Entity";
      census.types[type] = (census.types[type] || 0) + 1;
    } else census.passages += 1;
  }
  return census;
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
