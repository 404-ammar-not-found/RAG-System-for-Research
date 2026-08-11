import { describe, expect, it } from "vitest";
import {
  PAPER_PREFIX,
  buildGraphUrl,
  censusOf,
  colorForGroup,
  isGraphShape,
  viewFor,
  withPaperHubs,
} from "./graphData";

// Two papers, three passages, one entity mentioned in both papers.
const RAW = {
  nodes: [
    { id: "c1", kind: "chunk", group: "data/a.pdf", paper_title: "Paper A", section: "Intro" },
    { id: "c2", kind: "chunk", group: "data/a.pdf", paper_title: "Paper A", section: "Method" },
    { id: "c3", kind: "chunk", group: "data/b.pdf", paper_title: "Paper B", section: "Intro" },
    { id: "e1", kind: "entity", name: "Transformer", type: "Model" },
  ],
  links: [
    { source: "c1", target: "c2", kind: "sequence" },
    { source: "e1", target: "c1", kind: "mentions" },
    { source: "e1", target: "c2", kind: "mentions" },
    { source: "e1", target: "c3", kind: "mentions" },
  ],
};

describe("graphData utils", () => {
  it("produces deterministic colors per group", () => {
    const g1 = colorForGroup("alpha");
    const g2 = colorForGroup("alpha");
    const g3 = colorForGroup("beta");
    expect(g1).toBe(g2);
    expect(g1).not.toBe(g3);
  });

  it("gives known entity types a fixed color", () => {
    expect(colorForGroup("Dataset")).toBe(colorForGroup("Dataset"));
    expect(colorForGroup("Dataset")).not.toBe(colorForGroup("Model"));
  });

  it("builds a graph URL with normalized base", () => {
    expect(buildGraphUrl("/" )).toBe("/data/knowledge-graph.json");
    expect(buildGraphUrl("/foo")).toBe("/foo/data/knowledge-graph.json");
    expect(buildGraphUrl("/bar/" )).toBe("/bar/data/knowledge-graph.json");
  });

  it("derives one paper node per source, sized by passage count", () => {
    const hub = withPaperHubs(RAW);
    const papers = hub.nodes.filter((n) => n.kind === "paper");
    expect(papers.map((p) => p.title).sort()).toEqual(["Paper A", "Paper B"]);
    const a = papers.find((p) => p.title === "Paper A");
    expect(a.chunkCount).toBe(2);
    expect(a.sectionCount).toBe(2);
    // The bigger paper carries the heavier weight, normalised to 1.
    expect(a.weight).toBe(1);
    expect(papers.find((p) => p.title === "Paper B").weight).toBeLessThan(1);
  });

  it("rolls per-passage provenance up to entity->paper bridges", () => {
    const hub = withPaperHubs(RAW);
    const bridges = hub.links.filter((l) => l.kind === "bridges");
    // Three `mentions` edges over two papers collapse to two bridges, and the
    // one spanning both papers is what makes cross-paper resolution visible.
    expect(bridges).toHaveLength(2);
    expect(bridges.find((b) => b.target === `${PAPER_PREFIX}data/a.pdf`).passages).toBe(2);
    expect(hub.nodes.find((n) => n.id === "e1").papers).toBe(2);
  });

  it("only ever emits links between nodes that exist", () => {
    // A derived link whose endpoints are not real node ids is invisible rather
    // than wrong-looking: the view filters it out and the figure quietly loses
    // a whole relationship. This caught bridges being emitted with the joined
    // map key as their `source` and nothing at all as their `target`.
    const hub = withPaperHubs(RAW);
    const ids = new Set(hub.nodes.map((n) => n.id));
    for (const link of hub.links) {
      const source = typeof link.source === "object" ? link.source.id : link.source;
      const target = typeof link.target === "object" ? link.target.id : link.target;
      // `mentions`/`from_paper` are the raw export's own edges, left in place.
      if (link.kind === "mentions" || link.kind === "from_paper") continue;
      expect(ids.has(source), `${link.kind} source ${source}`).toBe(true);
      expect(ids.has(target), `${link.kind} target ${target}`).toBe(true);
    }
  });

  it("gives entities from unindexed sections their paper anyway", () => {
    // References are deliberately kept out of the passage store, so entities
    // extracted there can never reach a chunk — but the paper is still known.
    const hub = withPaperHubs({
      nodes: [
        ...RAW.nodes,
        { id: "e2", kind: "entity", name: "Some Author", type: "Author" },
      ],
      links: [...RAW.links, { source: "e2", target: "data/a.pdf", kind: "from_paper" }],
    });
    const bridge = hub.links.find((l) => l.kind === "bridges" && l.source === "e2");
    expect(bridge.target).toBe(`${PAPER_PREFIX}data/a.pdf`);
    // No passage cites it, so it must not outrank one that is genuinely cited.
    expect(bridge.passages).toBe(0);
  });

  it("folds passages away in the overview and opens them per paper", () => {
    const hub = withPaperHubs(RAW);
    const closed = viewFor(hub, "overview", new Set());
    expect(closed.nodes.map((n) => n.kind).sort()).toEqual(["entity", "paper", "paper"]);

    const opened = viewFor(hub, "overview", new Set([`${PAPER_PREFIX}data/a.pdf`]));
    const chunks = opened.nodes.filter((n) => n.kind === "chunk");
    expect(chunks.map((n) => n.id).sort()).toEqual(["c1", "c2"]);
  });

  it("never draws raw entity->passage edges — that was the hairball", () => {
    const hub = withPaperHubs(RAW);
    for (const layer of ["overview", "passages", "all"]) {
      const view = viewFor(hub, layer, new Set());
      expect(view.links.some((l) => l.kind === "mentions")).toBe(false);
    }
  });

  it("drops entities from the passage layer and keeps papers everywhere", () => {
    const hub = withPaperHubs(RAW);
    const view = viewFor(hub, "passages", new Set());
    expect(view.nodes.some((n) => n.kind === "entity")).toBe(false);
    expect(view.nodes.filter((n) => n.kind === "paper")).toHaveLength(2);
    expect(view.nodes.filter((n) => n.kind === "chunk")).toHaveLength(3);
  });

  it("pins papers to a stable ring so the figure is the same every load", () => {
    const a = withPaperHubs(RAW).nodes.filter((n) => n.kind === "paper");
    const b = withPaperHubs(RAW).nodes.filter((n) => n.kind === "paper");
    expect(a.map((p) => [p.title, p.fx, p.fy])).toEqual(b.map((p) => [p.title, p.fx, p.fy]));
    // Distinct positions, and flat: depth only shrinks labels here.
    expect(new Set(a.map((p) => `${p.fx},${p.fy}`)).size).toBe(a.length);
    expect(a.every((p) => p.fz === 0)).toBe(true);
  });

  it("counts what is on screen for the caption", () => {
    expect(censusOf(withPaperHubs(RAW))).toEqual({
      papers: 2,
      entities: 1,
      passages: 3,
      types: { Model: 1 },
    });
  });

  it("validates graph shape", () => {
    expect(isGraphShape({ nodes: [], links: [] })).toBe(true);
    expect(isGraphShape({ nodes: [] })).toBe(false);
    expect(isGraphShape(null)).toBe(false);
  });
});
