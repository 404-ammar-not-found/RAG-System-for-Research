import { describe, expect, it } from "vitest";
import {
  buildGraphUrl,
  colorForGroup,
  filterByLayer,
  isGraphShape,
} from "./graphData";

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

  it("splits the graph by layer", () => {
    const nodes = [
      { id: "e1", kind: "entity" },
      { id: "c1", kind: "chunk" },
      { id: "legacy" },
    ];
    expect(filterByLayer(nodes, "entities").map((n) => n.id)).toEqual(["e1"]);
    // Nodes predating the v2 schema count as chunks, not as nothing.
    expect(filterByLayer(nodes, "chunks").map((n) => n.id)).toEqual(["c1", "legacy"]);
    expect(filterByLayer(nodes, "both")).toHaveLength(3);
  });

  it("validates graph shape", () => {
    expect(isGraphShape({ nodes: [], links: [] })).toBe(true);
    expect(isGraphShape({ nodes: [] })).toBe(false);
    expect(isGraphShape(null)).toBe(false);
  });
});
