import React, { useEffect, useMemo, useRef } from "react";
import ForceGraph3D from "react-force-graph-3d";
import { forceCollide, forceManyBody } from "d3-force-3d";
import * as THREE from "three";

/**
 * The corpus drawn as a figure from a paper, not a starfield: flat discs,
 * hairline ink edges, dark type on white. No bloom, no glow — legibility is
 * the point, and a research tool should read like a diagram you could print.
 */

const PAPER = "#FFFFFF";
const INK = "#14201B";
const EDGE = "rgba(20, 32, 27, 0.16)";
const EDGE_FAINT = "rgba(20, 32, 27, 0.06)";
const EDGE_STRONG = "rgba(20, 32, 27, 0.42)";
const CITED = "#C2325F";

/**
 * Nothing GPU-side may be shared between node objects.
 *
 * three-forcegraph deallocates a node's object whenever it rebuilds one — and
 * `deallocate` disposes `material.map`, not just the material. Every re-render
 * (hover, click, layer change) rebuilds every node, so a module-level texture
 * or material cache is destroyed by the first interaction: the original code
 * cached both, and the first hover blanked the entire figure permanently.
 * Everything below is therefore built per object and left for the library to
 * dispose.
 */

const TEXTURE_PX = 256;

/**
 * Canvas textures default to a linear colour space, while the renderer outputs
 * sRGB — so every fill drawn here arrives on screen visibly lighter and flatter
 * than the colour asked for. Tagging the texture is what keeps the palette the
 * palette.
 */
function srgb(texture) {
  texture.colorSpace = THREE.SRGBColorSpace;
  return texture;
}

/** Break a name into at most `maxLines` lines that fit `maxWidth` pixels. */
function wrap(ctx, text, maxWidth, maxLines) {
  const lines = [];
  let line = "";
  for (const word of String(text).split(/\s+/)) {
    const candidate = line ? `${line} ${word}` : word;
    if (ctx.measureText(candidate).width <= maxWidth || !line) {
      line = candidate;
    } else {
      lines.push(line);
      line = word;
      if (lines.length === maxLines) break;
    }
  }
  if (lines.length < maxLines && line) lines.push(line);

  // Anything that still overflows gets an ellipsis rather than spilling out of
  // the circle it is drawn inside.
  const last = lines.length - 1;
  if (lines[last] && ctx.measureText(lines[last]).width > maxWidth) {
    let clipped = lines[last];
    while (clipped.length > 1 && ctx.measureText(`${clipped}…`).width > maxWidth) {
      clipped = clipped.slice(0, -1);
    }
    lines[last] = `${clipped}…`;
  }
  return lines;
}

/**
 * A filled circle with its name written inside it.
 *
 * Labels used to hang *below* a small disc, which meant the type collided with
 * neighbouring nodes and the layout had to reserve room for a label many times
 * wider than the thing it named. Drawing the name inside the circle makes the
 * node its own label: spacing is just the radius, and the figure reads as
 * named objects rather than dots with captions.
 */
function makeNode({ label, fill, textColor, ring, dim }) {
  const canvas = document.createElement("canvas");
  canvas.width = canvas.height = TEXTURE_PX;
  const ctx = canvas.getContext("2d");
  const centre = TEXTURE_PX / 2;

  ctx.beginPath();
  ctx.arc(centre, centre, centre - 8, 0, Math.PI * 2);
  ctx.fillStyle = fill;
  ctx.globalAlpha = dim ? 0.35 : 1;
  ctx.fill();
  ctx.lineWidth = 6;
  ctx.strokeStyle = ring;
  ctx.stroke();
  ctx.globalAlpha = 1;

  if (label) {
    const size = label.length > 28 ? 26 : label.length > 14 ? 30 : 34;
    ctx.font = `600 ${size}px 'IBM Plex Sans', system-ui, sans-serif`;
    ctx.fillStyle = textColor;
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    const lines = wrap(ctx, label, TEXTURE_PX * 0.74, 3);
    const step = size * 1.12;
    const top = centre - ((lines.length - 1) * step) / 2;
    lines.forEach((line, i) => ctx.fillText(line, centre, top + i * step));
  }

  const sprite = new THREE.Sprite(
    new THREE.SpriteMaterial({
      map: srgb(new THREE.CanvasTexture(canvas)),
      transparent: true,
      depthWrite: false,
    })
  );
  return sprite;
}

/** Relation names, drawn on the edge itself as in a hand-made diagram. */
const EDGE_LABEL_PX = 26;

function makeEdgeLabel(text) {
  const font = `500 ${EDGE_LABEL_PX}px 'IBM Plex Sans', system-ui, sans-serif`;
  const measure = document.createElement("canvas").getContext("2d");
  measure.font = font;
  const width = Math.ceil(measure.measureText(text).width) + 20;
  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = 40;
  const ctx = canvas.getContext("2d");
  ctx.font = font;
  ctx.textBaseline = "middle";
  // Paper-coloured halo so the name stays readable where it crosses its edge.
  ctx.lineWidth = 7;
  ctx.strokeStyle = PAPER;
  ctx.strokeText(text, 10, 20);
  ctx.fillStyle = "#6B7A73";
  ctx.fillText(text, 10, 20);

  const sprite = new THREE.Sprite(
    new THREE.SpriteMaterial({
      map: srgb(new THREE.CanvasTexture(canvas)),
      transparent: true,
      depthWrite: false,
    })
  );
  const h = 7;
  sprite.scale.set((width / 40) * h, h, 1);
  return sprite;
}

function endpointId(endpoint) {
  return endpoint && typeof endpoint === "object" ? endpoint.id : endpoint;
}

// Circle diameter per tier, in world units. Papers dominate, entities sit a
// step below, passages are texture — the hierarchy should be legible before you
// read a word. Sized to hold the name written inside.
function sizeOf(node, focused) {
  const grow = focused ? 1.12 : 1;
  if (node.kind === "paper") return (44 + 22 * (node.weight || 0)) * grow;
  // A circle is only as big as the name it has to hold. An unnamed one is a
  // dot: giving it the same footprint would spend the figure's whole area on
  // nodes that say nothing.
  if (node.kind === "entity") return (node.labelled ? 26 + 10 * (node.weight || 0) : 11) * grow;
  return (node.labelled ? 20 : 9) * grow;
}

/** An entity earns a standing label by appearing in more than one paper. */
const bridges = (node) => node.kind === "entity" && (node.papers || 0) > 1;

// Spacing is now just the circle plus a gap: with the name drawn inside, a node
// no longer has to reserve room for a caption several times its own width.
// Passages reserve the room a *labelled* one needs, since whether they carry a
// section name depends on how many are open — and the forces are configured
// once, before that is known.
function collisionRadius(node) {
  if (node.kind === "chunk") return 12;
  if (node.kind === "entity") return 16;
  return sizeOf(node, false) / 2 + 6;
}

const LINK_DISTANCE = { contains: 60, sequence: 34, bridges: 190, relation: 96 };
const LINK_STRENGTH = { contains: 0.85, sequence: 0.2, bridges: 0.25, relation: 0.5 };

/** How many papers an entity spans, read off whichever end of the edge it is. */
function spanOf(link) {
  for (const end of [link.source, link.target]) {
    if (end && typeof end === "object" && end.kind === "entity") return end.papers || 1;
  }
  return 1;
}

// An entity found in one paper belongs to it, and is drawn tight against it as
// a fan; one found in several belongs to none of them, so its edges are long
// and slack and it settles in the space between. The layout then says which is
// which before you read a single label.
function linkDistance(link) {
  if (link.kind === "bridges") return spanOf(link) > 1 ? 190 : 74;
  return LINK_DISTANCE[link.kind] ?? 60;
}

function linkStrength(link) {
  if (link.kind === "bridges") return spanOf(link) > 1 ? 0.2 : 0.75;
  return LINK_STRENGTH[link.kind] ?? 0.4;
}

/**
 * ForceGraph3D sizes itself to the window unless told otherwise. Inside a
 * framed plate that means it renders off-centre and overflows, so measure the
 * container and feed it real dimensions.
 */
function useMeasuredSize(ref) {
  const [size, setSize] = React.useState({ width: 0, height: 0 });
  // Measure synchronously in a layout effect, then keep watching. Relying on
  // the observer's first callback alone is a race: if it reports 0 before the
  // plate has been laid out and the element never resizes again, no second
  // callback ever arrives, the graph is never mounted, and the plate stays
  // blank — with the container sitting there at full size.
  React.useLayoutEffect(() => {
    const el = ref.current;
    if (!el) return undefined;
    const measure = () => {
      const { width, height } = el.getBoundingClientRect();
      const next = { width: Math.floor(width), height: Math.floor(height) };
      setSize((prev) =>
        prev.width === next.width && prev.height === next.height ? prev : next
      );
    };
    measure();
    const observer = new ResizeObserver(measure);
    observer.observe(el);
    return () => observer.disconnect();
  }, [ref]);
  return size;
}

function useNeighbors(data) {
  return useMemo(() => {
    const map = new Map();
    data.links.forEach((link) => {
      const source = endpointId(link.source);
      const target = endpointId(link.target);
      if (!source || !target) return;
      if (!map.has(source)) map.set(source, new Set());
      if (!map.has(target)) map.set(target, new Set());
      map.get(source).add(target);
      map.get(target).add(source);
    });
    return map;
  }, [data]);
}

function GlobeGraph({
  data,
  highlightNode,
  activeNodeIds = [],
  showEdges,
  showLabels = true,
  onNodeHover,
  onNodeClick,
}) {
  const fgRef = useRef();
  const holderRef = useRef(null);
  const { width, height } = useMeasuredSize(holderRef);
  const neighborMap = useNeighbors(data);
  const activeSet = useMemo(() => new Set(activeNodeIds || []), [activeNodeIds]);

  // With few enough of a kind on screen there is room to name them all; past
  // that, only the ones spanning papers keep a standing label.
  const [labelEveryEntity, labelPassages] = useMemo(() => {
    let entities = 0;
    let chunks = 0;
    for (const node of data.nodes) {
      if (node.kind === "entity") entities += 1;
      else if (node.kind === "chunk") chunks += 1;
    }
    // Named nodes are the point of the figure, so name them until the count
    // itself becomes the problem. Past that the graph is one to zoom into
    // rather than read whole.
    return [entities <= 250, chunks > 0 && chunks <= 60];
  }, [data]);

  const labelEveryEdge = useMemo(
    () =>
      data.links.filter((l) => l.kind === "relation" || l.kind === "bridges").length <= 40,
    [data]
  );

  // Papers are the frame of the figure, so they never dim. In the overview they
  // share no edge with anything, and dimming by adjacency would erase every
  // other paper the moment you hovered one.
  const dimmed = (node) => {
    if (!highlightNode || node.kind === "paper") return false;
    if (node.id === highlightNode) return false;
    return !(neighborMap.get(highlightNode) || new Set()).has(node.id);
  };

  /**
   * Which nodes carry their name.
   *
   * Papers always. Entities when they carry information — one appearing in
   * several papers is the finding, while a hundred single-paper names stacked
   * on each other is the noise this view exists to remove. Passages only inside
   * a paper you opened, and only while few enough are on screen to read.
   */
  const labels = (node, focused, cited) => {
    if (!showLabels) return false;
    if (node.kind === "paper") return true;
    if (node.kind === "entity") return labelEveryEntity || bridges(node) || focused || cited;
    return labelPassages || focused || cited;
  };

  const nodeThreeObject = (node) => {
    const cited = activeSet.has(node.id);
    const focused = node.id === highlightNode;
    const faded = dimmed(node);
    const named = labels(node, focused, cited);

    // Nodes are their own labels now, so the sprite has to be built knowing
    // whether it will hold text — the size depends on it.
    node.labelled = named;

    const fill = cited ? CITED : node.color || "#7C8B85";
    const sprite = makeNode({
      label: named ? node.name || node.section || "" : "",
      fill,
      // White reads on every colour in the palette; the passage grey is the one
      // light enough to need dark type.
      textColor: node.kind === "chunk" && !cited ? INK : PAPER,
      ring: cited ? CITED : "rgba(20, 32, 27, 0.28)",
      dim: faded,
    });

    const size = sizeOf(node, focused);
    sprite.scale.set(size, size, 1);
    return sprite;
  };

  const linkKind = (link) => link.kind || "sequence";

  // An entity that reaches three papers is the finding; the passage chain
  // inside one paper is scaffolding. Weight the ink accordingly.
  const linkColor = (link) => {
    const src = endpointId(link.source);
    const tgt = endpointId(link.target);
    if (activeSet.has(src) || activeSet.has(tgt)) return "rgba(194, 50, 95, 0.55)";
    if (highlightNode) {
      const connected = src === highlightNode || tgt === highlightNode;
      return connected ? EDGE_STRONG : EDGE_FAINT;
    }
    const kind = linkKind(link);
    if (kind === "relation" || kind === "bridges") return EDGE_STRONG;
    return EDGE;
  };

  const linkWidth = (link) => {
    const src = endpointId(link.source);
    const tgt = endpointId(link.target);
    if (activeSet.has(src) || activeSet.has(tgt)) return 1.6;
    const kind = linkKind(link);
    if (kind === "bridges") return 0.6 + Math.min(1.4, (link.passages || 1) * 0.25);
    if (kind === "relation") return 0.8;
    return 0.35;
  };

  // What an edge is called. Named relations between entities carry the real
  // claim ("Outperforms", "EvaluatedOn"), and provenance says how many passages
  // back it; the structural edges — a paper containing a passage, one passage
  // following another — say nothing a reader cannot already see.
  const edgeName = (link) => {
    const kind = linkKind(link);
    if (kind === "relation") return link.label || "related";
    if (kind === "bridges") {
      return link.passages > 1 ? `mentions ×${link.passages}` : "mentions";
    }
    return null;
  };

  /**
   * Edge labels, budgeted.
   *
   * Naming every edge at once is what a printed figure can do and a live canvas
   * cannot: past a few dozen the type covers the graph it annotates. So all
   * named edges are labelled while they are few, and beyond that only the ones
   * touching whatever you are looking at.
   */
  const linkThreeObject = (link) => {
    const name = edgeName(link);
    if (!name || !showLabels) return null;
    const src = endpointId(link.source);
    const tgt = endpointId(link.target);
    const relevant =
      src === highlightNode ||
      tgt === highlightNode ||
      activeSet.has(src) ||
      activeSet.has(tgt);
    return labelEveryEdge || relevant ? makeEdgeLabel(name) : null;
  };

  const linkPositionUpdate = (sprite, { start, end }) => {
    if (!sprite) return;
    sprite.position.set(
      start.x + (end.x - start.x) / 2,
      start.y + (end.y - start.y) / 2,
      start.z + (end.z - start.z) / 2
    );
  };

  // Keyed on `width`, not mount: the graph is only rendered once the container
  // has been measured, so on the first commit there is no instance to configure
  // and a mount-only effect would silently do nothing.
  useEffect(() => {
    if (!fgRef.current || !width) return;
    fgRef.current.scene().fog = null;

    // Cap how far repulsion reaches. The overview is mostly *unlinked* hubs —
    // papers share no edge until an entity bridges them — and unbounded
    // long-range charge pushes them apart until they leave the frustum
    // entirely, which is why an unlinked overview rendered as a blank plate.
    // A finite distanceMax turns the same force into local spacing.
    // Register our own charge rather than tuning the built-in one: the library
    // rebuilds its forces when the layout re-initialises, which silently threw
    // away every setting applied to the instance it happened to hold at mount.
    // A force we install by name survives, and re-running this effect on each
    // data change re-installs it if it ever does not.
    //
    // Weak and short-range: collision already guarantees nodes do not overlap,
    // so charge only has to stop a cluster collapsing onto a point. A strong
    // one fights the links and pushes an entity fan so far from the paper it
    // belongs to that the membership stops reading.
    fgRef.current.d3Force("charge", forceManyBody().strength(-38).distanceMax(260));

    // Space nodes by the room their *label* needs, not by their disc. A paper
    // disc is ~20 units across and its title is over a hundred, so collision
    // sized to the geometry still stacks the type into an unreadable pile.
    fgRef.current.d3Force("collide", forceCollide(collisionRadius));

    // Edge length carries meaning here. A passage belongs to exactly one paper,
    // so `contains` is short and stiff and the passages gather around their own
    // hub instead of pooling in the middle of the ring; `bridges` is long and
    // slack, letting a shared entity float between the papers that cite it.
    const link = fgRef.current.d3Force("link");
    if (link) link.distance(linkDistance).strength(linkStrength);
    // No d3ReheatSimulation() here: `numDimensions` rebuilds the simulation, and
    // reheating mid-rebuild throws inside the layout tick and blanks the canvas.
    // Forces registered during the cooldown window take effect on the next tick
    // on their own.
  }, [width, data]);

  // Frame the graph from its geometry, then refine once the layout settles.
  //
  // The papers are pinned to a ring of known radius, so the opening camera can
  // be derived from the positions themselves rather than guessed from a node
  // count — the previous heuristic assumed a layout that collapses toward the
  // origin and left the whole ring outside the frustum, i.e. a blank plate.
  // zoomToFit still runs on `onEngineStop`, the one moment positions are final,
  // to tighten around whatever passages have been opened.
  // Half-extent of the pinned layout, plus the room its labels need.
  const extent = useMemo(() => {
    let x = 0;
    let y = 0;
    for (const node of data.nodes) {
      x = Math.max(x, Math.abs(node.fx ?? node.x ?? 0) + collisionRadius(node));
      y = Math.max(y, Math.abs(node.fy ?? node.y ?? 0) + 20);
    }
    return { x: x || 150, y: y || 150 };
  }, [data]);

  useEffect(() => {
    if (!fgRef.current || !width || !height || !data.nodes.length) return;
    // Solve the camera distance from the frustum rather than fitting after the
    // fact: the layout is pinned, so the box to frame is known before a single
    // tick runs. `zoomToFit` still refines once passages open, but the figure no
    // longer depends on when — or whether — the engine reports it has settled.
    const fov = (fgRef.current.camera().fov * Math.PI) / 180;
    const halfFov = Math.tan(fov / 2);
    const aspect = width / height;
    const distance = 1.08 * Math.max(extent.y / halfFov, extent.x / (halfFov * aspect));
    fgRef.current.cameraPosition({ x: 0, y: 0, z: distance }, { x: 0, y: 0, z: 0 }, 0);

    // Zoom and pan, but never rotate. The layout is planar, so any tumble turns
    // the figure edge-on and the plate goes blank — a click with a pixel of
    // drag in it was enough to lose the whole graph. Pan and zoom keep every
    // useful degree of freedom for reading it.
    // Set on the default trackball controls rather than by switching to orbit:
    // orbit captures the pointer on mousedown, so the graph's own click
    // detection never sees the release and nodes stop responding to clicks.
    const controls = fgRef.current.controls();
    if (controls) {
      controls.noRotate = true; // trackball
      controls.enableRotate = false; // orbit, should the control type change
    }
  }, [extent, width, height]);

  const handleEngineStop = () => {
    if (!fgRef.current) return;
    fgRef.current.zoomToFit(600, 70);
  };

  const nodeLabel = (node) => {
    if (node.kind === "paper") {
      const counts = [
        `${node.chunkCount} passages`,
        node.sectionCount ? `${node.sectionCount} sections` : "",
        node.entityCount ? `${node.entityCount} entities` : "",
      ].filter(Boolean);
      return `${node.title} — ${counts.join(", ")} · click to open`;
    }
    if (node.kind === "entity") {
      const span = node.papers > 1 ? ` · in ${node.papers} papers` : "";
      return (node.summary ? `${node.type} · ${node.summary}` : node.type) + span;
    }
    return [node.paper_title, node.section].filter(Boolean).join(" · ");
  };

  return (
    <div ref={holderRef} className="graph-holder">
      {width > 0 && (
    <ForceGraph3D
      ref={fgRef}
      width={width}
      height={height}
      graphData={data}
      backgroundColor={PAPER}
      nodeThreeObject={nodeThreeObject}
      nodeLabel={nodeLabel}
      linkColor={linkColor}
      linkWidth={linkWidth}
      linkOpacity={1}
      linkVisibility={() => showEdges}
      linkThreeObject={linkThreeObject}
      linkThreeObjectExtend={true}
      linkPositionUpdate={linkPositionUpdate}
      enableNodeDrag={false}
      showNavInfo={false}
      // Relations have a direction and the diagram should say so: "Transformer
      // Outperforms LSTM" is not the same claim read backwards. Structural
      // edges get no head — nothing is being asserted along them.
      linkDirectionalArrowLength={(link) =>
        linkKind(link) === "relation" || linkKind(link) === "bridges" ? 7 : 0
      }
      linkDirectionalArrowRelPos={0.92}
      linkDirectionalArrowColor={() => "rgba(20, 32, 27, 0.45)"}
      // Lay the graph out in a plane. Depth buys nothing here — it scales
      // labels by distance, hides hubs behind each other, and makes the same
      // corpus look different every load — while a flat arrangement of named
      // discs is the figure this is trying to be. Still the 3D renderer, still
      // orbitable; only the simulation is planar.
      numDimensions={2}
      // Default cooldown is 15s, so the fit below fired long before the layout
      // had settled. This graph is small; 5s is well past equilibrium.
      cooldownTime={5000}
      onEngineStop={handleEngineStop}
      onNodeHover={(node) => onNodeHover(node ? node.id : null)}
      onNodeClick={(node) => onNodeClick(node ? node.id : null)}
    />
      )}
    </div>
  );
}

export default GlobeGraph;
