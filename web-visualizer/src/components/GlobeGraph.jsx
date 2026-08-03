import React, { useEffect, useMemo, useRef } from "react";
import ForceGraph3D from "react-force-graph-3d";
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

// One white disc texture, tinted per node. Drawing a canvas per node per frame
// (as this did originally) leaks a GPU texture for every render pass.
let discTexture = null;
function getDiscTexture() {
  if (discTexture) return discTexture;
  const size = 128;
  const canvas = document.createElement("canvas");
  canvas.width = canvas.height = size;
  const ctx = canvas.getContext("2d");
  ctx.beginPath();
  ctx.arc(size / 2, size / 2, size / 2 - 6, 0, Math.PI * 2);
  ctx.fillStyle = "#ffffff";
  ctx.fill();
  discTexture = new THREE.CanvasTexture(canvas);
  return discTexture;
}

const discMaterials = new Map();
function getDiscMaterial(color) {
  let material = discMaterials.get(color);
  if (!material) {
    material = new THREE.SpriteMaterial({
      map: getDiscTexture(),
      color,
      transparent: true,
      depthWrite: false,
    });
    discMaterials.set(color, material);
  }
  return material;
}

const textTextures = new Map();
function getTextTexture(text, color) {
  const key = `${color}|${text}`;
  let texture = textTextures.get(key);
  if (texture) return texture;

  const font = "500 30px 'IBM Plex Sans', system-ui, sans-serif";
  const measure = document.createElement("canvas").getContext("2d");
  measure.font = font;
  const width = Math.ceil(measure.measureText(text).width) + 24;
  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = 48;
  const ctx = canvas.getContext("2d");
  ctx.font = font;
  ctx.textBaseline = "middle";
  // Paper-coloured halo so type stays readable where it crosses an edge.
  ctx.lineWidth = 6;
  ctx.strokeStyle = PAPER;
  ctx.strokeText(text, 12, 24);
  ctx.fillStyle = color;
  ctx.fillText(text, 12, 24);

  texture = new THREE.CanvasTexture(canvas);
  textTextures.set(key, texture);
  return texture;
}

// 10 world-units tall reads clearly at the distance zoomToFit settles on for a
// graph of this size; the width follows the measured texture so glyphs are
// never stretched.
const LABEL_HEIGHT = 10;

function makeLabel(text, color, scale = 1) {
  const texture = getTextTexture(text, color);
  const sprite = new THREE.Sprite(
    new THREE.SpriteMaterial({ map: texture, transparent: true, depthWrite: false })
  );
  const h = LABEL_HEIGHT * scale;
  sprite.scale.set((texture.image.width / texture.image.height) * h, h, 1);
  return sprite;
}

function endpointId(endpoint) {
  return endpoint && typeof endpoint === "object" ? endpoint.id : endpoint;
}

/**
 * ForceGraph3D sizes itself to the window unless told otherwise. Inside a
 * framed plate that means it renders off-centre and overflows, so measure the
 * container and feed it real dimensions.
 */
function useMeasuredSize(ref) {
  const [size, setSize] = React.useState({ width: 0, height: 0 });
  useEffect(() => {
    const el = ref.current;
    if (!el) return undefined;
    const observer = new ResizeObserver(([entry]) => {
      const { width, height } = entry.contentRect;
      setSize({ width: Math.floor(width), height: Math.floor(height) });
    });
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

  const dimmed = (id) => {
    if (!highlightNode) return false;
    if (id === highlightNode) return false;
    return !(neighborMap.get(highlightNode) || new Set()).has(id);
  };

  const nodeThreeObject = (node) => {
    const isEntity = node.kind === "entity";
    const cited = activeSet.has(node.id);
    const focused = node.id === highlightNode;
    const faded = dimmed(node.id);

    const color = cited ? CITED : faded ? "#C8D2CC" : node.color || "#7C8B85";
    const size = isEntity ? (focused ? 13 : 10) : focused ? 7 : 4.5;

    const group = new THREE.Group();
    const disc = new THREE.Sprite(getDiscMaterial(color));
    disc.scale.set(size, size, 1);
    group.add(disc);

    // Entities are the meaningful layer, so they carry their name at all times
    // rather than hiding it behind a hover.
    if (isEntity && showLabels && !faded) {
      const label = makeLabel(node.name || node.id, cited ? CITED : INK, focused ? 1.2 : 1);
      label.position.set(0, -(size / 2 + 7), 0);
      group.add(label);
    }
    return group;
  };

  const linkKind = (link) => link.kind || "sequence";

  const linkColor = (link) => {
    const src = endpointId(link.source);
    const tgt = endpointId(link.target);
    if (activeSet.has(src) || activeSet.has(tgt)) return "rgba(194, 50, 95, 0.55)";
    if (highlightNode) {
      const connected = src === highlightNode || tgt === highlightNode;
      return connected ? EDGE_STRONG : EDGE_FAINT;
    }
    return linkKind(link) === "relation" ? EDGE : EDGE_FAINT;
  };

  const linkWidth = (link) => {
    const src = endpointId(link.source);
    const tgt = endpointId(link.target);
    if (activeSet.has(src) || activeSet.has(tgt)) return 1.6;
    return linkKind(link) === "relation" ? 0.8 : 0.4;
  };

  // Relation names appear only for the focused node. Labelling every edge at
  // once is unreadable, and unlabelled edges are what made the old graph a
  // hairball — this is the compromise that keeps both readable.
  const linkThreeObject = (link) => {
    if (linkKind(link) !== "relation" || !link.label) return null;
    const src = endpointId(link.source);
    const tgt = endpointId(link.target);
    const relevant =
      src === highlightNode ||
      tgt === highlightNode ||
      activeSet.has(src) ||
      activeSet.has(tgt);
    return relevant ? makeLabel(link.label, "#6B7A73", 0.72) : null;
  };

  const linkPositionUpdate = (sprite, { start, end }) => {
    if (!sprite) return;
    sprite.position.set(
      start.x + (end.x - start.x) / 2,
      start.y + (end.y - start.y) / 2,
      start.z + (end.z - start.z) / 2
    );
  };

  useEffect(() => {
    if (!fgRef.current) return;
    fgRef.current.scene().fog = null;
  }, []);

  // Place the camera by node count instead of zoomToFit.
  //
  // zoomToFit measures whatever positions exist at the instant it is called, so
  // on a force layout that is still collapsing out of the origin it frames a
  // near-zero box and parks the camera inside the graph — a blank white plate.
  // Delaying it only makes the race less frequent, not absent. A force-directed
  // layout settles to a radius that grows roughly with sqrt(n) around the
  // origin, so distance can be derived instead of measured: deterministic,
  // immune to timing, and identical on every load.
  const nodeCount = data.nodes.length;
  useEffect(() => {
    if (!fgRef.current || !width || !nodeCount) return;
    const distance = 180 + 26 * Math.sqrt(nodeCount);
    fgRef.current.cameraPosition({ x: 0, y: 0, z: distance }, { x: 0, y: 0, z: 0 }, 600);
  }, [nodeCount, width, height]);

  const nodeLabel = (node) => {
    if (node.kind === "entity") {
      return node.summary ? `${node.type} · ${node.summary}` : node.type;
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
      onNodeHover={(node) => onNodeHover(node ? node.id : null)}
      onNodeClick={(node) => onNodeClick(node ? node.id : null)}
    />
      )}
    </div>
  );
}

export default GlobeGraph;
