import React, { useEffect, useMemo, useState } from "react";
import GlobeGraph from "./components/GlobeGraph.jsx";
import ControlPanel from "./components/ControlPanel.jsx";
import FigurePlate from "./components/FigurePlate.jsx";
import { addArxiv, askQuestion, uploadPdf } from "./utils/api.js";
import {
  EMPTY_GRAPH,
  buildGraphUrl,
  censusOf,
  hasEntities,
  isGraphShape,
  loadCachedGraph,
  saveCachedGraph,
  viewFor,
  withPaperHubs,
} from "./utils/graphData.js";

function endpointId(endpoint) {
  return endpoint && typeof endpoint === "object" ? endpoint.id : endpoint;
}

function App() {
  const [search, setSearch] = useState("");
  const [showEdges, setShowEdges] = useState(true);
  const [showLabels, setShowLabels] = useState(true);
  const [layer, setLayer] = useState("overview");
  const [groupFilter, setGroupFilter] = useState("all");
  const [expanded, setExpanded] = useState(() => new Set());
  const [highlightNode, setHighlightNode] = useState(null);
  const [activeNodeIds, setActiveNodeIds] = useState([]);
  const [graphData, setGraphData] = useState(EMPTY_GRAPH);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [uploading, setUploading] = useState(false);
  const [uploadStatus, setUploadStatus] = useState("");
  const [asking, setAsking] = useState(false);
  const [answer, setAnswer] = useState("");
  const [qaError, setQaError] = useState("");
  const [matches, setMatches] = useState([]);
  const [facts, setFacts] = useState([]);

  const loadGraph = async ({ preferCache = true } = {}) => {
    const cached = preferCache ? loadCachedGraph() : null;
    if (cached) {
      setGraphData(cached);
      setLoading(false);
    }
    try {
      const res = await fetch(buildGraphUrl(), { cache: "no-cache" });
      if (!res.ok) throw new Error(`Failed to load data (${res.status})`);
      const json = await res.json();
      const next = isGraphShape(json) ? json : EMPTY_GRAPH;
      setGraphData(next);
      saveCachedGraph(next);
      setError("");
    } catch (err) {
      console.error(err);
      if (!cached) setGraphData(EMPTY_GRAPH);
      setError(
        cached
          ? "Showing a cached copy — could not reach the latest graph."
          : "No graph data yet. Run an ingest to build one."
      );
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadGraph();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Papers are derived once, not per layer — the hierarchy is a property of the
  // corpus, not of what happens to be on screen.
  const hubGraph = useMemo(() => withPaperHubs(graphData), [graphData]);
  const census = useMemo(() => censusOf(hubGraph), [hubGraph]);

  // No mapping over nodes here: colours are resolved once in `withPaperHubs`,
  // because the simulation keeps each node's position on the object itself and
  // a per-render copy would reset the layout (see the note in that function).
  const coloredData = useMemo(
    () => viewFor(hubGraph, layer, expanded),
    [hubGraph, layer, expanded]
  );

  const groups = useMemo(
    () => [
      "all",
      ...new Set(
        coloredData.nodes
          .filter((n) => (layer === "passages" ? n.kind === "paper" : n.kind === "entity"))
          .map((n) => (layer === "passages" ? n.title || n.group : n.type || "Entity"))
      ),
    ],
    [coloredData, layer]
  );

  const graphHasEntities = useMemo(() => hasEntities(graphData.nodes), [graphData]);

  const filteredData = useMemo(() => {
    const activeSet = new Set(activeNodeIds);
    const needle = search.toLowerCase();
    const nodes = coloredData.nodes.filter((n) => {
      const matchesSearch = needle
        ? n.id.toLowerCase().includes(needle) ||
          (n.name || "").toLowerCase().includes(needle) ||
          (n.label || "").toLowerCase().includes(needle)
        : true;
      // The filter names a paper in the passage layer and an entity type
      // elsewhere; papers always stay, so filtering never empties the frame.
      const facet = layer === "passages" ? n.title || n.group : n.type;
      const matchesGroup =
        groupFilter === "all" || n.kind === "paper" || facet === groupFilter;
      return (matchesSearch || activeSet.has(n.id)) && matchesGroup;
    });
    const ids = new Set(nodes.map((n) => n.id));
    return {
      nodes,
      links: coloredData.links.filter(
        (l) => ids.has(endpointId(l.source)) && ids.has(endpointId(l.target))
      ),
    };
  }, [search, groupFilter, coloredData, activeNodeIds]);

  const changeLayer = (next) => {
    setLayer(next);
    setGroupFilter("all");
  };

  const corpus = useMemo(
    () => ({ papers: census.papers, chunks: census.passages, entities: census.entities }),
    [census]
  );

  const addPaper = async (fetchPaper) => {
    try {
      setUploading(true);
      setUploadStatus("Reading, chunking and embedding — this takes a minute.");
      setError("");
      const res = await fetchPaper();
      setUploadStatus(
        `Added ${res.filename}: ${res.newChunks} passages, ${res.newEpisodes ?? 0} graph episodes.`
      );
      await loadGraph({ preferCache: false });
      return true;
    } catch (err) {
      setUploadStatus("");
      setError(err.message || "Could not add that paper.");
      return false;
    } finally {
      setUploading(false);
    }
  };

  const handleUpload = (file) => (file ? addPaper(() => uploadPdf(file)) : false);
  const handleArxiv = (url) => addPaper(() => addArxiv(url));

  const handleAsk = async (query) => {
    if (!query) return;
    try {
      setAsking(true);
      setQaError("");
      const res = await askQuestion(query);
      const used = Array.isArray(res.usedNodeIds) ? res.usedNodeIds : [];
      setAnswer(res.answer || "");
      setMatches(Array.isArray(res.matches) ? res.matches : []);
      setFacts(Array.isArray(res.facts) ? res.facts : []);
      setActiveNodeIds(used);
      // Open the papers the answer came from, so the cited passages are on
      // screen instead of hidden inside a collapsed hub.
      const cited = new Set(used);
      const papersCited = hubGraph.nodes
        .filter((n) => n.kind === "chunk" && cited.has(n.id))
        .map((n) => n.paper);
      if (papersCited.length) {
        setExpanded((prev) => new Set([...prev, ...papersCited]));
      }
      const visible = new Set(coloredData.nodes.map((n) => n.id));
      const focus = used.find((id) => visible.has(id)) || papersCited[0];
      if (focus) setHighlightNode(focus);
    } catch (err) {
      setQaError(err.message || "That question could not be answered.");
      setAnswer("");
      setMatches([]);
      setFacts([]);
      setActiveNodeIds([]);
    } finally {
      setAsking(false);
    }
  };

  const focusNode = (nodeId) => {
    if (!nodeId) return;
    setHighlightNode(nodeId);
    setSearch("");
  };

  // Clicking a paper opens or closes it. That is the whole navigation model:
  // the overview stays readable, and detail is one click away where you asked
  // for it rather than everywhere at once.
  const clickNode = (nodeId) => {
    if (!nodeId) return;
    const node = hubGraph.nodes.find((n) => n.id === nodeId);
    if (node?.kind === "paper") {
      setExpanded((prev) => {
        const next = new Set(prev);
        next.has(nodeId) ? next.delete(nodeId) : next.add(nodeId);
        return next;
      });
    }
    setHighlightNode(nodeId);
  };

  return (
    <div className="app">
      <header className="masthead">
        <div className="wordmark">
          <span className="mark">Research</span> Knowledge Graph
        </div>
        <dl className="stats">
          <div>
            <dt>Papers</dt>
            <dd>{corpus.papers || "—"}</dd>
          </div>
          <div>
            <dt>Passages</dt>
            <dd>{corpus.chunks || "—"}</dd>
          </div>
          <div>
            <dt>Entities</dt>
            <dd>{corpus.entities || "—"}</dd>
          </div>
        </dl>
      </header>

      {error && <div className="banner">{error}</div>}

      <main className="layout">
        <ControlPanel
          onAsk={handleAsk}
          asking={asking}
          answer={answer}
          qaError={qaError}
          matches={matches}
          facts={facts}
          onFocusNode={focusNode}
          onUploadFile={handleUpload}
          onAddArxiv={handleArxiv}
          uploading={uploading}
          uploadStatus={uploadStatus}
        />

        <FigurePlate
          layer={layer}
          setLayer={changeLayer}
          groupFilter={groupFilter}
          setGroupFilter={setGroupFilter}
          groups={groups}
          showEdges={showEdges}
          setShowEdges={setShowEdges}
          showLabels={showLabels}
          setShowLabels={setShowLabels}
          hasEntities={graphHasEntities}
          census={census}
          shown={{ nodes: filteredData.nodes.length, links: filteredData.links.length }}
          openCount={expanded.size}
          onCollapseAll={() => setExpanded(new Set())}
          onExpandAll={() =>
            setExpanded(
              new Set(hubGraph.nodes.filter((n) => n.kind === "paper").map((n) => n.id))
            )
          }
          loading={loading}
        >
          <GlobeGraph
            data={filteredData}
            highlightNode={highlightNode}
            activeNodeIds={activeNodeIds}
            showEdges={showEdges}
            showLabels={showLabels}
            onNodeHover={setHighlightNode}
            onNodeClick={clickNode}
          />
        </FigurePlate>
      </main>
    </div>
  );
}

export default App;
