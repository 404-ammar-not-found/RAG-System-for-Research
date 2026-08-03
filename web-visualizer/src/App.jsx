import React, { useEffect, useMemo, useState } from "react";
import GlobeGraph from "./components/GlobeGraph.jsx";
import ControlPanel from "./components/ControlPanel.jsx";
import FigurePlate from "./components/FigurePlate.jsx";
import { askQuestion, uploadPdf } from "./utils/api.js";
import {
  EMPTY_GRAPH,
  buildGraphUrl,
  colorForGroup,
  filterByLayer,
  hasEntities,
  isGraphShape,
  loadCachedGraph,
  saveCachedGraph,
} from "./utils/graphData.js";

function endpointId(endpoint) {
  return endpoint && typeof endpoint === "object" ? endpoint.id : endpoint;
}

function App() {
  const [search, setSearch] = useState("");
  const [showEdges, setShowEdges] = useState(true);
  const [showLabels, setShowLabels] = useState(true);
  const [layer, setLayer] = useState("entities");
  const [groupFilter, setGroupFilter] = useState("all");
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

  const coloredData = useMemo(() => {
    const nodes = filterByLayer(graphData.nodes, layer).map((n) => {
      const group = n.group || "ungrouped";
      return { ...n, group, color: n.color || colorForGroup(group) };
    });
    const visible = new Set(nodes.map((n) => n.id));
    const links = graphData.links.filter(
      (l) => visible.has(endpointId(l.source)) && visible.has(endpointId(l.target))
    );
    return { nodes, links };
  }, [graphData, layer]);

  const groups = useMemo(
    () => ["all", ...new Set(coloredData.nodes.map((n) => n.group || "ungrouped"))],
    [coloredData]
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
      const matchesGroup = groupFilter === "all" || n.group === groupFilter;
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

  const corpus = useMemo(() => {
    const papers = new Set();
    let chunks = 0;
    let entities = 0;
    graphData.nodes.forEach((n) => {
      if (n.kind === "entity") entities += 1;
      else {
        chunks += 1;
        if (n.group) papers.add(n.group);
      }
    });
    return { papers: papers.size, chunks, entities };
  }, [graphData]);

  const handleUpload = async (file) => {
    if (!file) return;
    try {
      setUploading(true);
      setUploadStatus("Reading, chunking and embedding — this takes a minute.");
      setError("");
      const res = await uploadPdf(file);
      setUploadStatus(
        `Added ${res.filename}: ${res.newChunks} passages, ${res.newEpisodes ?? 0} graph episodes.`
      );
      await loadGraph({ preferCache: false });
    } catch (err) {
      setUploadStatus("");
      setError(err.message || "Could not add that paper.");
    } finally {
      setUploading(false);
    }
  };

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
      const visible = new Set(coloredData.nodes.map((n) => n.id));
      const focus = used.find((id) => visible.has(id));
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
          nodeCount={filteredData.nodes.length}
          linkCount={filteredData.links.length}
          loading={loading}
        >
          <GlobeGraph
            data={filteredData}
            highlightNode={highlightNode}
            activeNodeIds={activeNodeIds}
            showEdges={showEdges}
            showLabels={showLabels}
            onNodeHover={setHighlightNode}
            onNodeClick={setHighlightNode}
          />
        </FigurePlate>
      </main>
    </div>
  );
}

export default App;
