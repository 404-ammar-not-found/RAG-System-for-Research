from __future__ import annotations

"""Relationships between papers and concepts, extracted with Graphify.

Graphify reads each PDF once and proposes concepts and links. Its node ids are
per-file, so the same concept in two papers would never connect; merging by
normalised label here is what turns per-paper fragments into a corpus graph.

The graph only steers retrieval and prompts. Its links are LLM output (Graphify
tags nearly all of them EXTRACTED, even "conceptually_related_to"), so a claim
still has to be cited to a passage and pass verification.
"""

import json
import re
from itertools import combinations
from pathlib import Path
from typing import Any

import networkx as nx

from .deps import chat_provider, key_for, model_for
from .retrieval import Bm25Index, Passage, _tokenize
from .settings import PipelineSettings

_PARENS = re.compile(r"\([^)]*\)")
_NON_WORD = re.compile(r"[^\w]+")
# Words too common in this corpus to identify a concept on their own.
_GENERIC = frozenset(
    "a about an and are approach architecture data deep did do does for from how in is learning "
    "method methods model models network networks neural of on paper papers relate task the to "
    "training use used uses using what which with".split()
)
_VAGUE = {"conceptually_related_to", "semantically_similar_to"}
_MIN_TITLE_CHARS = 12
_MAX_SEEDS = 6
_BACKENDS = {"gemini": "gemini", "anthropic": "claude", "openai": "openai"}


def norm_label(label: str) -> str:
    text = _PARENS.sub(" ", label).strip() or label
    return " ".join(_NON_WORD.sub(" ", text.lower()).replace("_", " ").split())


def _squash(text: str) -> str:
    """Letters and digits only, so line breaks and hyphenation cannot hide a title."""
    return re.sub(r"[^a-z0-9]", "", text.lower())


def _paper_id(stem: str) -> str:
    return "paper_" + _NON_WORD.sub("_", stem.lower())


def title_citations(texts: dict[str, str], titles: dict[str, list[str]]) -> list[tuple[str, str]]:
    """(citing, cited) pairs where one paper's text contains another's title."""
    wanted = {
        stem: {t for t in map(_squash, names) if len(t) >= _MIN_TITLE_CHARS}
        for stem, names in titles.items()
    }
    pairs = []
    for citing, text in texts.items():
        body = _squash(text)
        for cited, names in wanted.items():
            if cited != citing and any(name in body for name in names):
                pairs.append((citing, cited))
    return sorted(pairs)


def merge_extractions(
    extractions: dict[str, dict[str, Any]],
    titles: dict[str, list[str]],
    citations: list[tuple[str, str]] = (),
) -> dict[str, list[dict[str, Any]]]:
    """One graph from per-paper Graphify output, with concepts merged across papers."""
    by_title = {norm_label(t): stem for stem, names in titles.items() for t in names}
    nodes: dict[str, dict[str, Any]] = {
        _paper_id(stem): {
            "id": _paper_id(stem),
            "label": names[0] if names else stem,
            "file_type": "paper",
            "source_file": f"{stem}.pdf",
            "stem": stem,
            "papers": [stem],
        }
        for stem, names in titles.items()
    }
    edges: list[dict[str, Any]] = []

    for stem, extraction in extractions.items():
        ids: dict[str, str] = {}
        for raw in extraction.get("nodes", []):
            label = norm_label(str(raw.get("label") or ""))
            if not label:
                continue
            is_stub = bool((raw.get("metadata") or {}).get("graphify_reference_stub"))
            if raw.get("file_type") == "paper" and not is_stub and stem in titles:
                node_id = _paper_id(stem)
            elif label in by_title:
                node_id = _paper_id(by_title[label])
            else:
                node_id = "concept_" + label.replace(" ", "_")
            ids[raw["id"]] = node_id
            merged = nodes.setdefault(
                node_id,
                {
                    "id": node_id,
                    "label": raw["label"],
                    "file_type": raw.get("file_type") or "concept",
                    "source_file": f"{stem}.pdf",
                    "papers": [],
                },
            )
            if "stem" not in merged and stem not in merged["papers"]:
                merged["papers"] = sorted({*merged["papers"], stem})
        for raw in extraction.get("edges", []):
            source, target = ids.get(raw.get("source")), ids.get(raw.get("target"))
            if source and target and source != target:
                edges.append(
                    {
                        "source": source,
                        "target": target,
                        "relation": raw.get("relation") or "references",
                        "confidence": raw.get("confidence") or "INFERRED",
                        "confidence_score": raw.get("confidence_score") or 0.5,
                        "source_file": f"{stem}.pdf",
                        "paper": stem,
                    }
                )

    for citing, cited in citations:
        if citing in titles and cited in titles:
            edges.append(
                {
                    "source": _paper_id(citing),
                    "target": _paper_id(cited),
                    "relation": "cites",
                    "confidence": "EXTRACTED",
                    "confidence_score": 1.0,
                    "source_file": f"{citing}.pdf",
                    "paper": citing,
                }
            )
    return {"nodes": list(nodes.values()), "edges": edges}


class RelationGraph:
    """Read-side of the graph: what does a question touch, and what links to it?"""

    def __init__(self, graph: nx.DiGraph) -> None:
        self.graph = graph
        self._terms = {
            node_id: frozenset(_tokenize(norm_label(str(data.get("label") or "")))) - _GENERIC
            for node_id, data in graph.nodes(data=True)
        }

    @classmethod
    def from_extraction(cls, extraction: dict[str, list[dict[str, Any]]]) -> "RelationGraph":
        graph = nx.DiGraph()
        for node in extraction["nodes"]:
            graph.add_node(node["id"], **{k: v for k, v in node.items() if k != "id"})
        for edge in extraction["edges"]:
            attrs = {k: v for k, v in edge.items() if k not in ("source", "target")}
            graph.add_edge(edge["source"], edge["target"], **attrs)
        return cls(graph)

    @classmethod
    def load(cls, path: Path) -> "RelationGraph | None":
        if not path.is_file():
            return None
        data = json.loads(path.read_text())
        return cls(nx.node_link_graph(data, directed=True, edges="links"))

    def _seeds(self, question: str) -> list[str]:
        asked = set(_tokenize(question)) - _GENERIC
        scored = []
        for node_id, terms in self._terms.items():
            if terms:
                overlap = len(terms & asked) / len(terms)
                if overlap >= 0.5:
                    scored.append((-overlap, -self.graph.degree(node_id), node_id))
        return [node_id for *_, node_id in sorted(scored)[:_MAX_SEEDS]]

    def _neighbours(self, seeds: list[str]) -> set[str]:
        return {n for s in seeds for n in (*self.graph.successors(s), *self.graph.predecessors(s))}

    def related_papers(self, question: str) -> set[str]:
        """Papers holding a concept the question names, or one link away from it."""
        seeds = self._seeds(question)
        return {
            stem
            for node_id in {*seeds, *self._neighbours(seeds)}
            for stem in self.graph.nodes[node_id].get("papers", [])
        }

    def expansion_terms(self, question: str, limit: int = 8) -> list[str]:
        seeds = self._seeds(question)
        linked = sorted(self._neighbours(seeds) - set(seeds), key=self.graph.degree, reverse=True)
        return [norm_label(str(self.graph.nodes[n].get("label"))) for n in linked[:limit]]

    def facts(self, question: str, limit: int = 12) -> list[dict[str, str]]:
        seeds = set(self._seeds(question))
        touching = [
            (u, v, d) for u, v, d in self.graph.edges(data=True) if u in seeds or v in seeds
        ]
        touching.sort(
            key=lambda e: (e[2].get("relation") in _VAGUE, -float(e[2].get("confidence_score") or 0))
        )
        return [
            {
                "source": str(self.graph.nodes[u].get("label")),
                "relation": str(d.get("relation")),
                "target": str(self.graph.nodes[v].get("label")),
                "paper": str(d.get("paper")),
                "confidence": str(d.get("confidence")),
            }
            for u, v, d in touching[:limit]
        ]

    def paper_links(self) -> list[tuple[str, str, str]]:
        """(paper, relation, paper) for links directly between library papers."""
        stem = self.graph.nodes
        return sorted(
            (stem[u]["stem"], str(d.get("relation")), stem[v]["stem"])
            for u, v, d in self.graph.edges(data=True)
            if "stem" in stem[u] and "stem" in stem[v]
        )

    def shared_concepts(self) -> list[dict[str, Any]]:
        """Pairs of papers connected by a concept that both of them discuss."""
        pairs: dict[tuple[str, str], list[str]] = {}
        for _, data in self.graph.nodes(data=True):
            if "stem" not in data:
                for a, b in combinations(sorted(data.get("papers", [])), 2):
                    pairs.setdefault((a, b), []).append(str(data.get("label")))
        return [
            {"a": a, "b": b, "concepts": sorted(concepts)} for (a, b), concepts in sorted(pairs.items())
        ]


def graph_search(bm25: Bm25Index, relations: RelationGraph, query: str, k: int) -> list[Passage]:
    """Lexical search widened by linked concepts, limited to the papers they touch."""
    papers = relations.related_papers(query)
    if not papers:
        return []
    widened = " ".join([query, *relations.expansion_terms(query)])
    hits = bm25.search(widened, k * 4)
    return [p for p in hits if Path(p.source).stem in papers][:k]


def format_relations(facts: list[dict[str, str]]) -> str:
    """Prompt block for the links a question touches; empty when there are none."""
    if not facts:
        return ""
    lines = "\n".join(f"- {f['source']} -[{f['relation']}]-> {f['target']} (from {f['paper']})" for f in facts)
    return (
        "RELATIONSHIPS — an LLM-extracted map of how these papers and concepts connect. Use it "
        "to notice connections, then confirm each one in PASSAGES. Never cite a relationship "
        "itself; if no passage supports a connection, leave it out.\n"
        f"{lines}\n\n"
    )


def with_relations(base: dict[str, Any], relations: RelationGraph | None) -> dict[str, Any]:
    """Literature-map data plus graph citations and concepts shared between papers."""
    if relations is None:
        return {**base, "shared": []}
    cites = {(a, b) for a, relation, b in relations.paper_links() if relation == "cites"}
    edges = cites | {tuple(edge) for edge in base["edges"]}
    return {**base, "edges": [list(e) for e in sorted(edges)], "shared": relations.shared_concepts()}


def graph_path(settings: PipelineSettings) -> Path:
    return Path(settings.relations_dir) / "graph.json"


def extract_paper(pdf: Path, settings: PipelineSettings) -> dict[str, Any]:
    """Graphify's semantic pass over one PDF, kept on disk because it is slow and paid."""
    out_dir = Path(settings.relations_dir)
    cached = out_dir / f"{pdf.stem}.json"
    if cached.is_file():
        return json.loads(cached.read_text())

    # Imported here: Graphify reads OPENAI_BASE_URL at import, after .env has loaded.
    from graphify.llm import extract_corpus_parallel

    provider = chat_provider()
    result = extract_corpus_parallel(
        [pdf.resolve()],
        backend=_BACKENDS[provider],
        api_key=key_for(provider),
        model=model_for(settings, "graph"),
        root=pdf.resolve().parent,
        max_concurrency=2,
        max_retry_depth=1,
        cache_root=out_dir.resolve() / "graphify",
    )
    # A partial result must not be cached, or the rest of the paper is never retried.
    if not result.get("failed_chunks"):
        out_dir.mkdir(parents=True, exist_ok=True)
        cached.write_text(json.dumps(result))
    return result


def _paper_titles(pdfs: list[Path]) -> dict[str, list[str]]:
    from .parsing import pdf_title
    from .scholarly import corpus_graph

    records = {p["file"]: p.get("title") for p in corpus_graph(pdfs)["papers"]}
    titles: dict[str, list[str]] = {}
    for pdf in pdfs:
        names = [records.get(pdf.stem), pdf_title(pdf)]
        titles[pdf.stem] = list(dict.fromkeys(n for n in names if n))
    return titles


def build_graph(settings: PipelineSettings) -> RelationGraph:
    """Assemble graph.json from cached extractions. No LLM calls."""
    import pymupdf
    from graphify.build import build_from_json
    from graphify.cluster import cluster
    from graphify.export import to_json

    pdfs = sorted(Path(settings.pdf_directory).glob("*.pdf"))
    extractions = {}
    for pdf in pdfs:
        cached = Path(settings.relations_dir) / f"{pdf.stem}.json"
        if cached.is_file():
            extractions[pdf.stem] = json.loads(cached.read_text())

    texts = {}
    for pdf in pdfs:
        with pymupdf.open(pdf) as doc:
            texts[pdf.stem] = "\n".join(page.get_text() for page in doc)
    titles = _paper_titles(pdfs)

    merged = merge_extractions(extractions, titles, title_citations(texts, titles))
    # dedup=False: Graphify's fuzzy merge is tuned for code symbols, and exact
    # merging above keeps distinct concepts such as SQuAD v1.1 and v2.0 apart.
    graph = build_from_json(merged, directed=True, dedup=False)
    path = graph_path(settings)
    path.parent.mkdir(parents=True, exist_ok=True)
    to_json(graph, cluster(graph), str(path), force=True)
    return RelationGraph.load(path)


__all__ = [
    "RelationGraph",
    "build_graph",
    "extract_paper",
    "format_relations",
    "graph_path",
    "graph_search",
    "merge_extractions",
    "norm_label",
    "title_citations",
    "with_relations",
]
