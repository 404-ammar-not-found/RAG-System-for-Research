from __future__ import annotations

"""Export the knowledge graph for the 3D visualizer.

Two layers in one file:
  entities — Graphiti nodes, joined by named relations (the default view)
  chunks   — Chroma passages, joined in reading order, linked to the entities
             their section mentions

Rebuilt from scratch every time. The old exporter merged into the previous
file and never evicted, which is why the committed JSON accumulated duplicate
nodes under two different id schemes.
"""

import json
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_PATH = REPO_ROOT / "web-visualizer" / "public" / "data" / "knowledge-graph.json"

_SKIP_LABELS = {"Entity"}  # Graphiti's base label, present on everything


def _entity_type(node: Any) -> str:
    labels = [l for l in (getattr(node, "labels", []) or []) if l not in _SKIP_LABELS]
    return labels[0] if labels else "Entity"


def _rel_source(source: str) -> str:
    """Normalise to a repo-relative path so one paper is one group."""
    try:
        return str(Path(source).resolve().relative_to(REPO_ROOT))
    except (ValueError, OSError):
        return str(source)


def build_entity_layer(nodes: list[Any], edges: list[Any]) -> tuple[list[dict], list[dict]]:
    out_nodes = [
        {
            "id": n.uuid,
            "kind": "entity",
            "name": n.name,
            "label": n.name,
            "type": _entity_type(n),
            "group": _entity_type(n),
            "summary": (getattr(n, "summary", "") or "")[:400],
        }
        for n in nodes
    ]
    known = {n["id"] for n in out_nodes}
    out_links = [
        {
            "source": e.source_node_uuid,
            "target": e.target_node_uuid,
            "kind": "relation",
            "label": getattr(e, "name", "") or "RELATED",
            "fact": (getattr(e, "fact", "") or "")[:300],
        }
        for e in edges
        if e.source_node_uuid in known and e.target_node_uuid in known
    ]
    return out_nodes, out_links


def build_chunk_layer(vectordb: Any) -> tuple[list[dict], list[dict], dict[str, list[str]]]:
    """Chunk nodes + reading-order links. Also returns episode_name -> chunk ids."""
    raw = vectordb._collection.get(include=["documents", "metadatas"])
    ids = raw.get("ids") or []
    docs = raw.get("documents") or []
    metas = raw.get("metadatas") or []

    nodes: list[dict] = []
    by_source: dict[str, list[tuple[int, str]]] = {}
    by_episode: dict[str, list[str]] = {}

    for cid, doc, meta in zip(ids, docs, metas):
        meta = meta or {}
        source = _rel_source(str(meta.get("source", "unknown")))
        section = str(meta.get("section", ""))
        text = doc or ""
        nodes.append(
            {
                "id": cid,
                "kind": "chunk",
                "label": text[:120] + ("…" if len(text) > 120 else ""),
                "group": source,
                "section": section,
                "paper_title": meta.get("paper_title", ""),
                "page": meta.get("page", 0),
            }
        )
        by_source.setdefault(source, []).append((int(meta.get("chunk_index", 0)), cid))
        episode = f"{Path(source).stem}::{section}"
        by_episode.setdefault(episode, []).append(cid)

    links: list[dict] = []
    for chunk_list in by_source.values():
        chunk_list.sort()  # by chunk_index, not lexically — "chunk-10" < "chunk-9"
        for (_, a), (_, b) in zip(chunk_list, chunk_list[1:]):
            links.append({"source": a, "target": b, "kind": "sequence"})

    return nodes, links, by_episode


def build_graph(
    entity_nodes: list[Any],
    entity_edges: list[Any],
    vectordb: Any | None,
    episode_entities: dict[str, list[str]] | None = None,
) -> dict[str, Any]:
    nodes, links = build_entity_layer(entity_nodes, entity_edges)

    if vectordb is not None:
        chunk_nodes, chunk_links, chunks_by_episode = build_chunk_layer(vectordb)
        nodes.extend(chunk_nodes)
        links.extend(chunk_links)

        # Provenance: entity -> chunk, joined on the shared "<stem>::<section>" key.
        known_entities = {n["id"] for n in nodes if n["kind"] == "entity"}
        for episode, entity_uuids in (episode_entities or {}).items():
            for chunk_id in chunks_by_episode.get(episode, []):
                for uuid in entity_uuids:
                    if uuid in known_entities:
                        links.append(
                            {"source": uuid, "target": chunk_id, "kind": "mentions"}
                        )

        # Provenance: entity -> paper, for the entities the join above cannot
        # reach. A section with no chunks has none to point at — references are
        # excluded from the passage store on purpose, and they are where most of
        # the author and institution entities come from. The paper is still
        # known, because the episode is named "<stem>::<section>", so the entity
        # gets its source paper rather than floating unattached.
        paper_of_stem = {Path(n["group"]).stem: n["group"] for n in chunk_nodes}
        seen: set[tuple[str, str]] = set()
        for episode, entity_uuids in (episode_entities or {}).items():
            source = paper_of_stem.get(str(episode).split("::", 1)[0])
            if not source:
                continue
            for uuid in entity_uuids:
                if uuid in known_entities and (uuid, source) not in seen:
                    seen.add((uuid, source))
                    links.append({"source": uuid, "target": source, "kind": "from_paper"})

    return {"nodes": nodes, "links": links}


def write_graph(graph: dict[str, Any], path: Path = OUTPUT_PATH) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(graph, indent=2), encoding="utf-8")
    counts: dict[str, int] = {}
    for link in graph["links"]:
        counts[link["kind"]] = counts.get(link["kind"], 0) + 1
    entities = sum(1 for n in graph["nodes"] if n["kind"] == "entity")
    print(
        f"Wrote {path.relative_to(REPO_ROOT)}: {entities} entities, "
        f"{len(graph['nodes']) - entities} chunks, links={counts}"
    )
    return path


async def export_graph(graph_store: Any, vectordb: Any | None, path: Path = OUTPUT_PATH) -> Path:
    nodes, edges = await graph_store.dump()
    episode_entities = await graph_store.episode_map()
    return write_graph(build_graph(nodes, edges, vectordb, episode_entities), path)


__all__ = [
    "OUTPUT_PATH",
    "build_chunk_layer",
    "build_entity_layer",
    "build_graph",
    "export_graph",
    "write_graph",
]
