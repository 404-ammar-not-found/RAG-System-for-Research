#!/usr/bin/env python
"""Full rebuild: parse every PDF once, fill Chroma and the graph, export.

Graph ingest is ordered shortest-paper-first so a daily embedding quota that
runs out mid-run leaves the widest cross-paper coverage rather than one giant
paper and nothing else.
"""

from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.pipeline.graph_export import export_graph  # noqa: E402
from src.pipeline.graph_store import GraphStore  # noqa: E402
from src.pipeline.ingestion import ingest_chroma, open_vectorstore  # noqa: E402
from src.pipeline.parsing import parse_paper  # noqa: E402
from src.pipeline.settings import PipelineSettings  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]


async def main() -> None:
    t0 = time.time()
    settings = PipelineSettings(
        pdf_directory=REPO_ROOT / "data",
        chroma_path=str(REPO_ROOT / "chroma_db"),
    )

    print("=== Chroma ===", flush=True)
    vectordb, chunks, _ = ingest_chroma(settings)
    print(f"chroma: {chunks} new chunks, {vectordb._collection.count()} total", flush=True)

    print("\n=== Graph ===", flush=True)
    graph = GraphStore(settings)
    await graph.initialize()

    by_paper = [parse_paper(p, settings.max_pages) for p in sorted(settings.pdf_directory.glob("*.pdf"))]
    by_paper.sort(key=len)
    sections = [s for paper in by_paper for s in paper]
    print(f"{len(sections)} sections from {len(by_paper)} papers", flush=True)

    added = await graph.ingest_sections(sections)
    nodes, edges = await graph.dump()
    print(f"\ngraph: {added}/{len(sections)} episodes, {len(nodes)} entities, {len(edges)} relations",
          flush=True)

    await export_graph(graph, open_vectorstore(settings))
    await graph.close()
    print(f"done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
