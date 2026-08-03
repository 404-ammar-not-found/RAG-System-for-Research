#!/usr/bin/env python
"""Rebuild web-visualizer/public/data/knowledge-graph.json from both stores."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.pipeline.graph_export import export_graph  # noqa: E402
from src.pipeline.graph_store import GraphStore  # noqa: E402
from src.pipeline.ingestion import open_vectorstore  # noqa: E402
from src.pipeline.settings import PipelineSettings  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]


async def main() -> None:
    settings = PipelineSettings(
        pdf_directory=REPO_ROOT / "data",
        chroma_path=str(REPO_ROOT / "chroma_db"),
    )
    graph = GraphStore(settings)
    try:
        await export_graph(graph, open_vectorstore(settings))
    finally:
        await graph.close()


if __name__ == "__main__":
    asyncio.run(main())
