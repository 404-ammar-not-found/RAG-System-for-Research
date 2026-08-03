from __future__ import annotations

"""Shared runtime: opens both stores, ingests, answers. Used by the CLI and the API."""

import asyncio
from typing import Any

from .graph_export import export_graph
from .graph_store import GraphStore
from .ingestion import ingest_chroma, open_vectorstore
from .qa import QaEngine
from .settings import PipelineSettings


class Runtime:
    """Keeps the vector store, BM25 index, graph client and LLM warm."""

    def __init__(self, settings: PipelineSettings) -> None:
        self.settings = settings
        self.vectordb: Any | None = None
        self.graph: GraphStore | None = None
        self.qa: QaEngine | None = None
        self._lock = asyncio.Lock()

    async def start(self, *, with_graph: bool = True) -> None:
        """Open both stores. Falls back to passages-only if FalkorDB is unreachable."""
        self.vectordb = await asyncio.to_thread(open_vectorstore, self.settings)

        if with_graph:
            try:
                graph = GraphStore(self.settings)
                await graph.initialize()
                self.graph = graph
            except Exception as exc:
                print(
                    f"[WARN] FalkorDB unavailable ({exc}).\n"
                    "       Running passage-only. Start it with:\n"
                    "       docker run -d -p 6380:6379 -p 3000:3000 falkordb/falkordb:latest"
                )
                self.graph = None

        self.qa = QaEngine(self.settings, self.vectordb, self.graph)
        await asyncio.to_thread(self.qa.rebuild_bm25)

    async def close(self) -> None:
        if self.graph is not None:
            await self.graph.close()

    async def ingest(self) -> dict[str, int]:
        """Parse each new PDF once, feed both stores, refresh the index and export."""
        async with self._lock:
            vectordb, chunks, sections = await asyncio.to_thread(ingest_chroma, self.settings)
            self.vectordb = vectordb
            if self.qa is not None:
                self.qa.vectordb = vectordb

            episodes = 0
            if self.graph is not None and sections:
                print(f"Extracting entities from {len(sections)} sections...")
                episodes = await self.graph.ingest_sections(sections)

            if self.qa is not None:
                await asyncio.to_thread(self.qa.rebuild_bm25)
            await self.refresh_export()
            return {"chunks": chunks, "episodes": episodes}

    async def refresh_export(self) -> None:
        if self.graph is None:
            print("[WARN] no graph store; skipping export")
            return
        await export_graph(self.graph, self.vectordb)

    async def ask(self, question: str) -> dict[str, Any]:
        if self.qa is None:
            raise RuntimeError("Runtime not started")
        return await self.qa.answer(question)


async def run_cli(settings: PipelineSettings) -> None:
    """Ingest, then answer one question from stdin."""
    if not settings.pdf_directory.exists():
        print(f"PDF directory not found: {settings.pdf_directory}")
        return

    runtime = Runtime(settings)
    await runtime.start()
    try:
        counts = await runtime.ingest()
        print(f"Ingested {counts['chunks']} chunks, {counts['episodes']} graph episodes.")

        question = input("\nEnter your research question (empty to quit): ").strip()
        if not question:
            return

        result = await runtime.ask(question)
        if result["facts"]:
            print("\n--- Graph facts ---")
            for f in result["facts"]:
                print(f"  [{f['name']}] {f['fact']}")
        print("\n--- Answer ---\n")
        print(result["answer"])
    finally:
        await runtime.close()


__all__ = ["Runtime", "run_cli"]
