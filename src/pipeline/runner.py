from __future__ import annotations

"""Shared runtime: opens the store, ingests, answers. Used by the CLI and the API."""

import asyncio
import hashlib
from pathlib import Path
from typing import Any

from .ingestion import ingest_chroma, open_vectorstore
from .qa import QaEngine
from .relations import RelationGraph, build_graph, extract_paper, graph_path
from .settings import PipelineSettings


class Runtime:
    """Keeps the vector store, BM25 index and LLM warm."""

    def __init__(self, settings: PipelineSettings) -> None:
        self.settings = settings
        self.vectordb: Any | None = None
        self.qa: QaEngine | None = None
        self._lock = asyncio.Lock()
        self._graph_tasks: set[asyncio.Task[None]] = set()

    async def start(self) -> None:
        self.vectordb = await asyncio.to_thread(open_vectorstore, self.settings)
        self.qa = QaEngine(self.settings, self.vectordb)
        await asyncio.to_thread(self.qa.rebuild_bm25)
        self.qa.relations = await asyncio.to_thread(RelationGraph.load, graph_path(self.settings))

    async def _refresh_relations(self, new_pdf: Path | None = None) -> None:
        """Update the graph without ever failing the ingest or delete that asked for it."""
        try:
            if new_pdf is not None:
                await asyncio.to_thread(extract_paper, new_pdf, self.settings)
            relations = await asyncio.to_thread(build_graph, self.settings)
            if self.qa is not None:
                self.qa.relations = relations
        except Exception as exc:
            print(f"[WARN] relation graph not updated: {str(exc)[:200]}")

    def _refresh_in_background(self, new_pdf: Path | None = None) -> None:
        # Extraction takes minutes, and retrieval works without the new links.
        task = asyncio.create_task(self._refresh_relations(new_pdf))
        self._graph_tasks.add(task)
        task.add_done_callback(self._graph_tasks.discard)

    async def ingest(self, only: Path | None = None) -> dict[str, int]:
        """Parse each new PDF once, index it, refresh the BM25 index.

        `only` limits the run to one PDF; the default sweeps the whole directory.
        """
        async with self._lock:
            vectordb, chunks, _sections = await asyncio.to_thread(
                ingest_chroma, self.settings, only
            )
            self.vectordb = vectordb
            if self.qa is not None:
                self.qa.vectordb = vectordb
                await asyncio.to_thread(self.qa.rebuild_bm25)
            if only is not None and chunks and self.settings.relation_graph:
                self._refresh_in_background(only)
            return {"chunks": chunks}

    async def delete_paper(self, pdf: Path) -> None:
        """Remove a paper's chunks and its PDF, then refresh the lexical index."""
        async with self._lock:
            # Same content-hash key ingestion dedupes on (and its rollback deletes by).
            file_hash = hashlib.sha256(pdf.read_bytes()).hexdigest()
            if self.vectordb is not None:
                await asyncio.to_thread(
                    self.vectordb._collection.delete, where={"file_hash": file_hash}
                )
            pdf.unlink()
            if self.qa is not None:
                await asyncio.to_thread(self.qa.rebuild_bm25)
            if self.settings.relation_graph:
                self._refresh_in_background()

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
    counts = await runtime.ingest()
    print(f"Ingested {counts['chunks']} chunks.")

    question = input("\nEnter your research question (empty to quit): ").strip()
    if not question:
        return

    result = await runtime.ask(question)
    print("\n--- Answer ---\n")
    print(result["answer"])


__all__ = ["Runtime", "run_cli"]
