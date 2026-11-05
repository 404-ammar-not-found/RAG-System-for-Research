from __future__ import annotations

"""Shared runtime: opens the store, ingests, answers. Used by the CLI and the API."""

import asyncio
import hashlib
from pathlib import Path
from typing import Any

from .ingestion import ingest_chroma, open_vectorstore
from .qa import QaEngine
from .settings import PipelineSettings


class Runtime:
    """Keeps the vector store, BM25 index and LLM warm."""

    def __init__(self, settings: PipelineSettings) -> None:
        self.settings = settings
        self.vectordb: Any | None = None
        self.qa: QaEngine | None = None
        self._lock = asyncio.Lock()

    async def start(self) -> None:
        self.vectordb = await asyncio.to_thread(open_vectorstore, self.settings)
        self.qa = QaEngine(self.settings, self.vectordb)
        await asyncio.to_thread(self.qa.rebuild_bm25)

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
