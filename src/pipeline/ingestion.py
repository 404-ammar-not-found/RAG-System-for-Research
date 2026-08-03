from __future__ import annotations

from pathlib import Path
from typing import Any

from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from .deps import get_api_key, lc_deps
from .parsing import Section, parse_paper, section_to_chunks
from .settings import PipelineSettings

# Gemini free tier allows 100 embed requests/minute. Batch, and back off when
# the server says 429 — an ingest that dies half way leaves a partial store.
EMBED_BATCH = 50


class DailyQuotaExhausted(RuntimeError):
    """The per-day embedding cap is gone. Waiting will not help; stop cleanly."""


def _is_rate_limit(exc: BaseException) -> bool:
    """Retry per-minute throttling only.

    A PerDay quota does not refill on a 90-second backoff, so retrying it just
    burns wall-clock and a few more requests before failing anyway.
    """
    if isinstance(exc, DailyQuotaExhausted):
        return False
    text = str(exc)
    if "PerDay" in text:
        return False
    return "429" in text or "RESOURCE_EXHAUSTED" in text or "quota" in text.lower()


@retry(
    retry=retry_if_exception(_is_rate_limit),
    wait=wait_exponential(multiplier=8, min=8, max=90),
    stop=stop_after_attempt(6),
    reraise=True,
)
def _add_batch(vectordb: Any, texts: list[str], metadatas: list[dict], ids: list[str]) -> None:
    try:
        vectordb.add_texts(texts=texts, metadatas=metadatas, ids=ids)
    except Exception as exc:
        if "PerDay" in str(exc):
            raise DailyQuotaExhausted(
                "Daily embedding quota exhausted. Ingest resumes on reset — already "
                "ingested PDFs are skipped by content hash, so just re-run.\n"
                "       A single successful embed does NOT mean there is room for a "
                "batch; check projected cost with "
                "`python scripts/eval_retrieval.py --offline --report-cost`."
            ) from exc
        raise


def open_vectorstore(settings: PipelineSettings) -> Any:
    """Open (or create) the Chroma collection.

    `hnsw:space: cosine` matters: Chroma defaults to L2, which is the wrong
    metric for Gemini embeddings. Chroma only honours this at creation time —
    changing it requires deleting the collection.
    """
    deps = lc_deps()
    embeddings = deps["GoogleGenerativeAIEmbeddings"](
        model=settings.embed_model,
        google_api_key=get_api_key(),
        dimensions=settings.embed_dimensions,
    )
    return deps["Chroma"](
        collection_name=settings.collection_name,
        embedding_function=embeddings,
        persist_directory=settings.chroma_path,
        collection_metadata={"hnsw:space": "cosine"},
    )


def _has_file_in_store(vectordb: Any, file_hash: str) -> bool:
    try:
        result = vectordb._collection.get(where={"file_hash": file_hash}, limit=1)
        return bool(result.get("ids"))
    except Exception:
        return False


def pending_pdfs(vectordb: Any, settings: PipelineSettings) -> list[tuple[Path, list[Section]]]:
    """Parse every not-yet-ingested PDF once. Sections feed Chroma and Graphiti both."""
    out: list[tuple[Path, list[Section]]] = []
    for pdf_path in sorted(settings.pdf_directory.glob("*.pdf")):
        sections = parse_paper(pdf_path, settings.max_pages)
        if not sections:
            print(f"- skipping {pdf_path.name} (no text extracted)")
            continue
        if _has_file_in_store(vectordb, sections[0].file_hash):
            print(f"- skipping {pdf_path.name} (already ingested)")
            continue
        out.append((pdf_path, sections))
    return out


def build_chunk_records(
    sections: list[Section], settings: PipelineSettings
) -> list[dict[str, Any]]:
    """Chunks with ids and prev/next links, ready to embed.

    Shared with the eval harness so the offline scorer sees exactly the chunks
    that would be indexed — otherwise the harness measures a fiction.
    """
    records: list[dict[str, Any]] = []
    per_source: dict[str, int] = {}

    for section in sections:
        if section.is_reference and not settings.index_references:
            continue
        for chunk in section_to_chunks(section, settings.chunk_size, settings.chunk_overlap):
            meta = chunk["metadata"]
            source = meta["source"]
            idx = per_source.get(source, 0)
            per_source[source] = idx + 1
            meta["chunk_index"] = idx
            # The citation lives in metadata only. Prepending it to the text
            # (as this pipeline used to) embeds a file path into every vector.
            meta["id"] = f"{Path(source).stem}-{meta['file_hash'][:8]}-chunk-{idx}"
            records.append({"text": chunk["text"], "metadata": meta})

    # Link neighbours within each paper so retrieval can expand by id later.
    by_source: dict[str, list[dict[str, Any]]] = {}
    for rec in records:
        by_source.setdefault(rec["metadata"]["source"], []).append(rec)
    for group in by_source.values():
        for i, rec in enumerate(group):
            if i:
                rec["metadata"]["prev_id"] = group[i - 1]["metadata"]["id"]
            if i + 1 < len(group):
                rec["metadata"]["next_id"] = group[i + 1]["metadata"]["id"]
    return records


def index_sections(vectordb: Any, sections: list[Section], settings: PipelineSettings) -> int:
    """Sub-split sections into chunks and add them to Chroma."""
    records = build_chunk_records(sections, settings)
    texts = [r["text"] for r in records]
    metadatas = [r["metadata"] for r in records]
    ids = [r["metadata"]["id"] for r in records]

    for start in range(0, len(texts), EMBED_BATCH):
        end = start + EMBED_BATCH
        _add_batch(vectordb, texts[start:end], metadatas[start:end], ids[start:end])
        print(f"  embedded {min(end, len(texts))}/{len(texts)} chunks")
    return len(texts)


def ingest_chroma(settings: PipelineSettings) -> tuple[Any, int, list[Section]]:
    """Ingest new PDFs into Chroma. Returns the store, chunk count, and sections.

    The returned sections are handed to the Graphiti layer so each PDF is parsed
    exactly once.
    """
    vectordb = open_vectorstore(settings)
    print(f"Reading PDFs from {settings.pdf_directory.resolve()}...")

    total = 0
    all_sections: list[Section] = []
    for pdf_path, sections in pending_pdfs(vectordb, settings):
        print(f"- ingesting {pdf_path.name} ({len(sections)} sections)")
        try:
            total += index_sections(vectordb, sections, settings)
        except Exception:
            # Roll back the partial write. Otherwise the file_hash dedupe sees
            # the surviving chunks next run and skips the PDF forever.
            print(f"  failed; rolling back partial chunks for {pdf_path.name}")
            vectordb._collection.delete(where={"file_hash": sections[0].file_hash})
            raise
        all_sections.extend(sections)

    # No persist() call: chromadb 0.4+ persists on write and the method was
    # removed from the LangChain wrapper.
    return vectordb, total, all_sections


__all__ = ["ingest_chroma", "index_sections", "open_vectorstore", "pending_pdfs"]
