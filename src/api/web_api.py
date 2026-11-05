from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager
from functools import lru_cache
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel

from src.pipeline.parsing import pdf_title
from src.pipeline.runner import Runtime
from src.pipeline.scholarly import (
    corpus_graph,
    download_arxiv_pdf,
    parse_arxiv_id,
    resolve_paper,
    to_bibtex,
)
from src.pipeline.settings import PipelineSettings

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
CHROMA_DIR = REPO_ROOT / "chroma_db"

SETTINGS = PipelineSettings(
    pdf_directory=DATA_DIR,
    chroma_path=str(CHROMA_DIR),
    debug_retrieval=False,
)

runtime = Runtime(SETTINGS)


class ArxivRequest(BaseModel):
    url: str


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: list[ChatMessage]


@asynccontextmanager
async def lifespan(app: FastAPI):
    await runtime.start()
    yield


app = FastAPI(title="Neuro-Symbolic RAG Web API", version="0.2.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _safe_pdf_name(filename: str) -> str:
    base = Path(filename or "uploaded.pdf").name
    if not base.lower().endswith(".pdf"):
        base = f"{base}.pdf"
    return base


@app.get("/api/health")
async def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "chunks": runtime.qa.bm25.size if runtime.qa else 0,
    }


@app.post("/api/upload")
async def upload(file: UploadFile = File(...)) -> dict[str, Any]:
    safe_name = _safe_pdf_name(file.filename or "uploaded.pdf")
    payload = await file.read()
    if not payload:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    if not payload.startswith(b"%PDF"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    target = DATA_DIR / safe_name
    counter = 1
    while target.exists():
        target = DATA_DIR / f"{Path(safe_name).stem}-{counter}.pdf"
        counter += 1
    target.write_bytes(payload)
    return await _ingest(target)


async def _ingest(pdf: Path) -> dict[str, Any]:
    """Ingest one paper. Already-indexed PDFs are skipped by content hash."""
    try:
        counts = await runtime.ingest(only=pdf)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return {
        "message": "Ingest complete.",
        "filename": pdf.name,
        "newChunks": counts["chunks"],
    }


@app.post("/api/arxiv")
async def add_arxiv(payload: ArxivRequest) -> dict[str, Any]:
    arxiv_id = parse_arxiv_id(payload.url)
    if not arxiv_id:
        raise HTTPException(
            status_code=400,
            detail="Not an arXiv link or id — try https://arxiv.org/abs/1706.03762",
        )

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    target = DATA_DIR / f"{arxiv_id.replace('/', '-')}.pdf"
    if not target.exists():
        try:
            pdf = await asyncio.to_thread(download_arxiv_pdf, arxiv_id)
        except Exception as exc:
            raise HTTPException(
                status_code=502, detail=f"Could not download arXiv:{arxiv_id} ({exc})"
            ) from exc
        target.write_bytes(pdf)

    return await _ingest(target)


_title_cache: dict[str, str] = {}


def _title_from_pdf(name: str) -> str:
    if name not in _title_cache:
        try:
            _title_cache[name] = pdf_title(DATA_DIR / name)
        except Exception:
            _title_cache[name] = ""
    return _title_cache[name]


@app.get("/api/papers")
async def papers() -> dict[str, Any]:
    """The corpus as {file, title} pairs.

    Titles come from ingested chunk metadata; PDFs without chunks in the
    current collection fall back to a first-page extraction, then filename.
    """
    titles: dict[str, str] = {}
    if runtime.vectordb is not None:
        got = await asyncio.to_thread(
            runtime.vectordb._collection.get, include=["metadatas"]
        )
        for meta in got.get("metadatas") or []:
            name = Path(str(meta.get("source", ""))).name
            title = str(meta.get("paper_title") or "").strip()
            if name and title:
                titles.setdefault(name, title)

    names = sorted(p.name for p in DATA_DIR.glob("*.pdf")) if DATA_DIR.exists() else []
    out = []
    for n in names:
        title = titles.get(n) or await asyncio.to_thread(_title_from_pdf, n)
        out.append({"file": n, "title": title or n.removesuffix(".pdf")})
    return {"papers": out}


@app.delete("/api/papers/{name}")
async def delete_paper(name: str) -> dict[str, Any]:
    safe = _safe_pdf_name(name)
    target = DATA_DIR / safe
    if not target.is_file():
        raise HTTPException(status_code=404, detail="Unknown paper.")
    await runtime.delete_paper(target)
    _title_cache.pop(safe, None)
    return {"message": "Deleted.", "filename": safe}


@app.get("/api/bibtex")
async def bibtex() -> Response:
    """The whole library as one .bib file. resolve_paper is cache-fast for
    ingested papers; entries degrade with missing metadata, never fail."""
    pdfs = sorted(DATA_DIR.glob("*.pdf")) if DATA_DIR.exists() else []
    entries = []
    for pdf in pdfs:
        title = await asyncio.to_thread(_title_from_pdf, pdf.name)
        record = await asyncio.to_thread(resolve_paper, pdf, title)
        entries.append(to_bibtex(record))
    return Response(
        "\n".join(entries),
        media_type="application/x-bibtex",
        headers={"Content-Disposition": 'attachment; filename="library.bib"'},
    )


@app.get("/api/graph")
async def graph() -> dict[str, Any]:
    """Corpus-internal citation map, from cached bibliographic records only."""
    pdfs = sorted(DATA_DIR.glob("*.pdf")) if DATA_DIR.exists() else []
    return await asyncio.to_thread(corpus_graph, pdfs)


@lru_cache(maxsize=32)  # ponytail: in-memory PNG cache; move to disk past ~100 papers
def _render_page(name: str, page: int, caption: str) -> bytes:
    import pymupdf

    with pymupdf.open(DATA_DIR / name) as doc:
        idx = max(0, min(page, doc.page_count - 1))
        # Chunk metadata carries the section's FIRST page; the caption may sit
        # a few pages later, so search a short window for it.
        if caption:
            for p in range(idx, min(idx + 5, doc.page_count)):
                if doc[p].search_for(caption):
                    idx = p
                    break
        return doc[idx].get_pixmap(dpi=150).tobytes("png")


@app.get("/api/page-image")
async def page_image(source: str, page: int = 0, caption: str = "") -> Response:
    name = _safe_pdf_name(source)  # Path(...).name — no traversal survives
    if not (DATA_DIR / name).is_file():
        raise HTTPException(status_code=404, detail="Unknown paper.")
    png = await asyncio.to_thread(_render_page, name, page, caption[:60])
    return Response(
        png, media_type="image/png", headers={"Cache-Control": "public, max-age=86400"}
    )


@app.post("/api/chat")
async def chat(payload: ChatRequest) -> StreamingResponse:
    messages = [m for m in payload.messages if m.content.strip()]
    if not messages or messages[-1].role != "user":
        raise HTTPException(status_code=400, detail="Last message must be from the user.")
    if runtime.qa is None:
        raise HTTPException(status_code=503, detail="Runtime not started.")

    question = messages[-1].content.strip()
    history = [{"role": m.role, "content": m.content} for m in messages[:-1]]

    async def gen():
        try:
            async for event in runtime.qa.answer_stream(question, history=history):
                yield json.dumps(event) + "\n"
        except Exception as exc:
            # Headers are already sent, so errors travel in-band.
            yield json.dumps({"type": "error", "detail": str(exc)}) + "\n"

    return StreamingResponse(gen(), media_type="application/x-ndjson")
