from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.pipeline.runner import Runtime
from src.pipeline.scholarly import download_arxiv_pdf, parse_arxiv_id
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


class AskRequest(BaseModel):
    question: str


class ArxivRequest(BaseModel):
    url: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Graphiti must be constructed inside the running loop.
    await runtime.start()
    yield
    await runtime.close()


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
        "graph": runtime.graph is not None,
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
        "message": "Ingest complete. Both stores refreshed.",
        "filename": pdf.name,
        "newChunks": counts["chunks"],
        "newEpisodes": counts["episodes"],
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


@app.post("/api/ask")
async def ask(payload: AskRequest) -> dict[str, Any]:
    question = payload.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    try:
        return await runtime.ask(question)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
