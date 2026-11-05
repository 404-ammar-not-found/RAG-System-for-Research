from __future__ import annotations

"""On-device embeddings — no API key, no quota, no network after the first run.

The hosted embedders are the pipeline's hard rate limit: the Gemini free tier
allows 1000 embeddings per day, and a single corpus re-embed spends most of
that in one sitting. This runs all-MiniLM-L6-v2 through
onnxruntime instead, which chromadb already depends on — so it costs no new
package, only a ~79MB model cached under ~/.cache/chroma on first use.

The trade is quality and width: 384 dimensions against Gemini's 3072, and
weaker retrieval on paraphrase-heavy questions. It matters less here than it
would elsewhere, because dense hits are fused with BM25 and then reranked by an
LLM — the vectors have to be good enough to get a passage into the candidate
set, not to rank it.
"""

from typing import Any

MODEL_NAME = "all-MiniLM-L6-v2"
DIMENSIONS = 384

_model: Any = None


def _embedder() -> Any:
    """The ONNX session, built once — loading it costs about a second."""
    global _model
    if _model is None:
        from chromadb.utils.embedding_functions import ONNXMiniLM_L6_V2

        _model = ONNXMiniLM_L6_V2()
    return _model


def embed_texts(texts: list[str]) -> list[list[float]]:
    return [list(map(float, vector)) for vector in _embedder()(texts)]


class LocalEmbeddings:
    """The LangChain `Embeddings` surface, which is all Chroma asks for."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return embed_texts(list(texts))

    def embed_query(self, text: str) -> list[float]:
        return embed_texts([text])[0]


__all__ = ["DIMENSIONS", "MODEL_NAME", "LocalEmbeddings", "embed_texts"]
