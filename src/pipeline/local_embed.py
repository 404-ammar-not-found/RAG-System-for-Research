from __future__ import annotations

"""On-device embeddings — no API key, no quota, no network after the first run.

The hosted embedders are the pipeline's hard rate limit: the Gemini free tier
allows 1000 embeddings per day, and a single corpus re-embed plus graph
extraction spends that in one sitting. This runs all-MiniLM-L6-v2 through
onnxruntime instead, which chromadb already depends on — so it costs no new
package, only a ~79MB model cached under ~/.cache/chroma on first use.

The trade is quality and width: 384 dimensions against Gemini's 3072, and
weaker retrieval on paraphrase-heavy questions. It matters less here than it
would elsewhere, because dense hits are fused with BM25 and then reranked by an
LLM — the vectors have to be good enough to get a passage into the candidate
set, not to rank it.
"""

import asyncio
from collections.abc import Iterable
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


def graphiti_embedder() -> Any:
    """Graphiti's `EmbedderClient`, wrapping the same session.

    Built lazily inside a function because importing graphiti_core costs a
    second of import time that the passage-only path should not pay.
    """
    from graphiti_core.embedder.client import EmbedderClient, EmbedderConfig

    class LocalGraphitiEmbedder(EmbedderClient):
        def __init__(self) -> None:
            self.config = EmbedderConfig(embedding_dim=DIMENSIONS)

        async def create(
            self, input_data: str | list[str] | Iterable[int] | Iterable[Iterable[int]]
        ) -> list[float]:
            texts = [input_data] if isinstance(input_data, str) else list(input_data)
            # to_thread: onnxruntime is blocking, and Graphiti calls this from
            # inside the event loop that is also driving its LLM requests.
            vectors = await asyncio.to_thread(embed_texts, [str(t) for t in texts])
            return vectors[0]

        async def create_batch(self, input_data_list: list[str]) -> list[list[float]]:
            if not input_data_list:
                return []
            return await asyncio.to_thread(embed_texts, list(input_data_list))

    return LocalGraphitiEmbedder()


__all__ = ["DIMENSIONS", "MODEL_NAME", "LocalEmbeddings", "embed_texts", "graphiti_embedder"]
