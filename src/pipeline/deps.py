from __future__ import annotations

"""Provider wiring: which LLM and embedding service the pipeline talks to.

Three providers, selected by whichever API key is present (`LLM_PROVIDER` in
`.env` forces one). Anthropic publishes no embedding endpoint, so an Anthropic
key covers chat only and embeddings fall back to Gemini or OpenAI — both stores
need vectors from somewhere, so an Anthropic-only setup cannot ingest.
"""

import os
from typing import Any

PROVIDERS: dict[str, dict[str, Any]] = {
    "gemini": {
        "keys": ("GEMINI_API_KEY", "GOOGLE_API_KEY"),
        "text": "gemini-flash-latest",
        "graph": "gemini-flash-lite-latest",
        # Embedding quota is metered per model on the free tier, so the passage
        # store and the graph deliberately use different ones — see settings.
        "embed": "models/gemini-embedding-2",
        "graph_embed": "models/gemini-embedding-2-preview",
    },
    "anthropic": {
        "keys": ("ANTHROPIC_API_KEY",),
        "text": "claude-opus-5",
        # Extraction and reranking run once per section, so they take the cheap
        # model. It is also the only safe choice for Graphiti's AnthropicClient,
        # which always sends `temperature` — a parameter the Claude 5 series
        # rejects with a 400. Haiku 4.5 still accepts it.
        "graph": "claude-haiku-4-5",
        "embed": None,  # no embedding API
        "graph_embed": None,
    },
    "openai": {
        "keys": ("OPENAI_API_KEY",),
        # OpenAI retires model ids on its own schedule. If one 404s, override it
        # in .env (TEXT_LLM_MODEL / GRAPH_LLM_MODEL / EMBED_MODEL) rather than
        # editing this table.
        "text": "gpt-5",
        "graph": "gpt-5-mini",
        "embed": "text-embedding-3-small",
        "graph_embed": "text-embedding-3-large",
    },
    # On-device, via onnxruntime (already a chromadb dependency). No key, no
    # quota — the answer to "the free tier is out of embeddings for today".
    # Embeddings only: there is no local chat model here.
    "local": {
        "keys": (),
        "text": None,
        "graph": None,
        "embed": "all-MiniLM-L6-v2",
        "graph_embed": "all-MiniLM-L6-v2",
    },
}

# Order tried when nothing is forced: Gemini first, because its free tier is
# what this pipeline's quota accounting is tuned for.
_ORDER = ("gemini", "anthropic", "openai")


def _load_env() -> None:
    from dotenv import load_dotenv

    load_dotenv()


def key_for(provider: str) -> str | None:
    """The configured API key for one provider, or None."""
    _load_env()
    for name in PROVIDERS[provider]["keys"]:
        if os.getenv(name):
            return os.getenv(name)
    return None


def chat_provider() -> str:
    """Provider used for answers, extraction and reranking."""
    _load_env()
    forced = (os.getenv("LLM_PROVIDER") or "").strip().lower()
    if forced:
        if forced not in PROVIDERS or not PROVIDERS[forced]["text"]:
            raise ValueError(
                f"LLM_PROVIDER must be one of {sorted(p for p in PROVIDERS if PROVIDERS[p]['text'])}"
                f", got {forced!r}"
            )
        if not key_for(forced):
            raise ValueError(f"LLM_PROVIDER={forced} but {PROVIDERS[forced]['keys'][0]} is not set.")
        return forced
    for provider in _ORDER:
        if key_for(provider):
            return provider
    raise ValueError(
        "No API key found. Set GEMINI_API_KEY, ANTHROPIC_API_KEY or OPENAI_API_KEY "
        "in your environment or .env file."
    )


def embed_provider() -> str:
    """Provider used for vectors.

    Set `EMBED_PROVIDER=local` to keep embeddings on-device regardless of which
    keys exist — the way out of a spent daily quota. Otherwise the chat provider
    embeds for itself where it can, and Anthropic (which publishes no embedding
    endpoint) borrows whichever other provider is configured, falling back to
    local rather than refusing to run.
    """
    _load_env()
    forced = (os.getenv("EMBED_PROVIDER") or "").strip().lower()
    if forced:
        if forced not in PROVIDERS or not PROVIDERS[forced]["embed"]:
            raise ValueError(
                "EMBED_PROVIDER must be one of "
                f"{sorted(p for p in PROVIDERS if PROVIDERS[p]['embed'])}, got {forced!r}"
            )
        if PROVIDERS[forced]["keys"] and not key_for(forced):
            raise ValueError(f"EMBED_PROVIDER={forced} but {PROVIDERS[forced]['keys'][0]} is not set.")
        return forced

    chat = chat_provider()
    if PROVIDERS[chat]["embed"]:
        return chat
    for provider in _ORDER:
        if PROVIDERS[provider]["embed"] and key_for(provider):
            return provider
    return "local"


def get_api_key() -> str:
    """Chat-provider key. Kept for callers that only need a key."""
    return key_for(chat_provider()) or ""


def _reranker_provider() -> str | None:
    """A configured provider Graphiti actually ships a cross-encoder for."""
    for provider in ("gemini", "openai"):
        if key_for(provider):
            return provider
    return None


def model_for(settings: Any, role: str) -> str:
    """Resolve one model slot: explicit setting first, else the provider default.

    Roles: `text` (answers), `graph` (extraction + reranking), `embed`
    (passages), `graph_embed` (graph nodes).
    """
    override = {
        "text": settings.text_llm_model,
        "graph": settings.graph_llm_model,
        "embed": settings.embed_model,
        "graph_embed": settings.graph_embed_model,
    }[role]
    provider = embed_provider() if role.endswith("embed") else chat_provider()
    return override or PROVIDERS[provider][role]


def chat_model(settings: Any, role: str, temperature: float) -> Any:
    """A LangChain chat model for `role`, on whichever provider is configured.

    `temperature` reaches Gemini only. Anthropic's Claude 5 series and OpenAI's
    reasoning models both reject sampling parameters outright (HTTP 400), so
    they are steered by prompt alone rather than by sniffing model names.
    """
    provider = chat_provider()
    model = model_for(settings, role)
    api_key = key_for(provider)

    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=model, google_api_key=api_key, temperature=temperature
        )
    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        # max_tokens is explicit because langchain-anthropic defaults to 1024,
        # which truncates an answer once extended thinking is on by default.
        return ChatAnthropic(model=model, api_key=api_key, max_tokens=8192)

    from langchain_openai import ChatOpenAI

    return ChatOpenAI(model=model, api_key=api_key)


def collection_for(settings: Any) -> str:
    """Collection name, namespaced by embedding provider.

    Vectors from different providers have different widths — 384 local against
    Gemini's 3072 — and Chroma fixes a collection's dimensionality at creation.
    Writing the wrong width into an existing collection is a hard error, so each
    provider gets its own. Gemini keeps the bare name it has always used.
    """
    provider = embed_provider()
    return (
        settings.collection_name
        if provider == "gemini"
        else f"{settings.collection_name}-{provider}"
    )


def embeddings(settings: Any) -> Any:
    """LangChain embeddings for the passage store."""
    provider = embed_provider()
    model = model_for(settings, "embed")
    api_key = key_for(provider)

    if provider == "local":
        from .local_embed import LocalEmbeddings

        return LocalEmbeddings()

    if provider == "gemini":
        from langchain_google_genai import GoogleGenerativeAIEmbeddings

        return GoogleGenerativeAIEmbeddings(
            model=model, google_api_key=api_key, dimensions=settings.embed_dimensions
        )

    from langchain_openai import OpenAIEmbeddings

    return OpenAIEmbeddings(
        model=model, api_key=api_key, dimensions=settings.embed_dimensions
    )


def graphiti_clients(settings: Any) -> dict[str, Any]:
    """The three clients Graphiti needs: LLM, embedder, cross-encoder.

    They can straddle two providers: with an Anthropic key the LLM is Claude but
    the vectors have to come from Gemini or OpenAI. Graphiti ships no Anthropic
    reranker either, so the cross-encoder follows the embedding provider in that
    case.
    """
    from graphiti_core.llm_client.config import LLMConfig

    chat, embed = chat_provider(), embed_provider()
    rerank = chat if chat in {"gemini", "openai"} else _reranker_provider()

    if chat == "gemini":
        from graphiti_core.llm_client.gemini_client import GeminiClient

        llm_client = GeminiClient(
            config=LLMConfig(
                api_key=key_for(chat),
                model=model_for(settings, "graph"),
                small_model=model_for(settings, "graph"),
            )
        )
    elif chat == "anthropic":
        from graphiti_core.llm_client.anthropic_client import AnthropicClient

        llm_client = AnthropicClient(
            config=LLMConfig(api_key=key_for(chat), model=model_for(settings, "graph"))
        )
    else:
        from graphiti_core.llm_client.openai_client import OpenAIClient

        llm_client = OpenAIClient(
            config=LLMConfig(
                api_key=key_for(chat),
                model=model_for(settings, "graph"),
                small_model=model_for(settings, "graph"),
            )
        )

    if embed == "local":
        from .local_embed import graphiti_embedder

        embedder = graphiti_embedder()
    elif embed == "gemini":
        from graphiti_core.embedder.gemini import GeminiEmbedder, GeminiEmbedderConfig

        embedder = GeminiEmbedder(
            config=GeminiEmbedderConfig(
                api_key=key_for(embed), embedding_model=model_for(settings, "graph_embed")
            )
        )
    else:
        from graphiti_core.embedder.openai import OpenAIEmbedder, OpenAIEmbedderConfig

        embedder = OpenAIEmbedder(
            config=OpenAIEmbedderConfig(
                api_key=key_for(embed), embedding_model=model_for(settings, "graph_embed")
            )
        )

    clients = {"llm_client": llm_client, "embedder": embedder}

    # Graphiti ships rerankers for Gemini and OpenAI only, so an Anthropic chat
    # provider borrows one — and if neither key exists (Anthropic plus local
    # embeddings), none is passed at all. Nothing here calls it: retrieval
    # reranks with a single listwise LLM call of its own, and Graphiti's
    # pointwise reranker is exactly what that replaced.
    if rerank == "gemini":
        from graphiti_core.cross_encoder.gemini_reranker_client import GeminiRerankerClient

        clients["cross_encoder"] = GeminiRerankerClient(
            config=LLMConfig(api_key=key_for(rerank), model=model_for(settings, "graph"))
        )
    elif rerank == "openai":
        from graphiti_core.cross_encoder.openai_reranker_client import OpenAIRerankerClient

        clients["cross_encoder"] = OpenAIRerankerClient(
            config=LLMConfig(api_key=key_for(rerank), model=PROVIDERS[rerank]["graph"])
        )

    return clients


def message_text(raw: Any) -> str:
    """Flatten an LLM response to plain text.

    LangChain's `.content` is a string for simple replies but a list of content
    blocks (dicts with a "text" key, or bare strings) for others. Passing that
    list straight to a regex raises TypeError, which every caller here catches
    as "the model failed" — so a formatting detail silently disabled reranking
    and verification. Normalise once, centrally. Anthropic responses arrive in
    the block form whenever thinking is on, so this is load-bearing there.
    """
    content = getattr(raw, "content", raw)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                # Skip thinking blocks: they are reasoning, not the answer.
                if block.get("type") in {"thinking", "redacted_thinking"}:
                    continue
                parts.append(str(block.get("text") or block.get("content") or ""))
        return "".join(parts)
    return str(content)


def lc_deps() -> dict[str, Any]:
    """Lazy-import LangChain dependencies to avoid hard import errors at module import time."""

    try:
        try:
            from langchain.text_splitter import RecursiveCharacterTextSplitter
        except ImportError:
            from langchain_text_splitters import RecursiveCharacterTextSplitter  # type: ignore

        from langchain_community.document_loaders import PyPDFLoader
        from langchain_community.vectorstores import Chroma
        from langchain_core.prompts import PromptTemplate
        from langchain_core.output_parsers import StrOutputParser
    except ImportError as exc:  # pragma: no cover - only when deps missing
        raise ImportError(
            "LangChain dependencies are required. Install langchain, langchain-community, langchain-text-splitters, langchain-core, langchain-google-genai, and pymupdf."
        ) from exc

    return {
        "RecursiveCharacterTextSplitter": RecursiveCharacterTextSplitter,
        "PyPDFLoader": PyPDFLoader,
        "Chroma": Chroma,
        "PromptTemplate": PromptTemplate,
        "StrOutputParser": StrOutputParser,
    }


__all__ = [
    "PROVIDERS",
    "chat_model",
    "chat_provider",
    "embed_provider",
    "embeddings",
    "get_api_key",
    "graphiti_clients",
    "key_for",
    "lc_deps",
    "message_text",
    "model_for",
]
