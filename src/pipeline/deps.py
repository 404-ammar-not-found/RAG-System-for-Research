from __future__ import annotations

from typing import Any


def get_api_key() -> str:
    """Return the Gemini API key from env."""

    import os
    from dotenv import load_dotenv

    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("Set GEMINI_API_KEY (or GOOGLE_API_KEY) in your environment or .env file.")
    return api_key


def message_text(raw: Any) -> str:
    """Flatten an LLM response to plain text.

    LangChain's `.content` is a string for simple replies but a list of content
    blocks (dicts with a "text" key, or bare strings) for others. Passing that
    list straight to a regex raises TypeError, which every caller here catches
    as "the model failed" — so a formatting detail silently disabled reranking
    and verification. Normalise once, centrally.
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
        from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
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
        "GoogleGenerativeAIEmbeddings": GoogleGenerativeAIEmbeddings,
        "ChatGoogleGenerativeAI": ChatGoogleGenerativeAI,
        "Chroma": Chroma,
        "PromptTemplate": PromptTemplate,
        "StrOutputParser": StrOutputParser,
    }


__all__ = ["get_api_key", "lc_deps", "message_text"]
