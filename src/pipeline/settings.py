from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

# The env-backed defaults below are read when this module is imported.
load_dotenv()


@dataclass
class PipelineSettings:
    """Configuration for the RAG pipeline."""

    # Storage
    pdf_directory: Path = Path("data")
    chroma_path: str = "chroma_db"
    # Bump the suffix whenever chunking changes, instead of deleting the old
    # collection first. A re-ingest then builds *alongside* the live index and
    # the old one stays queryable if it fails part-way. Learned the hard way:
    # wiping first and then hitting the daily embed cap leaves you with nothing.
    # Drop stale collections manually once the new one is verified.
    collection_name: str = "papers-v2"

    # Chunking (sections are split into these for the passage store)
    chunk_size: int = 1200
    chunk_overlap: int = 150
    max_pages: int | None = 200
    # Bibliographies are 14% of embed spend and pure retrieval noise — nobody
    # asks what is in one, but they compete for every top-k slot. Measured:
    # keeping them drops recall@8 from 0.273 to 0.227.
    index_references: bool = False

    # Retrieval
    candidate_k: int = 25  # per channel, before fusion
    top_k: int = 8  # passages sent to the LLM
    use_reranker: bool = True
    debug_retrieval: bool = True

    # Stop one paper monopolising the results. Measured before this existed:
    # 6 of the top 8 passages for one question came from a single paper.
    max_per_source: int = 4
    # Add a retrieval channel and prompt context from the Graphify relation map.
    # Costs nothing per question; the LLM spend happens once per paper at ingest.
    relation_graph: bool = True
    relations_dir: Path = Path("cache/relations")
    # Pull prev/next chunks of the top hits so the LLM reads contiguous text.
    # Free: fetched by id, no embedding, no search.
    neighbour_expansion: bool = True
    max_context_chars: int = 24_000
    # One LLM call to generate sub-queries. OFF: on the free tier each sub-query
    # needs its own query embedding, tripling per-question embed cost against
    # the same daily bucket the corpus competes for. Turn on only if the eval
    # harness shows it winning.
    expand_queries: bool = False
    # Label each passage's stance before synthesis so conflicting findings are
    # reported as a split rather than averaged into one confident paragraph.
    cluster_stance: bool = True
    # Check every generated sentence against the span it cited.
    verify_claims: bool = True
    # One cheap-model call after each answer proposing follow-up questions.
    suggest_followups: bool = True

    # Models. Empty means "whatever the configured provider defaults to" —
    # the per-provider table lives in `deps.PROVIDERS`, and `deps.model_for`
    # resolves a slot. Set one here (or via the matching env var) to override.
    #
    # Answers use the full model; the high-volume classification paths —
    # reranking, stance labelling, claim verification — use the cheap model
    # (the `graph` slot, named for the graph-extraction work it once also did).
    embed_model: str = os.getenv("EMBED_MODEL", "")
    embed_dimensions: int | None = None
    text_llm_model: str = os.getenv("TEXT_LLM_MODEL", "")
    graph_llm_model: str = os.getenv("GRAPH_LLM_MODEL", "")


__all__ = ["PipelineSettings"]
