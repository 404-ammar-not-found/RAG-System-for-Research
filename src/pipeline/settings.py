from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


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
    # keeping them drops recall@8 from 0.273 to 0.227. They are also what
    # `Cites` edges are built from, so flip this on when graph quota allows.
    index_references: bool = False

    # Retrieval
    candidate_k: int = 25  # per channel, before fusion
    top_k: int = 8  # passages sent to the LLM
    fact_k: int = 10  # graph facts sent to the LLM
    use_reranker: bool = True
    debug_retrieval: bool = True

    # Stop one paper monopolising the results. Measured before this existed:
    # 6 of the top 8 passages for one question came from a single paper.
    max_per_source: int = 4
    # RRF bonus for passages naming an entity from a retrieved graph fact.
    graph_boost: float = 0.01
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

    # Models.
    # NB: gemini-3-flash-preview has a free-tier cap of 20 generate requests
    # PER DAY, which cannot support graph extraction (~5 LLM calls per episode,
    # ~95 episodes for four papers). Answers use a full flash model; the
    # high-volume paths — Graphiti extraction and passage reranking, both
    # structured/classification work — use a lite model.
    # Embedding quota is metered PER MODEL (1000/day on the free tier), so
    # giving the passage store and the graph different embedding models buys
    # two independent daily budgets instead of one shared. They index separate
    # stores, so the vectors never have to be comparable across the two.
    embed_model: str = "models/gemini-embedding-2"
    graph_embed_model: str = "models/gemini-embedding-2-preview"
    embed_dimensions: int | None = None
    text_llm_model: str = "gemini-flash-latest"
    graph_llm_model: str = "gemini-flash-lite-latest"

    # Graphiti / FalkorDB.
    # One group_id for the whole corpus: group_id partitions the graph, and
    # cross-paper entity resolution ("Transformer" in two papers = one node) is
    # the entire point of building it.
    graphiti_group_id: str = "papers"
    # Passed to Graphiti as max_coroutines. Do NOT rely on the SEMAPHORE_LIMIT
    # env var instead: graphiti_core.helpers reads it at *import* time, which
    # happens before load_dotenv(), so a value in .env is silently ignored and
    # you get the default of 20 concurrent ops straight into a 100/min quota.
    max_coroutines: int = 2
    # 6380, not redis' default 6379 — this machine already runs a plain
    # redis-server there, and connecting to it fails with "unknown command
    # GRAPH.QUERY" rather than anything that looks like a port clash.
    falkordb_host: str = os.getenv("FALKORDB_HOST", "localhost")
    falkordb_port: int = int(os.getenv("FALKORDB_PORT", "6380"))
    falkordb_database: str = "papers"


__all__ = ["PipelineSettings"]
