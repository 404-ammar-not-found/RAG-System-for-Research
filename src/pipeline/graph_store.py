from __future__ import annotations

"""Graphiti knowledge-graph layer over FalkorDB.

This is the symbolic half of the system: LLM-extracted typed entities and
relations across the whole corpus, queryable by hybrid search (BM25 + cosine +
graph traversal). Chroma keeps the verbatim passages; this keeps the structure.
"""

import asyncio
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, Field

from .deps import graphiti_clients
from .parsing import Section
from .settings import PipelineSettings

# ---------------------------------------------------------------------------
# Ontology. Deliberately small: too many types degrades extraction quality
# because the model has to disambiguate between near-identical categories.
# ---------------------------------------------------------------------------


class Paper(BaseModel):
    """A research paper or publication."""

    year: Optional[int] = Field(None, description="Publication year")
    venue: Optional[str] = Field(None, description="Conference or journal, e.g. NeurIPS, ICLR")


class Method(BaseModel):
    """A technique, algorithm, or training procedure (e.g. dropout, self-attention)."""

    category: Optional[str] = Field(None, description="Kind of method, e.g. regularization")


class Model(BaseModel):
    """A named model or architecture (e.g. Transformer, AlexNet, ViT-L/16)."""

    parameters: Optional[str] = Field(None, description="Parameter count if stated")
    architecture_type: Optional[str] = Field(None, description="e.g. CNN, transformer, RNN")


class Dataset(BaseModel):
    """A named dataset or benchmark (e.g. ImageNet, WMT 2014 EN-DE, S&P 500)."""

    domain: Optional[str] = Field(None, description="e.g. vision, NLP, finance")


class Metric(BaseModel):
    """An evaluation measure (e.g. BLEU, top-5 error, F1, accuracy)."""


class Task(BaseModel):
    """A problem being solved (e.g. machine translation, image classification)."""


class Author(BaseModel):
    """A researcher credited on a paper."""


class Institution(BaseModel):
    """A university, lab, or company affiliation."""


ENTITY_TYPES: dict[str, type[BaseModel]] = {
    "Paper": Paper,
    "Method": Method,
    "Model": Model,
    "Dataset": Dataset,
    "Metric": Metric,
    "Task": Task,
    "Author": Author,
    "Institution": Institution,
}


class Proposes(BaseModel):
    """A paper introduces a new model or method."""


class Uses(BaseModel):
    """One thing employs or builds on another."""


class EvaluatedOn(BaseModel):
    """A model or method is evaluated on a dataset or task."""


class Outperforms(BaseModel):
    """One approach beats another on some benchmark."""

    margin: Optional[str] = Field(None, description="Reported improvement, e.g. '2.0 BLEU'")


class AchievesScore(BaseModel):
    """A model attains a specific score on a metric."""

    value: Optional[str] = Field(None, description="The reported value, e.g. '28.4 BLEU'")


class AuthoredBy(BaseModel):
    """A paper is written by an author."""


class Cites(BaseModel):
    """A paper references prior work."""


EDGE_TYPES: dict[str, type[BaseModel]] = {
    "Proposes": Proposes,
    "Uses": Uses,
    "EvaluatedOn": EvaluatedOn,
    "Outperforms": Outperforms,
    "AchievesScore": AchievesScore,
    "AuthoredBy": AuthoredBy,
    "Cites": Cites,
}

EDGE_TYPE_MAP: dict[tuple[str, str], list[str]] = {
    ("Paper", "Model"): ["Proposes", "Uses"],
    ("Paper", "Method"): ["Proposes", "Uses"],
    ("Paper", "Dataset"): ["Uses"],
    ("Paper", "Author"): ["AuthoredBy"],
    ("Paper", "Paper"): ["Cites"],
    ("Author", "Institution"): ["Uses"],
    ("Model", "Dataset"): ["EvaluatedOn"],
    ("Model", "Task"): ["EvaluatedOn"],
    ("Model", "Method"): ["Uses"],
    ("Model", "Metric"): ["AchievesScore"],
    ("Model", "Model"): ["Outperforms"],
    ("Method", "Dataset"): ["EvaluatedOn"],
    ("Method", "Method"): ["Uses"],
    # Wildcard fallback so the extractor is never stuck without a valid edge.
    ("Entity", "Entity"): ["Uses"],
}

EXTRACTION_INSTRUCTIONS = (
    "This text is from an academic research paper. Extract concrete, named research "
    "objects: models and architectures, datasets and benchmarks, methods and techniques, "
    "evaluation metrics, tasks, authors, and institutions. "
    "Use the canonical name an author would recognise (e.g. 'Transformer', 'ImageNet', "
    "'BLEU', 'dropout'), never a sentence fragment or a generic phrase like "
    "'the proposed model' or 'our approach' — resolve those to the actual name when the "
    "text makes it clear, and skip them otherwise. Ignore citation markers, figure and "
    "table numbers, and equation labels."
)


class GraphStore:
    """Thin wrapper over Graphiti. Must be constructed inside a running event loop."""

    def __init__(self, settings: PipelineSettings) -> None:
        from graphiti_core import Graphiti
        from graphiti_core.driver.falkordb_driver import FalkorDriver

        self.settings = settings

        driver = FalkorDriver(
            host=settings.falkordb_host,
            port=settings.falkordb_port,
            username=os.getenv("FALKORDB_USERNAME") or None,
            password=os.getenv("FALKORDB_PASSWORD") or None,
            database=settings.falkordb_database,
        )
        self.client = Graphiti(
            graph_driver=driver,
            **graphiti_clients(settings),
            max_coroutines=settings.max_coroutines,
        )

    @property
    def group_id(self) -> str:
        return self.settings.graphiti_group_id

    async def initialize(self) -> None:
        await self.client.build_indices_and_constraints()

    async def close(self) -> None:
        await self.client.close()

    async def _add_one(self, section: Section) -> None:
        """Add a single episode, backing off when the API rate-limits us.

        Each episode costs several LLM calls and a handful of embeddings, and
        the Gemini free tier allows only 100 embeds/minute — so 429s are the
        normal case during a bulk ingest, not an error.
        """
        from graphiti_core.nodes import EpisodeType

        mtime = datetime.fromtimestamp(Path(section.source).stat().st_mtime, tz=timezone.utc)
        delay = 15.0
        for attempt in range(5):
            try:
                await self.client.add_episode(
                    name=section.episode_name,
                    episode_body=section.text,
                    source=EpisodeType.text,
                    source_description=f"{section.paper_title} ({Path(section.source).name})",
                    reference_time=mtime,
                    group_id=self.group_id,
                    entity_types=ENTITY_TYPES,
                    edge_types=EDGE_TYPES,
                    edge_type_map=EDGE_TYPE_MAP,
                    custom_extraction_instructions=EXTRACTION_INSTRUCTIONS,
                )
                return
            except Exception as exc:
                text = str(exc)
                rate_limited = "429" in text or "RESOURCE_EXHAUSTED" in text or "Rate limit" in text
                if not rate_limited or attempt == 4:
                    raise
                print(f"  rate limited, waiting {delay:.0f}s ({section.episode_name})")
                await asyncio.sleep(delay)
                delay = min(delay * 2, 120.0)

    async def ingest_sections(self, sections: list[Section]) -> int:
        """Add one episode per section. Entities resolve across the whole corpus."""
        added = 0
        for i, section in enumerate(sections, start=1):
            try:
                await self._add_one(section)
                added += 1
            except Exception as exc:
                # One bad section must not abort a 95-episode ingest.
                print(f"[WARN] episode failed: {section.episode_name}: {str(exc)[:160]}")
            if i % 5 == 0:
                print(f"  graph: {i}/{len(sections)} episodes ({added} added)")
        return added

    async def search_facts(self, query: str, k: int) -> list[Any]:
        """Hybrid search over relations. Returns EntityEdge objects."""
        try:
            return await self.client.search(
                query, group_ids=[self.group_id], num_results=k
            )
        except Exception as exc:
            print(f"[WARN] graph search failed: {exc}")
            return []

    async def dump(self) -> tuple[list[Any], list[Any]]:
        """All entities and relations, for the visualizer export.

        Graphiti's bulk getters *raise* on an empty result instead of returning
        an empty list, so an un-ingested graph would otherwise crash the export.
        """
        from graphiti_core.edges import EntityEdge
        from graphiti_core.errors import GroupsEdgesNotFoundError, GroupsNodesNotFoundError
        from graphiti_core.nodes import EntityNode

        try:
            nodes = await EntityNode.get_by_group_ids(self.client.driver, [self.group_id])
        except GroupsNodesNotFoundError:
            nodes = []
        try:
            edges = await EntityEdge.get_by_group_ids(self.client.driver, [self.group_id])
        except GroupsEdgesNotFoundError:
            edges = []
        return nodes, edges

    async def episode_map(self) -> dict[str, list[str]]:
        """episode_name -> entity uuids, for provenance links.

        Read from the `MENTIONS` edges Graphiti writes from each episode to
        every entity it extracted. The obvious-looking alternative — walking an
        episode's `entity_edges` — only finds entities that ended up on a
        *relation*, which for this corpus was under a fifth of them: the rest
        were extracted, stored, and then had no visible provenance at all.
        """
        query = (
            "MATCH (ep:Episodic)-[:MENTIONS]->(n:Entity) "
            "WHERE ep.group_id = $group_id "
            "RETURN ep.name AS episode, collect(n.uuid) AS uuids"
        )
        try:
            records, _, _ = await self.client.driver.execute_query(
                query, group_id=self.group_id
            )
        except Exception as exc:  # a missing graph is not an error worth raising
            print(f"[WARN] episode map unavailable ({str(exc)[:80]})")
            return {}

        out: dict[str, list[str]] = {}
        for record in records or []:
            episode = record["episode"] if "episode" in record else record[0]
            uuids = record["uuids"] if "uuids" in record else record[1]
            if episode:
                out[str(episode)] = sorted({str(u) for u in uuids or []})
        return out


def format_facts(facts: list[Any]) -> str:
    """Numbered fact block for the prompt."""
    lines = []
    for i, edge in enumerate(facts, start=1):
        when = ""
        valid_at = getattr(edge, "valid_at", None)
        if valid_at:
            when = f" (as of {valid_at:%Y-%m-%d})"
        lines.append(f"{i}. [{getattr(edge, 'name', 'RELATED')}] {edge.fact}{when}")
    return "\n".join(lines)


__all__ = [
    "EDGE_TYPES",
    "EDGE_TYPE_MAP",
    "ENTITY_TYPES",
    "GraphStore",
    "format_facts",
]
