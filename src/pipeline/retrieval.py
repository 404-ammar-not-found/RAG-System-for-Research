from __future__ import annotations

"""Passage retrieval: dense + BM25, fused with RRF, optionally cross-encoded.

The previous version fused three scores that all derived from the same dense
hit list, so the "hybrid" and "multi-query" layers added ranking noise rather
than recall. BM25 here is a genuinely independent channel.
"""

import asyncio
import json
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

RRF_K = 60.0
# Never return fewer than this, even if the reranker only liked one passage —
# a single chunk is rarely enough to answer from.
MIN_PASSAGES = 3


@dataclass
class Passage:
    """A retrieved chunk plus its provenance."""

    id: str
    text: str
    metadata: dict[str, Any]
    score: float = 0.0
    channels: list[str] = field(default_factory=list)

    @property
    def source(self) -> str:
        return str(self.metadata.get("source", "unknown"))

    def as_match(self) -> dict[str, Any]:
        return {"document": self.text, "metadata": self.metadata, "score": self.score}


def debug(enabled: bool, message: str) -> None:
    if enabled:
        print(f"[DEBUG] {message}")


def _tokenize(text: str) -> list[str]:
    return re.sub(r"[^\w\s]", " ", text.lower()).split()


def body_of(text: str) -> str:
    """Strip the contextual header that ingestion prepends for embedding.

    The header ("<paper title> — <section>") is deliberately embedded, because it
    gives the vector its topical anchor. It must NOT be indexed lexically: it is
    identical for every chunk in a section, so it inflates title words in BM25
    and makes sibling chunks indistinguishable.
    """
    head, sep, rest = text.partition("\n\n")
    return rest if sep and len(head) < 200 else text


class Bm25Index:
    """In-memory BM25 over chunk bodies.

    ponytail: rebuilt from scratch on ingest and held in RAM. Fine to ~50k
    chunks (a few hundred MB of tokens); past that, move to a real inverted
    index (Tantivy/Lucene) or Chroma's server-side full-text search.
    """

    # Collapsed equations are symbol soup; indexing them only ever produces
    # false lexical matches.
    SKIP_TYPES = {"math"}

    def __init__(self) -> None:
        self._bm25: Any | None = None
        self._passages: list[Passage] = []

    def load_texts(self, rows: list[tuple[str, str, dict]]) -> int:
        """Build from (id, text, metadata) triples. Used by ingest and the eval harness."""
        from rank_bm25 import BM25Okapi

        self._passages = [
            Passage(id=str(i), text=t or "", metadata=dict(m or {})) for i, t, m in rows
        ]
        corpus = [
            []
            if p.metadata.get("content_type") in self.SKIP_TYPES
            else _tokenize(body_of(p.text))
            for p in self._passages
        ]
        self._bm25 = BM25Okapi(corpus) if any(corpus) else None
        return len(self._passages)

    def build(self, vectordb: Any) -> int:
        raw = vectordb._collection.get(include=["documents", "metadatas"])
        return self.load_texts(
            list(
                zip(
                    raw.get("ids") or [],
                    raw.get("documents") or [],
                    raw.get("metadatas") or [],
                )
            )
        )

    @property
    def size(self) -> int:
        return len(self._passages)

    def search(self, query: str, k: int) -> list[Passage]:
        if self._bm25 is None or not self._passages:
            return []
        scores = self._bm25.get_scores(_tokenize(query))
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        out: list[Passage] = []
        for i in ranked[:k]:
            if scores[i] <= 0:
                break
            p = self._passages[i]
            out.append(Passage(id=p.id, text=p.text, metadata=dict(p.metadata)))
        return out


def dense_search(vectordb: Any, query: str, k: int) -> list[Passage]:
    """One query, unmangled. No regex 'variants'."""
    try:
        scored = vectordb.similarity_search_with_relevance_scores(query, k=k)
        docs = [d for d, _ in scored]
    except Exception:
        docs = vectordb.similarity_search(query, k=k)
    out: list[Passage] = []
    for doc in docs:
        meta = dict(getattr(doc, "metadata", {}) or {})
        out.append(
            Passage(
                id=str(meta.get("id", "")),
                text=getattr(doc, "page_content", ""),
                metadata=meta,
            )
        )
    return out


def rrf_fuse(*ranked_lists: tuple[str, list[Passage]], k: float = RRF_K) -> list[Passage]:
    """Reciprocal rank fusion over independently ranked channels."""
    merged: dict[str, Passage] = {}
    scores: dict[str, float] = {}

    for channel, passages in ranked_lists:
        for rank, passage in enumerate(passages, start=1):
            key = passage.id or passage.text[:200]
            if key not in merged:
                merged[key] = passage
                scores[key] = 0.0
            scores[key] += 1.0 / (k + rank)
            if channel not in merged[key].channels:
                merged[key].channels.append(channel)

    for key, passage in merged.items():
        passage.score = scores[key]
    return sorted(merged.values(), key=lambda p: p.score, reverse=True)


RERANK_PROMPT = """Rank these numbered passages by how well they help answer the question.

Question: {query}

{passages}

Reply with ONLY a JSON array of passage numbers, most relevant first, best {limit} \
only. Example: [3, 1, 7]"""

# Each passage gets truncated to this before being shown to the reranker. The
# lead of a chunk carries the contextual header plus the topic sentence, which
# is enough to judge relevance, and it keeps the prompt affordable.
RERANK_SNIPPET = 500


async def rerank(llm: Any, query: str, passages: list[Passage], limit: int) -> list[Passage]:
    """Listwise rerank in ONE LLM call. Falls back to RRF order on any failure.

    Graphiti's GeminiRerankerClient scores pointwise — one API call per passage,
    so 25 candidates cost 25 calls per question. One listwise call does the same
    job, and lets the model compare candidates against each other rather than
    scoring each in isolation.
    """
    if len(passages) <= 1:
        return passages[:limit]

    # body_of, not p.text: every chunk in a section carries the same header, so
    # including it wastes rerank tokens and blurs the comparison.
    listing = "\n\n".join(
        f"[{i}] {body_of(p.text)[:RERANK_SNIPPET]}" for i, p in enumerate(passages, start=1)
    )
    prompt = RERANK_PROMPT.format(query=query, passages=listing, limit=limit)

    try:
        raw = await llm.ainvoke(prompt)
        text = getattr(raw, "content", None) or str(raw)
        match = re.search(r"\[[\d,\s]*\]", text)
        if match is None:
            raise ValueError(f"no JSON array in reranker reply: {text[:120]!r}")
        order = json.loads(match.group(0))
    except Exception as exc:  # a reranker outage must not fail the question
        print(f"[WARN] rerank failed ({str(exc)[:120]}); using RRF order")
        return passages[:limit]

    out: list[Passage] = []
    seen: set[int] = set()
    for rank, number in enumerate(order):
        idx = int(number) - 1
        if idx in seen or not (0 <= idx < len(passages)):
            continue
        seen.add(idx)
        passage = passages[idx]
        passage.score = 1.0 / (1 + rank)
        out.append(passage)
        if len(out) >= limit:
            break

    # Score floor: if the model ranked only a few, that is a signal there are
    # only a few relevant passages. Top up just enough to stay useful rather
    # than padding to `limit` with material the reranker rejected.
    floor = min(limit, max(MIN_PASSAGES, len(out)))
    for i, passage in enumerate(passages):
        if len(out) >= floor:
            break
        if i not in seen:
            out.append(passage)
    return out


def cap_per_source(passages: list[Passage], max_per_source: int) -> list[Passage]:
    """Keep rank order, but stop any one paper from crowding out the corpus.

    Observed without this: 6 of the top 8 passages for a dropout question all
    came from AlexNet. Harmless there, fatal for "which papers use attention?".
    """
    if max_per_source <= 0:
        return passages
    counts: Counter[str] = Counter()
    kept: list[Passage] = []
    overflow: list[Passage] = []
    for p in passages:
        if counts[p.source] < max_per_source:
            counts[p.source] += 1
            kept.append(p)
        else:
            overflow.append(p)
    # Overflow is not discarded, just demoted — a genuinely single-paper
    # question should still be answerable.
    return kept + overflow


def boost_by_facts(passages: list[Passage], facts: list[Any], weight: float) -> list[Passage]:
    """Nudge up passages whose text mentions an entity named in a retrieved fact.

    This is the one place the symbolic and neural channels can inform each
    other, and it is free — the facts were already fetched for the prompt.
    """
    if not facts or weight <= 0:
        return passages
    names: set[str] = set()
    for edge in facts:
        for token in re.findall(r"[A-Z][A-Za-z0-9.\-]{2,}", getattr(edge, "fact", "") or ""):
            names.add(token.lower())
    if not names:
        return passages

    for p in passages:
        low = body_of(p.text).lower()
        overlap = sum(1 for n in names if n in low)
        if overlap:
            p.score += weight * min(overlap, 3)
            if "graph" not in p.channels:
                p.channels.append("graph")
    return sorted(passages, key=lambda x: x.score, reverse=True)


def expand_neighbours(
    vectordb: Any, passages: list[Passage], max_chars: int
) -> list[Passage]:
    """Pull each hit's prev/next chunk so the LLM sees contiguous runs.

    Fetched by id, so this costs no embedding and no search — the reason it is
    affordable on a free tier. Chunks written before prev_id/next_id existed
    simply have no neighbours and pass through unchanged.
    """
    wanted: list[str] = []
    have = {p.id for p in passages}
    for p in passages:
        for key in ("prev_id", "next_id"):
            nid = p.metadata.get(key)
            if nid and nid not in have and nid not in wanted:
                wanted.append(str(nid))
    if not wanted:
        return passages

    budget = max_chars - sum(len(p.text) for p in passages)
    if budget <= 0:
        return passages

    try:
        got = vectordb._collection.get(ids=wanted, include=["documents", "metadatas"])
    except Exception as exc:
        print(f"[WARN] neighbour fetch failed ({str(exc)[:100]})")
        return passages

    extra: list[Passage] = []
    for nid, doc, meta in zip(
        got.get("ids") or [], got.get("documents") or [], got.get("metadatas") or []
    ):
        text = doc or ""
        if len(text) > budget:
            continue
        budget -= len(text)
        extra.append(
            Passage(id=str(nid), text=text, metadata=dict(meta or {}),
                    score=0.0, channels=["neighbour"])
        )

    # Order by document position so the LLM reads contiguous prose.
    merged = passages + extra
    merged.sort(
        key=lambda x: (str(x.metadata.get("source", "")), int(x.metadata.get("chunk_index", 0)))
    )
    return merged


EXPAND_PROMPT = """Rewrite this research question as {n} short, distinct search queries that \
would find the passage answering it. Use the vocabulary a paper would use, not the \
question's phrasing (e.g. "how does it avoid overfitting" -> "dropout regularization").

Question: {query}

Reply with ONLY a JSON array of strings."""


async def expand_query(llm: Any, query: str, n: int = 3) -> list[str]:
    """Generate sub-queries in one LLM call. Always includes the original.

    Off by default: on the free tier each sub-query needs its own query
    embedding, tripling per-question embed cost against the same daily bucket
    the corpus competes for.
    """
    try:
        raw = await llm.ainvoke(EXPAND_PROMPT.format(query=query, n=n))
        text = getattr(raw, "content", None) or str(raw)
        match = re.search(r"\[.*\]", text, re.S)
        subs = json.loads(match.group(0)) if match else []
    except Exception as exc:
        print(f"[WARN] query expansion failed ({str(exc)[:100]}); using original only")
        return [query]

    out = [query]
    for s in subs:
        if isinstance(s, str) and s.strip() and s.strip().lower() != query.lower():
            out.append(s.strip())
    return out[: n + 1]


async def retrieve_passages(
    vectordb: Any,
    bm25: Bm25Index,
    query: str,
    settings: Any,
    rerank_llm: Any | None = None,
    facts: list[Any] | None = None,
) -> list[Passage]:
    """Dense + BM25 concurrently, RRF-fused, diversified, reranked, expanded."""
    queries = [query]
    if settings.expand_queries and rerank_llm is not None:
        queries = await expand_query(rerank_llm, query)
        debug(settings.debug_retrieval, f"sub-queries: {queries}")

    jobs = [asyncio.to_thread(bm25.search, query, settings.candidate_k)]
    jobs += [
        asyncio.to_thread(dense_search, vectordb, q, settings.candidate_k) for q in queries
    ]
    lexical, *dense_lists = await asyncio.gather(*jobs)
    dense = dense_lists[0]
    debug(settings.debug_retrieval, f"dense={len(dense)} bm25={len(lexical)}")

    channels = [("bm25", lexical)] + [
        (f"dense{i or ''}", d) for i, d in enumerate(dense_lists)
    ]
    fused = rrf_fuse(*channels)
    fused = boost_by_facts(fused, facts or [], settings.graph_boost)
    fused = cap_per_source(fused, settings.max_per_source)
    debug(settings.debug_retrieval, f"fused unique candidates={len(fused)}")

    if rerank_llm is not None and settings.use_reranker:
        top = await rerank(rerank_llm, query, fused[: settings.candidate_k], settings.top_k)
    else:
        top = fused[: settings.top_k]

    for i, p in enumerate(top, start=1):
        debug(
            settings.debug_retrieval,
            f"#{i} score={p.score:.4f} channels={'+'.join(p.channels)} "
            f"{p.metadata.get('section', '?')} [{p.id}]",
        )

    if settings.neighbour_expansion:
        before = len(top)
        top = await asyncio.to_thread(
            expand_neighbours, vectordb, top, settings.max_context_chars
        )
        debug(settings.debug_retrieval, f"neighbour expansion: {before} -> {len(top)}")
    return top


def format_passages(passages: list[Passage]) -> str:
    """Citation-prefixed context blocks, in reading order after expansion."""
    blocks = []
    for p in passages:
        kind = p.metadata.get("content_type", "prose")
        tag = "" if kind == "prose" else f" [{kind}]"
        marker = " (context)" if p.channels == ["neighbour"] else ""
        blocks.append(
            f"[{p.id}]{tag}{marker} ({p.metadata.get('paper_title', '?')} — "
            f"{p.metadata.get('section', '?')})\n{body_of(p.text)}"
        )
    return "\n\n".join(blocks)


__all__ = [
    "Bm25Index",
    "Passage",
    "body_of",
    "boost_by_facts",
    "cap_per_source",
    "debug",
    "dense_search",
    "expand_neighbours",
    "format_passages",
    "rerank",
    "retrieve_passages",
    "rrf_fuse",
]
