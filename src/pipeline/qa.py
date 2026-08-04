from __future__ import annotations

"""Answer generation: graph facts + passages, one LLM call."""

import asyncio
import re
from typing import Any

from .deps import get_api_key, lc_deps
from .graph_store import GraphStore, format_facts
from .stance import describe_split, group_by_stance, label_stances
from .verification import (
    CITATION_RE,
    extract_claims,
    summarise,
    to_payload,
    verify_claims,
)
from .retrieval import Bm25Index, Passage, debug, format_passages, retrieve_passages
from .settings import PipelineSettings

PROMPT = """You are a research assistant answering questions about a corpus of academic papers.

You have two kinds of evidence.

KNOWLEDGE GRAPH FACTS — relations extracted across the whole corpus. Use these to \
reason about how papers, models, datasets and methods relate to each other, and to \
notice connections that span more than one paper. Facts are LLM-derived summaries, so \
do not quote them as the paper's own words, and where a fact and a passage disagree, \
the passage wins.

{facts}

{split}PASSAGES — verbatim excerpts, in reading order. Ground every specific claim (numbers, \
quotes, architectural details) in these.

Cite like this, with the passage id AND a short verbatim quote copied exactly from \
that passage:

    [{example_id} "the exact words you are relying on"]

The quote is checked against the passage automatically, so it must appear there \
word for word. If you cannot find words that support a claim, do not make the claim.

Reading the passages:
- Consecutive passages from the same paper and section are contiguous text; read them \
together rather than as separate findings. Ones marked (context) were pulled in as \
neighbours and may not mention the question directly.
- A passage marked [table] or [figure] is a caption plus a flattened grid. Column \
alignment is lost, so a number's row and column heading are inferred, not certain — \
attribute such numbers only when the caption or nearby text makes the mapping clear.
- A passage marked [math] is a display equation flattened onto one line.

{passages}

{cautions}Question: {question}

Answer the question directly. Cite passage ids for specific claims. If the evidence \
does not cover the question, say so plainly rather than speculating."""

def _strip_unknown_citations(answer: str, valid_ids: set[str]) -> tuple[str, list[str]]:
    """Drop citations to ids that were never retrieved; keep the rest verbatim.

    Uses the same pattern as the verifier so the quote-bearing form
    `[id "quote"]` is recognised — matching only bare `[id]` here would quietly
    stop policing invented ids the moment quotes were introduced.
    """
    used: list[str] = []

    def sub(match: re.Match[str]) -> str:
        cid = match.group(1)
        if cid in valid_ids:
            if cid not in used:
                used.append(cid)
            return match.group(0)
        return ""

    cleaned = CITATION_RE.sub(sub, answer)
    return re.sub(r"[ \t]{2,}", " ", cleaned), used


class QaEngine:
    """Holds the warm clients: vector store, BM25 index, graph, LLM."""

    def __init__(self, settings: PipelineSettings, vectordb: Any, graph: GraphStore | None) -> None:
        self.settings = settings
        self.vectordb = vectordb
        self.graph = graph
        self.bm25 = Bm25Index()
        deps = lc_deps()
        api_key = get_api_key()
        self.llm = deps["ChatGoogleGenerativeAI"](
            model=settings.text_llm_model,
            google_api_key=api_key,
            temperature=0.2,
        )
        self._chain = deps["StrOutputParser"]()

        # Reranking runs on the cheap model and is independent of whether
        # FalkorDB is reachable. temperature=0 because it is an ordering task.
        self.rerank_llm = deps["ChatGoogleGenerativeAI"](
            model=settings.graph_llm_model,
            google_api_key=api_key,
            temperature=0,
        )

    def rebuild_bm25(self) -> int:
        n = self.bm25.build(self.vectordb)
        debug(self.settings.debug_retrieval, f"BM25 index built over {n} chunks")
        return n

    @staticmethod
    def _warnings(passages: list[Passage]) -> list[dict[str, Any]]:
        """Retraction and provenance cautions for the papers actually retrieved."""
        seen: dict[str, dict[str, Any]] = {}
        for p in passages:
            meta = p.metadata or {}
            title = meta.get("paper_title") or p.source
            if meta.get("is_retracted") and title not in seen:
                seen[title] = {
                    "level": "retracted",
                    "paper": title,
                    "message": "This paper is recorded as retracted. Do not cite it as current.",
                }
            elif not meta.get("doi") and not meta.get("arxiv_id") and title not in seen:
                seen[title] = {
                    "level": "unverified",
                    "paper": title,
                    "message": "No DOI or arXiv id, so retraction status is unknown.",
                }
        return list(seen.values())

    def _cautions(self, passages: list[Passage]) -> str:
        lines = [f"- {w['paper']}: {w['message']}" for w in self._warnings(passages)]
        return "CAUTIONS\n" + "\n".join(lines) + "\n\n" if lines else ""

    async def _facts(self, question: str) -> list[Any]:
        if self.graph is None:
            return []
        return await self.graph.search_facts(question, self.settings.fact_k)

    async def retrieve(self, question: str, facts: list[Any] | None = None) -> list[Passage]:
        """Retrieval only, no answer generation. Used by the eval harness."""
        return await retrieve_passages(
            self.vectordb,
            self.bm25,
            question,
            self.settings,
            self.rerank_llm,
            facts=facts,
        )

    async def answer(self, question: str) -> dict[str, Any]:
        settings = self.settings

        # Facts first: they are a rerank signal for the passages, so the two
        # channels stop ignoring each other.
        graph_facts = await self._facts(question)
        passages = await self.retrieve(question, facts=graph_facts)
        debug(settings.debug_retrieval, f"passages={len(passages)} facts={len(graph_facts)}")

        # Cluster by stance *before* synthesis. Doing it after means the model
        # has already blended the conflict into one confident paragraph.
        labels: dict[str, Any] = {}
        split = ""
        if settings.cluster_stance and passages:
            labels = await label_stances(self.rerank_llm, question, passages)
            split = describe_split(group_by_stance(passages, labels), labels)
            if split:
                debug(settings.debug_retrieval, "disagreement detected in retrieved evidence")

        prompt = PROMPT.format(
            facts=format_facts(graph_facts) or "(none retrieved)",
            passages=format_passages(passages) or "(none retrieved)",
            question=question,
            example_id=passages[0].id if passages else "paper-abc12345-chunk-0",
            split=f"{split}\n\n" if split else "",
            cautions=self._cautions(passages),
        )
        raw = await self.llm.ainvoke(prompt)
        answer = self._chain.invoke(raw)

        valid_ids = {p.id for p in passages if p.id}
        answer, cited = _strip_unknown_citations(answer, valid_ids)

        # Verify what was written against the spans it cited.
        chunk_index = {
            p.id: {"text": p.text, "doc_start": int((p.metadata or {}).get("doc_start", 0))}
            for p in passages
        }
        claims = extract_claims(answer, chunk_index)
        if settings.verify_claims:
            claims = await verify_claims(self.rerank_llm, claims)
        report = summarise(claims)
        if report["flagged"]:
            debug(
                settings.debug_retrieval,
                f"{len(report['flagged'])} claim(s) flagged: {report['counts']}",
            )

        entity_uuids: list[str] = []
        for edge in graph_facts:
            for uuid in (edge.source_node_uuid, edge.target_node_uuid):
                if uuid not in entity_uuids:
                    entity_uuids.append(uuid)

        return {
            "answer": answer.strip(),
            "claims": to_payload(claims),
            "verification": report,
            "stances": labels,
            "warnings": self._warnings(passages),
            "matches": [p.as_match() for p in passages],
            "facts": [
                {
                    "name": getattr(e, "name", ""),
                    "fact": e.fact,
                    "source": e.source_node_uuid,
                    "target": e.target_node_uuid,
                }
                for e in graph_facts
            ],
            # Both layers, so the visualizer can highlight whichever is showing.
            "usedNodeIds": (cited or [p.id for p in passages if p.id]) + entity_uuids,
        }


__all__ = ["QaEngine", "Passage"]
