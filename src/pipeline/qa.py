from __future__ import annotations

"""Answer generation: retrieved passages, one streaming LLM call."""

import json
import re
from collections.abc import AsyncIterator
from typing import Any

from .deps import chat_model, lc_deps, message_text
from .stance import describe_split, group_by_stance, label_stances
from .verification import (
    CITATION_RE,
    extract_claims,
    summarise,
    to_payload,
    verify_claims,
)
from .relations import format_relations
from .retrieval import Bm25Index, Passage, debug, format_passages, retrieve_passages
from .settings import PipelineSettings

PROMPT = """You are a research assistant answering questions about a corpus of academic papers.

{history}{split}{relations}PASSAGES — verbatim excerpts, in reading order. Ground every specific claim (numbers, \
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

The passages often use standard concepts (softmax, dropout, convolution) without \
defining them. When the question asks what such a concept is and no passage defines \
it, explain it plainly from your own knowledge, without citations, and say in one \
short phrase that the explanation is general background rather than from these \
papers. Then connect it to how these papers use the concept, citing passages as \
usual. Never attach a citation to a claim its passage does not support.

{passages}

{cautions}Question: {question}

Answer the question directly. Cite passage ids for specific claims. Write any \
mathematics in LaTeX ($...$ inline, $$...$$ display) — but citation quotes must stay \
verbatim from the passage, never rewritten as LaTeX. If the evidence does not cover \
the question, say so plainly rather than speculating."""

CONDENSE_PROMPT = """Rewrite the user's latest message as ONE standalone search query, \
resolving any pronouns or references using the conversation. Reply with only the query.

{history}

Latest message: {question}"""

FOLLOWUP_PROMPT = """A user asked a question about a corpus of academic papers and got \
this answer. Suggest 3 short, distinct follow-up questions likely answerable from the \
same papers. Reply with only a JSON array of 3 strings.

Question: {question}

Answer: {answer}"""


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


def _format_history(history: list[dict[str, str]], max_turns: int = 6) -> str:
    """The last few turns as a prompt block, truncated per message."""
    lines = []
    for msg in history[-max_turns:]:
        role = "User" if msg.get("role") == "user" else "Assistant"
        text = (msg.get("content") or "").strip()
        if len(text) > 1500:
            text = text[:1500] + " …"
        lines.append(f"{role}: {text}")
    return "\n".join(lines)


async def _condense(llm: Any, history: list[dict[str, str]], question: str) -> str:
    """One cheap-model call turning a follow-up into a standalone retrieval query."""
    try:
        raw = await llm.ainvoke(
            CONDENSE_PROMPT.format(history=_format_history(history), question=question)
        )
        text = message_text(raw).strip().strip('"')
        return text.splitlines()[0].strip() if text else question
    except Exception as exc:
        print(f"[WARN] condense failed ({str(exc)[:100]}); using raw question")
        return question


async def _suggest_followups(llm: Any, question: str, answer: str) -> list[str]:
    """One cheap-model call proposing follow-ups. Degrades to none."""
    try:
        raw = await llm.ainvoke(
            FOLLOWUP_PROMPT.format(question=question, answer=answer[:3000])
        )
        match = re.search(r"\[.*\]", message_text(raw), re.S)
        items = json.loads(match.group(0)) if match else []
        return [s.strip() for s in items if isinstance(s, str) and s.strip()][:3]
    except Exception as exc:
        print(f"[WARN] followup suggestions failed ({str(exc)[:100]})")
        return []


class QaEngine:
    """Holds the warm clients: vector store, BM25 index, LLM."""

    relations: Any | None = None  # RelationGraph, once one has been built

    def __init__(self, settings: PipelineSettings, vectordb: Any) -> None:
        self.settings = settings
        self.vectordb = vectordb
        self.bm25 = Bm25Index()
        deps = lc_deps()
        self.llm = chat_model(settings, "text", temperature=0.2)
        self._chain = deps["StrOutputParser"]()

        # Reranking, stance labelling and verification run on the cheap model.
        # temperature=0 because they are ordering/classification tasks.
        self.rerank_llm = chat_model(settings, "graph", temperature=0)

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

    async def retrieve(self, question: str) -> list[Passage]:
        """Retrieval only, no answer generation. Used by the eval harness."""
        return await retrieve_passages(
            self.vectordb,
            self.bm25,
            question,
            self.settings,
            self.rerank_llm,
            self.relations,
        )

    async def answer_stream(
        self, question: str, history: list[dict[str, str]] | None = None
    ) -> AsyncIterator[dict[str, Any]]:
        """Yield {"type": "token"} events while generating, then one {"type": "done"}.

        The streamed tokens are the raw answer; the done event carries the
        cleaned answer (unknown citations stripped) plus verification, stances
        and passages, because those are post-hoc by design.
        """
        settings = self.settings
        history = history or []

        yield {"type": "stage", "name": "retrieving"}
        search_query = question
        if history:
            search_query = await _condense(self.rerank_llm, history, question)
            debug(settings.debug_retrieval, f"condensed query: {search_query}")

        passages = await self.retrieve(search_query)
        debug(settings.debug_retrieval, f"passages={len(passages)}")

        # Cluster by stance *before* synthesis. Doing it after means the model
        # has already blended the conflict into one confident paragraph.
        labels: dict[str, Any] = {}
        split = ""
        if settings.cluster_stance and passages:
            labels = await label_stances(self.rerank_llm, question, passages)
            split = describe_split(group_by_stance(passages, labels), labels)
            if split:
                debug(settings.debug_retrieval, "disagreement detected in retrieved evidence")

        facts = (
            self.relations.facts(search_query)
            if self.relations is not None and settings.relation_graph
            else []
        )

        hist_block = _format_history(history)
        prompt = PROMPT.format(
            history=f"CONVERSATION SO FAR\n{hist_block}\n\n" if hist_block else "",
            passages=format_passages(passages) or "(none retrieved)",
            question=question,
            example_id=passages[0].id if passages else "paper-abc12345-chunk-0",
            split=f"{split}\n\n" if split else "",
            cautions=self._cautions(passages),
            relations=format_relations(facts),
        )

        yield {"type": "stage", "name": "writing"}
        parts: list[str] = []
        async for chunk in self.llm.astream(prompt):
            text = message_text(chunk)
            if text:
                parts.append(text)
                yield {"type": "token", "text": text}
        answer = "".join(parts)

        yield {"type": "stage", "name": "verifying"}
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

        suggestions: list[str] = []
        if settings.suggest_followups and answer.strip():
            suggestions = await _suggest_followups(self.rerank_llm, question, answer)

        yield {
            "type": "done",
            "answer": answer.strip(),
            "suggestions": suggestions,
            "claims": to_payload(claims),
            "verification": report,
            "stances": labels,
            "warnings": self._warnings(passages),
            "relations": facts,
            "matches": [p.as_match() for p in passages],
            "usedNodeIds": cited or [p.id for p in passages if p.id],
        }

    async def answer(self, question: str) -> dict[str, Any]:
        """Drain the stream and return the final payload. CLI/eval entry point."""
        payload: dict[str, Any] = {}
        async for event in self.answer_stream(question):
            if event["type"] == "done":
                payload = {k: v for k, v in event.items() if k != "type"}
        return payload


__all__ = ["QaEngine", "Passage"]
