from __future__ import annotations

"""Check that each generated sentence is actually supported by the span cited.

Two layers, cheapest first:

1. Quote matching (deterministic, free). The model must emit a short verbatim
   quote with every citation. The quote is located inside the cited chunk to
   give exact character offsets. A quote that is not in the chunk is fabricated
   text, and that is caught with a string search rather than another model.

2. Entailment (one batched LLM call). A real quote can still fail to support
   the claim built on it. Each (sentence, span) pair is judged supported,
   partial or unsupported.

On offsets: a PDF has no stable character stream — two-column layout, ligature
folding, dehyphenation and reflow all move positions — so "offset into the
source" is only meaningful against a specific extraction. Offsets here are into
the normalised chunk text that was indexed, plus the chunk's own offset within
its paper, and that text is stored, so every offset stays dereferenceable.
"""

import json
import re
from dataclasses import dataclass, field
from typing import Any

from .deps import message_text

# [chunk-id "verbatim quote"] — the quote is what makes the citation checkable.
CITATION_RE = re.compile(r"\[([A-Za-z0-9._-]+chunk-\d+)(?:\s+\"([^\"]{4,400})\")?\]")
SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z(])")

SUPPORTED = "supported"
PARTIAL = "partial"
UNSUPPORTED = "unsupported"
FABRICATED = "fabricated"
UNCHECKED = "unchecked"


@dataclass
class Span:
    """A located quote inside a chunk."""

    chunk_id: str
    quote: str
    start: int          # offset into the chunk's indexed text
    end: int
    doc_start: int      # offset into the paper's normalised text
    doc_end: int
    exact: bool         # False when matched after whitespace normalisation


@dataclass
class Claim:
    """One sentence of the answer plus whatever backs it."""

    text: str
    spans: list[Span] = field(default_factory=list)
    verdict: str = UNCHECKED
    reason: str = ""

    @property
    def cited(self) -> bool:
        return bool(self.spans)


def _normalise(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def locate_quote(quote: str, chunk_text: str) -> tuple[int, int, bool] | None:
    """Find a quote in a chunk, returning (start, end, exact) or None.

    Tries an exact search first, then a whitespace-insensitive one, because a
    model re-typing a quote across a line break is a formatting difference, not
    a fabrication.
    """
    if not quote or not chunk_text:
        return None

    index = chunk_text.find(quote)
    if index != -1:
        return index, index + len(quote), True

    # Whitespace-insensitive: build a regex that lets any run of whitespace
    # match any other, then map back to real offsets.
    pattern = r"\s+".join(re.escape(word) for word in quote.split())
    match = re.search(pattern, chunk_text, re.IGNORECASE)
    if match:
        return match.start(), match.end(), False
    return None


def extract_claims(answer: str, chunks: dict[str, dict[str, Any]]) -> list[Claim]:
    """Split an answer into sentences and attach each citation to a located span.

    `chunks` maps chunk id -> {"text": indexed text, "doc_start": int}.
    """
    claims: list[Claim] = []
    for sentence in SENTENCE_SPLIT.split(answer.strip()):
        sentence = sentence.strip()
        if not sentence:
            continue
        claim = Claim(text=sentence)
        for chunk_id, quote in CITATION_RE.findall(sentence):
            chunk = chunks.get(chunk_id)
            if chunk is None:
                claim.verdict = FABRICATED
                claim.reason = f"cites {chunk_id}, which was not retrieved"
                continue
            if not quote:
                continue  # citation without a quote: entailment must judge it
            found = locate_quote(quote, chunk["text"])
            if found is None:
                claim.verdict = FABRICATED
                claim.reason = "quoted text does not appear in the cited passage"
                continue
            start, end, exact = found
            offset = int(chunk.get("doc_start", 0))
            claim.spans.append(
                Span(
                    chunk_id=chunk_id,
                    quote=quote,
                    start=start,
                    end=end,
                    doc_start=offset + start,
                    doc_end=offset + end,
                    exact=exact,
                )
            )
        claims.append(claim)
    return claims


ENTAIL_PROMPT = """You are checking whether each claim is supported by the exact passage \
quoted beside it. Judge only what the passage states or directly implies — not what \
you happen to know. A claim that is true in the world but absent from the passage is \
NOT supported.

{items}

Reply with ONLY a JSON array, one object per numbered item:
[{{"n": 1, "verdict": "supported"|"partial"|"unsupported", "why": "<8 words>"}}]"""


async def verify_claims(llm: Any, claims: list[Claim], max_items: int = 12) -> list[Claim]:
    """Entailment pass over located spans. One LLM call for the whole answer."""
    pending = [
        c for c in claims
        if c.verdict == UNCHECKED and c.cited
    ][:max_items]
    if not pending:
        return claims

    items = "\n\n".join(
        f"[{i}] CLAIM: {c.text}\n    PASSAGE: \"{' … '.join(s.quote for s in c.spans)}\""
        for i, c in enumerate(pending, start=1)
    )
    try:
        raw = await llm.ainvoke(ENTAIL_PROMPT.format(items=items))
        text = message_text(raw)
        match = re.search(r"\[.*\]", text, re.S)
        verdicts = json.loads(match.group(0)) if match else []
    except Exception as exc:
        print(f"[WARN] verification failed ({str(exc)[:110]}); claims left unchecked")
        return claims

    by_index = {int(v["n"]): v for v in verdicts if isinstance(v, dict) and "n" in v}
    for i, claim in enumerate(pending, start=1):
        entry = by_index.get(i)
        if not entry:
            continue
        verdict = str(entry.get("verdict", "")).lower()
        claim.verdict = verdict if verdict in (SUPPORTED, PARTIAL, UNSUPPORTED) else UNCHECKED
        claim.reason = str(entry.get("why", ""))[:120]
    return claims


def summarise(claims: list[Claim]) -> dict[str, Any]:
    counts: dict[str, int] = {}
    for claim in claims:
        counts[claim.verdict] = counts.get(claim.verdict, 0) + 1
    flagged = [
        {"text": c.text, "verdict": c.verdict, "reason": c.reason}
        for c in claims
        if c.verdict in (UNSUPPORTED, FABRICATED, PARTIAL)
    ]
    return {
        "counts": counts,
        "flagged": flagged,
        "uncited_sentences": sum(
            1 for c in claims if not c.cited and c.verdict == UNCHECKED and len(c.text) > 40
        ),
    }


def to_payload(claims: list[Claim]) -> list[dict[str, Any]]:
    """Serialise for the API, keeping the offsets so the UI can highlight."""
    return [
        {
            "text": c.text,
            "verdict": c.verdict,
            "reason": c.reason,
            "spans": [
                {
                    "chunkId": s.chunk_id,
                    "quote": s.quote,
                    "start": s.start,
                    "end": s.end,
                    "docStart": s.doc_start,
                    "docEnd": s.doc_end,
                    "exact": s.exact,
                }
                for s in c.spans
            ],
        }
        for c in claims
    ]


__all__ = [
    "Claim",
    "FABRICATED",
    "PARTIAL",
    "SUPPORTED",
    "Span",
    "UNCHECKED",
    "UNSUPPORTED",
    "CITATION_RE",
    "extract_claims",
    "locate_quote",
    "summarise",
    "to_payload",
    "verify_claims",
]
