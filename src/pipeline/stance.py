from __future__ import annotations

"""Cluster retrieved passages by stance before synthesis, not after.

Default RAG concatenates whatever it retrieved and lets the model produce one
confident paragraph, which quietly averages away disagreement. Labelling each
passage's stance toward the question first means the answer can say "four
papers find an effect, two do not" — and can report what the split tracks with,
since every passage already carries year, venue and citation count.

Honest scope note: this earns its keep on a literature that argues with itself.
On a corpus of ML methods papers, most questions come back unanimous, and the
label is then just an assurance that nothing was suppressed.
"""

import json
import re
from collections import defaultdict
from typing import Any

from .deps import message_text

AGREE = "supports"
DISAGREE = "contradicts"
NEUTRAL = "neutral"

STANCE_PROMPT = """A user asked: "{question}"

For each numbered passage, decide its stance on that question:
  supports    - it affirms or provides evidence for the proposition
  contradicts - it denies it, reports a null/negative result, or limits it
  neutral     - it is relevant background but takes no position

Judge only what the passage says.

{items}

Reply with ONLY a JSON array:
[{{"n": 1, "stance": "supports"|"contradicts"|"neutral", "claim": "<12 words>"}}]"""


async def label_stances(llm: Any, question: str, passages: list[Any], limit: int = 10) -> dict:
    """One LLM call labelling each passage's stance. Degrades to neutral."""
    subset = passages[:limit]
    if not subset:
        return {}

    from .retrieval import body_of

    items = "\n\n".join(
        f"[{i}] {body_of(p.text)[:420]}" for i, p in enumerate(subset, start=1)
    )
    try:
        raw = await llm.ainvoke(STANCE_PROMPT.format(question=question, items=items))
        text = message_text(raw)
        match = re.search(r"\[.*\]", text, re.S)
        rows = json.loads(match.group(0)) if match else []
    except Exception as exc:
        print(f"[WARN] stance labelling failed ({str(exc)[:100]}); treating all as neutral")
        return {}

    out: dict[str, dict[str, str]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        idx = int(row.get("n", 0)) - 1
        if not 0 <= idx < len(subset):
            continue
        stance = str(row.get("stance", "")).lower()
        out[subset[idx].id] = {
            "stance": stance if stance in (AGREE, DISAGREE, NEUTRAL) else NEUTRAL,
            "claim": str(row.get("claim", ""))[:120],
        }
    return out


def group_by_stance(passages: list[Any], labels: dict) -> dict[str, list[Any]]:
    groups: dict[str, list[Any]] = defaultdict(list)
    for p in passages:
        groups[(labels.get(p.id) or {}).get("stance", NEUTRAL)].append(p)
    return dict(groups)


def _paper(p: Any) -> str:
    meta = p.metadata or {}
    return meta.get("paper_title") or str(meta.get("source", "")).split("/")[-1]


def describe_split(groups: dict[str, list[Any]], labels: dict) -> str:
    """A prompt block that makes the disagreement explicit, with what it tracks.

    Year and citation count come from the bibliographic metadata already on each
    chunk, so a split can be reported against publication date rather than left
    as a bare count.
    """
    supports = {_paper(p) for p in groups.get(AGREE, [])}
    against = {_paper(p) for p in groups.get(DISAGREE, [])}
    if not against:
        return ""

    def years(bucket: str) -> list[int]:
        return sorted(
            {int((p.metadata or {}).get("pub_year") or 0) for p in groups.get(bucket, [])} - {0}
        )

    lines = [
        "DISAGREEMENT IN THE RETRIEVED EVIDENCE — do not average this away.",
        f"  {len(supports)} paper(s) support: {', '.join(sorted(supports)) or '—'}",
        f"  {len(against)} paper(s) push back: {', '.join(sorted(against)) or '—'}",
    ]
    sup_years, ag_years = years(AGREE), years(DISAGREE)
    if sup_years and ag_years and min(ag_years) > max(sup_years):
        lines.append(
            f"  The pushback is more recent ({min(ag_years)}+) than the support "
            f"(to {max(sup_years)}) — say so."
        )
    for p in groups.get(DISAGREE, [])[:4]:
        claim = (labels.get(p.id) or {}).get("claim")
        if claim:
            lines.append(f"  - {_paper(p)}: {claim}")
    lines.append("Report the split explicitly, and say which side each paper is on.")
    return "\n".join(lines)


__all__ = ["AGREE", "DISAGREE", "NEUTRAL", "describe_split", "group_by_stance", "label_stances"]
