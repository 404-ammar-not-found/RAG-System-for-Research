#!/usr/bin/env python
"""Measure answer citations over the golden set. Costs LLM calls for every question.

Quote accuracy is deterministic: is each quoted span really in the cited passage?
Entailment comes from the same LLM family that wrote the answer, so treat it as a
self-check rather than independent proof.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
from pathlib import Path
from typing import Any

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.pipeline.settings import PipelineSettings  # noqa: E402
from src.pipeline.verification import FABRICATED, PARTIAL, SUPPORTED, UNCHECKED, UNSUPPORTED  # noqa: E402

GOLDEN = Path(__file__).resolve().parents[1] / "tests" / "eval" / "golden.yaml"
_BARE_CITATION = re.compile(r"\[[^\[\]\s\"]+-chunk-\d+\]")


def _rate(part: int, whole: int) -> float | None:
    return round(part / whole, 3) if whole else None


def summarise_run(rows: list[dict[str, Any]]) -> dict[str, Any]:
    scored = [r for r in rows if not r.get("error")]
    verdicts = [v for r in scored for v in r["verdicts"]]
    judged = sum(v in (SUPPORTED, PARTIAL, UNSUPPORTED) for v in verdicts)
    fabricated = verdicts.count(FABRICATED)
    supported = verdicts.count(SUPPORTED)
    quotes = judged + fabricated
    evidence = sum(bool(r["cited_expected_evidence"]) for r in scored)
    return {
        "answers": {"scored": len(scored), "failed": len(rows) - len(scored)},
        "quotes": {"found": judged, "total": quotes, "rate": _rate(judged, quotes)},
        "entailment": {"supported": supported, "judged": judged, "rate": _rate(supported, judged)},
        "bare_citations": sum(r["bare_citations"] for r in scored),
        "cited_expected_evidence": {
            "answers": evidence,
            "total": len(scored),
            "rate": _rate(evidence, len(scored)),
        },
    }


def _cited_evidence(item: dict[str, Any], result: dict[str, Any]) -> bool:
    """Did the answer cite a passage from the expected paper holding the expected text?"""
    cited = {span["chunkId"] for claim in result.get("claims", []) for span in claim["spans"]}
    wanted = [text.lower() for text in item["contains"]]
    for match in result.get("matches", []):
        meta = match["metadata"]
        if (
            meta.get("id") in cited
            and Path(str(meta.get("source"))).stem == item["paper"]
            and any(text in match["document"].lower() for text in wanted)
        ):
            return True
    return False


async def _ask(runtime: Any, item: dict[str, Any], gate: asyncio.Semaphore) -> dict[str, Any]:
    out: dict[str, Any] = {"q": item["q"], "kind": item["kind"]}
    async with gate:
        try:
            result = await runtime.qa.answer(item["q"])
        except Exception as exc:
            print(f"FAILED {item['q'][:60]}: {str(exc)[:120]}", flush=True)
            return {**out, "error": str(exc)[:300]}
    # A sentence with no quoted citation has nothing to check, so only count
    # claims that carry a verdict or a located quote.
    claims = result.get("claims", [])
    verdicts = [c["verdict"] for c in claims if c["verdict"] != UNCHECKED]
    verdicts += ["judged_none" for c in claims if c["verdict"] == UNCHECKED and c["spans"]]
    answer = result.get("answer", "")
    print(f"done {item['q'][:60]}: {result.get('verification', {}).get('counts')}", flush=True)
    return {
        **out,
        "answer": answer,
        "claims": claims,
        "verdicts": verdicts,
        "uncited_sentences": result.get("verification", {}).get("uncited_sentences", 0),
        "bare_citations": len(_BARE_CITATION.findall(answer)),
        "cited_expected_evidence": _cited_evidence(item, result),
        "error": None,
    }


async def run(args: argparse.Namespace) -> None:
    from src.pipeline.runner import Runtime

    settings = PipelineSettings(
        debug_retrieval=False, suggest_followups=False, relation_graph=not args.no_graph
    )
    runtime = Runtime(settings)
    await runtime.start()
    golden = yaml.safe_load(GOLDEN.read_text())[: args.limit]
    gate = asyncio.Semaphore(args.concurrency)
    rows: list[dict[str, Any]] = []

    def save() -> None:
        if args.save:
            args.save.write_text(json.dumps({"summary": summarise_run(rows), "rows": rows}, indent=2))

    # Saved after every answer, because a full run takes hours and costs real money.
    for pending in asyncio.as_completed([_ask(runtime, item, gate) for item in golden]):
        rows.append(await pending)
        save()
    report = summarise_run(rows)
    print(json.dumps(report, indent=2))
    if args.save:
        print(f"saved {args.save}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-graph", action="store_true")
    ap.add_argument("--limit", type=int, help="only the first N questions")
    ap.add_argument("--concurrency", type=int, default=2)
    ap.add_argument("--save", type=Path)
    asyncio.run(run(ap.parse_args()))


if __name__ == "__main__":
    main()
