#!/usr/bin/env python
"""Measure retrieval quality.

    --offline   (default) ZERO API calls. Parses the PDFs, builds chunks in
                memory, scores the golden set through BM25 only, and reports
                structural chunk health. This is what makes chunking tunable on
                a free tier where you get roughly one corpus re-embed per day.

    --full      Dense + BM25 + fusion + rerank against the live Chroma index.
                Costs one query embedding (and one rerank call) per question.

Compare runs with --baseline to see whether a change actually helped.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import statistics
import sys
from collections import Counter
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.pipeline.ingestion import build_chunk_records  # noqa: E402
from src.pipeline.parsing import parse_paper  # noqa: E402
from src.pipeline.retrieval import Bm25Index  # noqa: E402
from src.pipeline.settings import PipelineSettings  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
GOLDEN = REPO_ROOT / "tests" / "eval" / "golden.yaml"
HYPHEN_BREAK = re.compile(r"[a-z]-\n[a-z]")
SENTENCE_END = re.compile(r"[.!?:;)\]]\s*$")


# --------------------------------------------------------------------------
# corpus
# --------------------------------------------------------------------------


def build_chunks(settings: PipelineSettings) -> list[dict]:
    """Exactly the chunks a real ingest would embed — same function, no API calls."""
    sections = [
        s
        for pdf in sorted(settings.pdf_directory.glob("*.pdf"))
        for s in parse_paper(pdf, settings.max_pages)
    ]
    return build_chunk_records(sections, settings)


# --------------------------------------------------------------------------
# structural health — the part that needs no API and catches real damage
# --------------------------------------------------------------------------


def chunk_health(chunks: list[dict]) -> dict:
    sizes = [len(c["text"]) for c in chunks]
    types = Counter(c["metadata"].get("content_type", "prose") for c in chunks)
    papers = Counter(Path(c["metadata"]["source"]).stem for c in chunks)

    hyphen_breaks = sum(len(HYPHEN_BREAK.findall(c["text"])) for c in chunks)
    refs = sum(1 for c in chunks if c["metadata"].get("content_type") == "reference")
    # A chunk that starts mid-sentence is one the splitter cut badly.
    body_starts = [c["text"].split("\n\n", 1)[-1].lstrip() for c in chunks]
    mid_sentence = sum(1 for b in body_starts if b[:1].islower())
    orphans = sum(1 for s in sizes if s < 120)

    return {
        "chunks": len(chunks),
        "papers": len(papers),
        "hyphen_breaks": hyphen_breaks,
        "reference_chunks": refs,
        "starts_mid_sentence": mid_sentence,
        "orphan_chunks": orphans,
        "size_mean": round(statistics.mean(sizes)) if sizes else 0,
        "size_median": round(statistics.median(sizes)) if sizes else 0,
        "size_max": max(sizes) if sizes else 0,
        "content_types": dict(types),
    }


# --------------------------------------------------------------------------
# scoring
# --------------------------------------------------------------------------


def is_hit(item: dict, meta: dict, text: str) -> bool:
    if Path(str(meta.get("source", ""))).stem != item["paper"]:
        return False
    low = text.lower()
    return any(c.lower() in low for c in item["contains"])


def score(results: list[tuple[dict, list[tuple[dict, str]]]], k: int) -> dict:
    """recall@k and MRR over (golden_item, ranked [(meta, text)]) pairs."""
    hits = 0
    rr_total = 0.0
    by_kind: dict[str, list[int]] = {}
    misses: list[str] = []

    for item, ranked in results:
        rank = 0
        for i, (meta, text) in enumerate(ranked[:k], start=1):
            if is_hit(item, meta, text):
                rank = i
                break
        if rank:
            hits += 1
            rr_total += 1.0 / rank
        else:
            misses.append(item["q"])
        by_kind.setdefault(item.get("kind", "other"), []).append(1 if rank else 0)

    n = len(results) or 1
    return {
        f"recall@{k}": round(hits / n, 3),
        "mrr": round(rr_total / n, 3),
        "hits": f"{hits}/{len(results)}",
        "by_kind": {kd: f"{sum(v)}/{len(v)}" for kd, v in sorted(by_kind.items())},
        "misses": misses,
    }


# --------------------------------------------------------------------------
# modes
# --------------------------------------------------------------------------


def run_offline(golden: list[dict], settings: PipelineSettings, k: int) -> dict:
    chunks = build_chunks(settings)
    health = chunk_health(chunks)

    # BM25 over the same bodies the real index would use.
    index = Bm25Index()
    index.load_texts(
        [(c["metadata"]["id"], c["text"], c["metadata"]) for c in chunks]
    )

    results = []
    for item in golden:
        hits = index.search(item["q"], k=k)
        results.append((item, [(h.metadata, h.text) for h in hits]))

    return {"mode": "offline", "health": health, **score(results, k)}


async def run_full(golden: list[dict], settings: PipelineSettings, k: int) -> dict:
    from src.pipeline.runner import Runtime

    runtime = Runtime(settings)
    await runtime.start()
    try:
        results = []
        for item in golden:
            passages = await runtime.qa.retrieve(item["q"])
            results.append((item, [(p.metadata, p.text) for p in passages]))
        out = {"mode": "full", **score(results, k)}
    finally:
        await runtime.close()
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true", help="hit the live index (costs quota)")
    ap.add_argument("--offline", action="store_true", help="default mode; accepted explicitly")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--report-cost", action="store_true", help="projected embeds for a re-ingest")
    ap.add_argument("--baseline", type=Path, help="compare against a saved JSON report")
    ap.add_argument("--save", type=Path, help="write this report to JSON")
    args = ap.parse_args()

    golden = yaml.safe_load(GOLDEN.read_text())
    settings = PipelineSettings(
        pdf_directory=REPO_ROOT / "data",
        chroma_path=str(REPO_ROOT / "chroma_db"),
        debug_retrieval=False,
        top_k=args.k,
    )

    report = (
        asyncio.run(run_full(golden, settings, args.k))
        if args.full
        else run_offline(golden, settings, args.k)
    )

    print(json.dumps({kk: vv for kk, vv in report.items() if kk != "misses"}, indent=2))
    if report.get("misses"):
        print("\nmissed:")
        for q in report["misses"]:
            print("  -", q)

    if args.report_cost and "health" in report:
        n = report["health"]["chunks"]
        print(f"\nprojected re-ingest: {n} embeds "
              f"({'fits' if n <= 1000 else 'EXCEEDS'} a 1000/day bucket)")

    if args.baseline and args.baseline.exists():
        base = json.loads(args.baseline.read_text())
        print("\nvs baseline:")
        for key in (f"recall@{args.k}", "mrr"):
            if key in base and key in report:
                delta = report[key] - base[key]
                print(f"  {key}: {base[key]} -> {report[key]}  ({delta:+.3f})")
        for key in ("hyphen_breaks", "reference_chunks", "chunks", "starts_mid_sentence"):
            if "health" in base and "health" in report and key in base["health"]:
                b, r = base["health"][key], report["health"][key]
                print(f"  {key}: {b} -> {r}  ({r - b:+d})")

    if args.save:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        args.save.write_text(json.dumps(report, indent=2))
        print(f"\nsaved {args.save}")


if __name__ == "__main__":
    main()
