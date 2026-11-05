#!/usr/bin/env python
"""Extract relationships from every PDF once, then assemble the graph.

Papers already extracted are skipped, so an interrupted run resumes for free.
"""

from __future__ import annotations

import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.pipeline.relations import build_graph, extract_paper  # noqa: E402
from src.pipeline.settings import PipelineSettings  # noqa: E402


def main() -> None:
    settings = PipelineSettings()
    pdfs = sorted(Path(settings.pdf_directory).glob("*.pdf"))

    def extract(pdf: Path) -> str:
        try:
            result = extract_paper(pdf, settings)
            failed = result.get("failed_chunks") or 0
            return f"{pdf.name}: {len(result['nodes'])} nodes, {len(result['edges'])} edges, {failed} failed"
        except Exception as exc:
            return f"{pdf.name}: FAILED {str(exc)[:200]}"

    with ThreadPoolExecutor(max_workers=3) as pool:
        for line in pool.map(extract, pdfs):
            print(line, flush=True)

    graph = build_graph(settings).graph
    print(f"graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")


if __name__ == "__main__":
    main()
