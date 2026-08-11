from __future__ import annotations

"""Run graph extraction for one already-ingested paper.

`main.py` only extracts entities for PDFs that are new to Chroma, so a corpus
that was embedded before the graph existed can never gain one. This does the
graph half on its own, for a single paper, which is also the cheapest way to
smoke-test extraction: a full corpus is several LLM calls plus embeddings per
section across hundreds of sections.

    python scripts/extract_paper.py data/1406.2661.pdf
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.pipeline.graph_export import export_graph
from src.pipeline.graph_store import GraphStore
from src.pipeline.ingestion import open_vectorstore
from src.pipeline.parsing import parse_paper
from src.pipeline.settings import PipelineSettings


async def extract(pdf: Path, settings: PipelineSettings) -> int:
    sections = parse_paper(pdf, settings.max_pages)
    if not sections:
        print(f"No text extracted from {pdf.name}")
        return 0

    print(f"{pdf.name}: {len(sections)} sections")
    graph = GraphStore(settings)
    await graph.initialize()
    try:
        episodes = await graph.ingest_sections(sections)
        print(f"Extracted {episodes} episodes")
        # Re-export so the visualiser picks up the entity layer.
        await export_graph(graph, open_vectorstore(settings))
        return episodes
    finally:
        await graph.close()


def main() -> None:
    if len(sys.argv) != 2:
        print(__doc__)
        raise SystemExit(2)
    pdf = Path(sys.argv[1])
    if not pdf.exists():
        raise SystemExit(f"No such file: {pdf}")
    asyncio.run(extract(pdf, PipelineSettings()))


if __name__ == "__main__":
    main()
