#!/usr/bin/env python
"""Full rebuild: parse every PDF once and fill Chroma."""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.pipeline.ingestion import ingest_chroma  # noqa: E402
from src.pipeline.settings import PipelineSettings  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    t0 = time.time()
    settings = PipelineSettings(
        pdf_directory=REPO_ROOT / "data",
        chroma_path=str(REPO_ROOT / "chroma_db"),
    )

    print("=== Chroma ===", flush=True)
    vectordb, chunks, _ = ingest_chroma(settings)
    print(f"chroma: {chunks} new chunks, {vectordb._collection.count()} total", flush=True)
    print(f"done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
