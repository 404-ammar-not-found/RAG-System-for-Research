from __future__ import annotations

"""Entry point for running the RAG pipeline interactively."""

import asyncio

from src.pipeline.runner import run_cli
from src.pipeline.settings import PipelineSettings


def main() -> None:
    asyncio.run(run_cli(PipelineSettings(debug_retrieval=True)))


if __name__ == "__main__":
    main()
