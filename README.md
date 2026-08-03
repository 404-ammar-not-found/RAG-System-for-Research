# Neuro-Symbolic RAG for Research

Ask questions across a corpus of papers, and see the corpus as a knowledge graph.

Two retrieval channels feed every answer:

- **Symbolic** — a [Graphiti](https://github.com/getzep/graphiti) temporal knowledge graph in
  FalkorDB. Typed entities (`Model`, `Dataset`, `Method`, `Metric`, `Task`, `Paper`, `Author`,
  `Institution`) and named relations (`Proposes`, `EvaluatedOn`, `Outperforms`, `AchievesScore`, …)
  extracted by an LLM. Entities resolve across papers, so "Transformer" in one paper is the same
  node as "Transformer" in another. This is what answers relational questions.
- **Neural** — Chroma passage retrieval (cosine) fused with BM25 by reciprocal rank fusion, then
  reranked by a single listwise LLM call. This is what grounds specific claims in verbatim text
  with citations. (Graphiti ships a cross-encoder reranker, but it scores *pointwise* — one API
  call per candidate, so 25 candidates cost 25 calls per question. One listwise call replaces it.)

```
PDF ─► section-aware parse ─┬─► sections ──► Graphiti episodes ──► FalkorDB
                            └─► chunks ────► Chroma + BM25

ask ─┬─► dense ──┐
     ├─► BM25 ───┼─► RRF ─► rerank ─► passages ─┐
     └─► graph search ────────────► facts ──────┴─► one LLM call
```

## Setup

Requires **Python 3.12+** (tested on 3.14.3) and a container runtime.

```bash
pip install -r requirements.txt
```

Docker Desktop works, but [Colima](https://github.com/abiosoft/colima) is lighter and needs no GUI:

```bash
brew install colima docker
colima start --cpu 2 --memory 4
```

Start FalkorDB. Port **6380**, because macOS dev boxes often already run a plain
`redis-server` on 6379 — and connecting to that fails with a confusing
`unknown command 'GRAPH.QUERY'` rather than a port clash:

```bash
docker run -d -p 6380:6379 -p 3000:3000 \
  -v $(pwd)/falkor_data:/var/lib/falkordb/data \
  --name falkordb falkordb/falkordb:latest
```

The graph browser is then at <http://localhost:3000> — useful for eyeballing extraction quality.

`.env`:

```
GEMINI_API_KEY=...
FALKORDB_HOST=localhost
FALKORDB_PORT=6380
```

Do **not** try to tune concurrency with `SEMAPHORE_LIMIT` in `.env`. `graphiti_core.helpers`
reads that variable at *import* time, which happens before `load_dotenv()`, so a value set there
is silently ignored and you get its default of 20 concurrent operations. Use
`PipelineSettings.max_coroutines`, which is passed to the `Graphiti` constructor directly.

### Model choice dominates everything on the free tier

Quotas vary by orders of magnitude between Gemini models, and are metered **per model**:

| model | free-tier cap | used for |
|---|---|---|
| `gemini-3-flash-preview` | **20 generate req/day** | nothing — cannot support extraction |
| `gemini-flash-latest` | workable | answers (`text_llm_model`) |
| `gemini-flash-lite-latest` | workable | extraction + reranking (`graph_llm_model`) |
| `gemini-embedding-2` | 1000 embeds/day | passages (`embed_model`) |
| `gemini-embedding-2-preview` | 1000 embeds/day | graph (`graph_embed_model`) |

Two details worth keeping:

- Graph extraction costs several LLM calls **and several embeddings per section**. A 12-paper
  corpus is ~335 sections, so budget ~2000 embeddings for the graph alone.
- Because the embedding quota is per model, the passage store and the graph deliberately use
  *different* embedding models. They index separate stores and their vectors never need to be
  comparable, so this buys two independent daily budgets instead of one shared. If you hit the
  wall, point `graph_embed_model` at another embedding model rather than waiting a day.

Ingest is resumable: PDFs already in Chroma are skipped by content hash, and episodes that fail
are logged and skipped rather than aborting the run. Re-running picks up where it left off.

## Use

Put PDFs in `data/`, then:

```bash
python main.py                                    # ingest + one question
uvicorn src.api.web_api:app --reload --port 8000  # API
cd web-visualizer && npm install && npm run dev   # UI on :5173
```

Uploading a PDF in the UI ingests it into both stores and rebuilds the graph. The visualizer
defaults to the **entity** layer; toggle to **passages** or **both**. Relation labels appear on
edges when a node is focused.

If FalkorDB is unreachable the system degrades to passage-only retrieval and says so, rather
than failing.

Regenerate the graph JSON by hand with `python scripts/export_graph.py`.

## Notes

- **Ingestion cost.** Graphiti makes several LLM calls per episode; the four sample papers are
  ~95 episodes. Expect minutes and real token spend on the first run. Chunk embedding is batched
  with exponential backoff, and a PDF that fails mid-ingest rolls back its partial chunks.
- **Re-ingesting.** The Chroma collection is created with `hnsw:space: cosine`, which Chroma only
  honours at creation. Changing chunking or the metric means `rm -rf chroma_db` and starting over.
- **Do not `pip install fitz`** — it is an unrelated PyPI package that shadows PyMuPDF's module
  name.

## Chunking

A PDF has no paragraphs. Every visual line arrives as its own string, words break across
line ends, and tables and equations are interleaved with prose as if they were sentences.
`src/pipeline/parsing.py` undoes that before anything is indexed:

- **Dehyphenation + reflow.** `computa-\ntional` is two tokens BM25 can never match; there were
  **1051** such breaks across 12 papers, now 0. Joining wrapped lines also creates the real
  paragraph boundaries the splitter needs — chunks starting mid-sentence went from **470/883 (53%)
  to 133**.
- **Typed blocks.** Captions (`Table 3:`, `Algorithm 1`) open *atomic* blocks that are never split,
  so a caption always travels with its numbers. Sustained runs of symbolic lines collapse into one
  `math` block and are excluded from the BM25 corpus; short runs stay inside their paragraph.
- **References dropped** (`index_references`). Measured: keeping them costs recall@8 0.273 → 0.227
  and 14% of embed spend.

Deliberately *not* using PyMuPDF's `find_tables`: `strategy="lines"` finds nothing in these
borderless academic layouts, and `strategy="text"` hallucinates tables out of prose — it shredded
the Attention abstract into cells like `['repr', 'oduce the ta', 'bles and']`.

## Evaluating retrieval

```bash
python scripts/eval_retrieval.py --offline --report-cost   # zero API calls
python scripts/eval_retrieval.py --full                    # costs quota
python scripts/eval_retrieval.py --offline --baseline tests/eval/baseline.json
```

`--offline` scores the golden set (`tests/eval/golden.yaml`) through BM25 only and reports chunk
health, using the *same* `build_chunk_records` the real ingest uses — so it measures the chunks
that would actually be indexed, not an approximation. On a free tier you get roughly one corpus
re-embed per day, so chunking has to be settled offline and committed in one batch.

Treat offline recall as a **regression guard**, not a quality target: several golden questions name
things the papers never call themselves (the AlexNet paper never says "AlexNet"), which lexical
search cannot resolve by construction. Chunk health is the primary offline signal.

## Tests

```bash
python -m unittest discover -s tests    # 54: parsing, chunking, fusion, reranking, graph export
cd web-visualizer && npm test           # Vitest
```
