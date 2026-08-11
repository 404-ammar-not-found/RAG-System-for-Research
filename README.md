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

```bash
git clone https://github.com/AmmarNagri/Neuro-Symbolic-Multi-Agent-RAG-System-for-Research
cd Neuro-Symbolic-Multi-Agent-RAG-System-for-Research
./start.sh
```

The first run writes a `.env` and stops so you can paste in an API key — a
[Gemini key](https://aistudio.google.com/apikey) (free tier is enough), an Anthropic key, or an
OpenAI key; see [Providers](#providers). Run it again and it
installs everything, starts the database, and serves the UI on <http://localhost:5173>. Ctrl-C
stops all of it.

Requires **Python 3.12+** (tested on 3.14.3), Node, and a container runtime. Docker Desktop works,
but [Colima](https://github.com/abiosoft/colima) is lighter and needs no GUI:

```bash
brew install colima docker
colima start --cpu 2 --memory 4
```

Without a container runtime everything still works, minus the knowledge graph — retrieval falls
back to passages alone and says so.

<details>
<summary>Running the pieces by hand</summary>

```bash
pip install -r requirements.txt
docker run -d -p 6380:6379 -p 3000:3000 \
  -v $(pwd)/falkor_data:/var/lib/falkordb/data \
  --name falkordb falkordb/falkordb:latest
uvicorn src.api.web_api:app --port 8000
cd web-visualizer && npm install && npm run dev
```

FalkorDB is on port **6380**, because macOS dev boxes often already run a plain `redis-server` on
6379 — and connecting to that fails with a confusing `unknown command 'GRAPH.QUERY'` rather than a
port clash. Its graph browser is at <http://localhost:3000>, useful for eyeballing extraction
quality.

</details>

Do **not** try to tune concurrency with `SEMAPHORE_LIMIT` in `.env`. `graphiti_core.helpers`
reads that variable at *import* time, which happens before `load_dotenv()`, so a value set there
is silently ignored and you get its default of 20 concurrent operations. Use
`PipelineSettings.max_coroutines`, which is passed to the `Graphiti` constructor directly.

### Providers

Any one of **Gemini**, **Anthropic** or **OpenAI**. Put a key in `.env` and it is picked up; set
`LLM_PROVIDER` to force one when several are present. Defaults per provider:

| | answers | extraction + reranking | passage vectors | graph vectors |
|---|---|---|---|---|
| gemini | `gemini-flash-latest` | `gemini-flash-lite-latest` | `gemini-embedding-2` | `gemini-embedding-2-preview` |
| anthropic | `claude-opus-5` | `claude-haiku-4-5` | — | — |
| openai | `gpt-5` | `gpt-5-mini` | `text-embedding-3-small` | `text-embedding-3-large` |
| local | — | — | `all-MiniLM-L6-v2` | `all-MiniLM-L6-v2` |

Override any slot with `TEXT_LLM_MODEL` / `GRAPH_LLM_MODEL` / `EMBED_MODEL` / `GRAPH_EMBED_MODEL`
in `.env` — the table lives in `src/pipeline/deps.py`.

### Embedding locally

Hosted embeddings are this pipeline's hard rate limit: the Gemini free tier allows **1000 per
day**, and one corpus re-embed plus graph extraction spends that in a sitting. So embeddings can
run on your machine instead:

```
EMBED_PROVIDER=local
```

That runs `all-MiniLM-L6-v2` through onnxruntime, which chromadb already depends on — no new
package, no key, no network after a ~79MB model downloads to `~/.cache/chroma` on first use. It is
also the fallback when the only key you have is Anthropic's, so an Anthropic-only setup now runs
rather than refusing to start.

The trade is width and quality: **384 dimensions against Gemini's 3072**, and weaker retrieval on
paraphrase-heavy questions. It matters less here than it would elsewhere, because dense hits are
fused with BM25 and then reranked by an LLM — the vectors only have to get a passage into the
candidate set, not rank it.

Vectors of different widths cannot share a Chroma collection, so each embedding provider gets its
own (`papers-v2` for Gemini, `papers-v2-local`, `papers-v2-openai`). Switching providers therefore
means re-embedding the corpus into a new collection, not a crash — but the same is not true of the
graph, whose vectors live in FalkorDB under one index: change the embedding provider after
building a graph and you should rebuild it.

**Embeddings are only half of extraction.** Building the entity layer also makes several *LLM*
calls per section, which still consume whatever quota your chat provider has. Local embeddings
remove the 1000/day embedding cap; they do not make graph extraction free.

Two more things this layout works around:

- **Anthropic publishes no embedding endpoint.** With only an `ANTHROPIC_API_KEY`, Claude answers
  and embeddings fall back to local (or to a Gemini/OpenAI key if one is present).
- **Claude's 5-series rejects `temperature` outright** (HTTP 400), and Graphiti's `AnthropicClient`
  always sends it — which is why graph extraction runs on Haiku 4.5 rather than a 5-series model.
  For the same reason no sampling parameter is sent to Anthropic or OpenAI at all; only Gemini
  gets one.

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

Paste an arXiv link into **Add a paper** — `https://arxiv.org/abs/1706.03762`, a `/pdf/` link, or
a bare `1706.03762` — and the PDF is fetched, chunked, embedded and added to the graph on the
spot. Uploading a PDF does the same for anything not on arXiv. Either way only *that* paper is
ingested, so one addition never spends quota on the rest of `data/`.

Then ask a question. Every claim comes back with a numbered citation you can click to focus its
passage in the graph, and the papers an answer drew on open automatically.

### Reading the figure

Every node is a filled circle with its name written **inside** it, every named edge carries its
relation, and relations point in the direction they assert — `Transformer Outperforms LSTM` is not
the same claim read backwards. Colour is class: each paper has its own hue and its passages inherit
it, while entities take the colour of their type (the legend under the plate).

The corpus is drawn as a hierarchy, because drawing it flat does not work: one node per passage is
700 equally-weighted dots with no structure, which is a swarm rather than a figure.

- **Overview** (default) — one labelled node per paper, pinned to a ring and sized by passage
  count, with entities in the middle. Click a paper to open its passages; click again to fold them
  away. Deterministic: the same corpus draws the same picture every load.
- **Passages** — every passage, tethered to the paper it came from.
- **Everything** — both, plus the relations between entities.

Entity → passage provenance is rolled up to **entity → paper**, so an entity linked to three papers
is visible as exactly that: the cross-paper resolution the graph exists to produce. The raw
per-passage provenance edges are never drawn — thousands of them are what made the old view a
hairball.

That roll-up also decides the layout. An entity found in **one** paper is drawn tight against it as
a fan; one found in **several** gets long, slack edges and settles in the space between them. The
picture says which is which before you read a label.

Names are budgeted, because naming everything is what a printed figure can do and a live canvas
cannot. Papers are always named. Entities are named while they number a couple of hundred or fewer,
and past that only the ones spanning papers keep a standing label. Passages are named only inside a
paper you opened, and only while few enough are on screen to read. Everything else names itself on
hover.

The caption reports what is actually on screen, including an empty entity layer. Papers and
passages come from Chroma, so they appear as soon as a PDF is ingested; entities need graph
extraction to have run.

### Building the entity layer for papers you already ingested

`main.py` only extracts entities for PDFs that are new to Chroma, so a corpus embedded before the
graph existed never gains one. Run the graph half on its own, one paper at a time:

```bash
python scripts/extract_paper.py data/1406.2661.pdf
```

Budget for it: extraction is several LLM calls **and** several embeddings per section, and the
Gemini free tier allows 1000 embeddings per day across the whole corpus.

For a corpus you already have, drop the PDFs in `data/` and run `python main.py` — it ingests
everything pending and answers one question from stdin.

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
  paragraph boundaries the splitter needs.
- **Sentence-aligned splits.** `RecursiveCharacterTextSplitter`'s default separators bottom out at
  `" "`, so any paragraph longer than `chunk_size` was cut at whatever word landed on the limit.
  Offering `". "`, `"? "`, `"! "` ahead of the space (with `keep_separator="end"`, or the
  terminator is deleted rather than kept) moves the cut to a sentence end: chunks starting
  mid-sentence went **470/883 (53%) → 85/949 (9%)**, recall@8 flat, MRR 0.176 → 0.199.

  The remaining 85 are not splitter damage and adding separators cannot remove them — they are
  measured with `chunk_overlap=0` too. They are *blocks* whose source text begins lowercase
  because a display equation, table or caption interrupted the paragraph: a chunk opening
  `where ζ = 0 if the true second moment…` is a paragraph resuming after its equation, which is
  correctly typeset and correctly split. About a third are not prose at all — table cell runs
  (`top-1 err. top-5 err. VGG-16 [41] 28.07 9.33`) that trip the metric's lowercase-start test.
  Driving this to 0 means merging continuation blocks back across the equation that separates
  them, which reorders the text; the honest number is 9%, not 0%.
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
python -m unittest discover -s tests    # 80: parsing, chunking, fusion, reranking, graph export
cd web-visualizer && npm test           # Vitest
```
