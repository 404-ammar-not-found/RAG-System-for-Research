# Research RAG Chat

Ask questions across a library of research papers and get answers in which every claim cites a
verbatim passage, and every citation is machine-checked.

The design rule is that **retrieved text is the only evidence**. Search, reranking and the relation
graph decide what the model reads; none of them can make a claim count as verified.

## Architecture

```
                 ┌─► parse ─► chunks ─► Chroma + BM25
PDF ─► ingest ───┤
                 └─► Graphify extraction ─► relation graph (cache/relations/graph.json)

question ─► condense follow-up ─┬─► dense ──┐
                                ├─► BM25 ───┼─► RRF ─► rerank ─► neighbours ─► stance check
                                └─► graph ──┘                                       │
                                                                                    ▼
                  relationships (context only) ───────────────────────────► stream answer
                                                                                    │
                                              verify quotes and entailment ◄────────┘
```

| Stage | What happens | Code |
|---|---|---|
| Parse | Dehyphenate and reflow PDF lines; keep tables, captions and math as typed blocks | `src/pipeline/parsing.py` |
| Index | Sentence-aligned chunks in Chroma (cosine) and an in-memory BM25 index | `src/pipeline/ingestion.py` |
| Bibliography | Resolve papers through arXiv, Semantic Scholar, OpenAlex and Crossref; flag retractions | `src/pipeline/scholarly.py` |
| Relations | Merge per-paper Graphify output into one concept and citation graph | `src/pipeline/relations.py` |
| Retrieve | Fuse the channels with reciprocal rank fusion, then one listwise LLM rerank | `src/pipeline/retrieval.py` |
| Answer | Label disagreement between sources, then stream a cited answer | `src/pipeline/qa.py`, `stance.py` |
| Verify | Match each quote verbatim against its passage, then run an entailment check | `src/pipeline/verification.py` |
| Serve | FastAPI backend; React chat UI and literature map | `src/api/web_api.py`, `web-visualizer/` |

Multi-turn chat is stateless on the server: the client sends the history with each message, and
follow-ups are condensed into a standalone search query by the cheap model.

## Setup

```bash
git clone https://github.com/404-ammar-not-found/RAG-System-for-Research
cd RAG-System-for-Research
./start.sh
```

The first run writes a `.env` and stops so you can paste in an API key — a
[Gemini key](https://aistudio.google.com/apikey) (free tier is enough), an Anthropic key, or an
OpenAI key; see [Providers](#providers). Run it again and it installs everything and serves the
UI on <http://localhost:5173>. Ctrl-C stops all of it.

Requires **Python 3.12+** (tested on 3.14.3) and Node.

<details>
<summary>Running the pieces by hand</summary>

```bash
pip install -r requirements.txt
uvicorn src.api.web_api:app --port 8000
cd web-visualizer && npm install && npm run dev
```

</details>

### Providers

Any one of **Gemini**, **Anthropic** or **OpenAI**. Put a key in `.env` and it is picked up; set
`LLM_PROVIDER` to force one when several are present. Defaults per provider:

| | answers | reranking + verification | passage vectors |
|---|---|---|---|
| gemini | `gemini-flash-latest` | `gemini-flash-lite-latest` | `gemini-embedding-2` |
| anthropic | `claude-opus-5` | `claude-haiku-4-5` | — |
| openai | `gpt-5` | `gpt-5-mini` | `text-embedding-3-small` |
| local | — | — | `all-MiniLM-L6-v2` |

Override any slot with `TEXT_LLM_MODEL` / `GRAPH_LLM_MODEL` / `EMBED_MODEL` in `.env` — the
table lives in `src/pipeline/deps.py`. `GRAPH_LLM_MODEL` is the cheap-model slot: reranking,
stance labelling, verification and relation extraction.

**Any OpenAI-compatible API** works through the `openai` provider. For GLM on Z.ai:

```
LLM_PROVIDER=openai
OPENAI_API_KEY=<your Z.ai key>
OPENAI_BASE_URL=https://api.z.ai/api/paas/v4
OPENAI_API_BASE=https://api.z.ai/api/paas/v4
TEXT_LLM_MODEL=glm-5.3
GRAPH_LLM_MODEL=glm-5.3
EMBED_PROVIDER=local
```

Expect slow answers with this setup: a one-sentence question took about two minutes with GLM 5.3
in both model slots, and GLM does not always return the quoted citation format the verifier needs,
so some answers come back `unchecked`. A faster model in `GRAPH_LLM_MODEL` is the first thing to try.

### Embedding locally

Hosted embeddings are this pipeline's hard rate limit: the Gemini free tier allows **1000 per
day**, and one corpus re-embed spends most of that in a sitting. So embeddings can run on your
machine instead:

```
EMBED_PROVIDER=local
```

That runs `all-MiniLM-L6-v2` through onnxruntime, which chromadb already depends on — no new
package, no key, no network after a ~79MB model downloads to `~/.cache/chroma` on first use. It is
also the fallback when the only key you have is Anthropic's (which publishes no embedding
endpoint), so an Anthropic-only setup runs rather than refusing to start.

The trade is width and quality: **384 dimensions against Gemini's 3072**, and weaker retrieval on
paraphrase-heavy questions. It matters less here than it would elsewhere, because dense hits are
fused with BM25 and then reranked by an LLM — the vectors only have to get a passage into the
candidate set, not rank it.

Vectors of different widths cannot share a Chroma collection, so each embedding provider gets its
own (`papers-v2` for Gemini, `papers-v2-local`, `papers-v2-openai`). Switching providers therefore
means re-embedding the corpus into a new collection, not a crash.

Ingest is resumable: PDFs already in Chroma are skipped by content hash. Re-running picks up
where it left off.

## Use

Paste an arXiv link into **Add a paper** — `https://arxiv.org/abs/1706.03762`, a `/pdf/` link, or
a bare `1706.03762` — and the PDF is fetched, chunked, embedded and indexed on the spot. Uploading
a PDF does the same for anything not on arXiv. Either way only *that* paper is ingested, so one
addition never spends quota on the rest of `data/`.

Then chat. Answers stream in; every claim carries a numbered citation you can click to open the
passage it came from, unsupported claims are flagged under the answer, and retracted or
unverifiable papers get a warning banner. Conversations persist in your browser and follow-up
questions are answered in context.

For a corpus you already have, drop the PDFs in `data/` and run `python main.py` — it ingests
everything pending and answers one question from stdin. `python scripts/rebuild_all.py` rebuilds
the whole index.

## Notes

- **Re-ingesting.** The Chroma collection is created with `hnsw:space: cosine`, which Chroma only
  honours at creation. Changing chunking or the metric means `rm -rf chroma_db` and starting over.
- **Do not `pip install fitz`** — it is an unrelated PyPI package that shadows PyMuPDF's module
  name.

## Relation graph

Passage search finds text that resembles the question. It struggles when the answer is one link
away, such as "what optimizer did the Transformer paper use, and how does it work?", where the
second half lives in the Adam paper. The relation graph supplies that link.

```bash
python scripts/build_relations.py    # extract new papers, then rebuild the graph
```

**Build.** [Graphify](https://github.com/Graphify-Labs/graphify) extracts concepts and links from
each PDF once, and the result is cached in `cache/relations/`. Papers added through the UI are
extracted in the background. Three steps then turn those fragments into a corpus graph:

- **Merge across papers.** Graphify's node ids are per-file, so the same concept in two papers would
  never connect. Nodes are merged by normalised label. Graphify's fuzzy dedup stays off, because it
  is tuned for code symbols and exact merging keeps SQuAD v1.1 and v2.0 apart.
- **Find citations in the PDFs.** A paper cites another when its text contains that paper's title,
  matched with spacing and hyphenation removed. This needs no LLM or network, and it fills gaps in
  the OpenAlex citation data.
- **Link references to library papers.** A reference whose label matches a library title points at
  that paper's node rather than creating a duplicate.

**Use.** A question's terms select matching concepts, and the graph adds their direct neighbours.
That drives three things:

- a `graph` retrieval channel: BM25 widened with linked concepts, limited to the papers they touch
- a `RELATIONSHIPS` block in the prompt, which tells the model to confirm each link in the passages
- a **Relationships considered** panel under answers, plus citations and shared concepts in the map

**Trust.** Links are LLM output and are never evidence. They cannot be cited, and claims still go
through verification. Graphify's confidence label is not used to filter links, because in testing
it marked 61 of 62 links `EXTRACTED`, including vague ones like `conceptually_related_to`.

**Cost.** Queries add no LLM calls, but extraction is expensive: the BERT paper took 6 minutes and
about 19,000 input and 32,000 output tokens on GLM 5.3. Graphify also installs about 25 tree-sitter
grammars that this project never uses. Without `graph.json` the app runs on dense and BM25 alone;
set `relation_graph=False` in `PipelineSettings` to turn the feature off.

**Results.** Retrieval over the 34 golden questions, with local embeddings and no reranker:

| Setup | Found (recall@8) | MRR | Relational questions |
|---|---|---|---|
| BM25 alone | 27/34 (0.794) | 0.637 | 6/8 |
| Dense + BM25 | 26/34 (0.765) | 0.551 | 6/8 |
| Dense + BM25 + graph | 27/34 (0.794) | 0.629 | 7/8 |

The graph gains one question and improves ranking over dense + BM25, but only matches BM25 alone. On
a set this small that is not evidence of a real retrieval improvement. The 8 relational questions
were written before the graph existed, but for this feature, so they are a targeted check.

The clearer gain is the citation map: 24 links between library papers, up from 8.

**Known limits.**

- Only 11 of the 12 papers have concepts. Extraction of GPT-3 (`2005.14165`) was stopped after about
  45 minutes without a result. Re-running `build_relations.py` resumes it.
- Only about 15 of 414 concepts span more than one paper, because merging needs matching labels.
- Concept matching is broad: "Transformer" also selects Vision Transformer, so some relationships
  shown for a question are off-topic.
- `scripts/eval_answers.py` measures citation accuracy, but no result is committed yet.

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
python scripts/eval_retrieval.py --offline --baseline reports/final.json
python scripts/eval_retrieval.py --full --no-graph --no-rerank   # dense + BM25, no LLM calls
python scripts/eval_answers.py --save reports/answers.json       # citation accuracy, slow and paid
```

`--full` scores ranking before neighbour expansion. Expansion re-sorts passages by document
position, so scoring after it measures reading order rather than ranking.

`--offline` scores the golden set (`tests/eval/golden.yaml`) through BM25 only and reports chunk
health, using the *same* `build_chunk_records` the real ingest uses — so it measures the chunks
that would actually be indexed, not an approximation. On a free tier you get roughly one corpus
re-embed per day, so chunking has to be settled offline and committed in one batch.

Treat offline recall as a **regression guard**, not a quality target: several golden questions name
things the papers never call themselves (the AlexNet paper never says "AlexNet"), which lexical
search cannot resolve by construction. Chunk health is the primary offline signal.

## Tests

```bash
python -m unittest discover -s tests    # parsing, retrieval, relations, verification, streaming
cd web-visualizer && npm test           # Vitest
```
