# Research RAG Chat

Ask questions across a corpus of papers in a claude.ai-style chat, with every claim
grounded in a verbatim, machine-checked citation.

Retrieval is Chroma passage search (cosine) fused with BM25 by reciprocal rank fusion, then
reranked by a single listwise LLM call. Answers stream token-by-token; after synthesis every
cited quote is checked verbatim against its passage and every sentence is verified by an
entailment call, so fabricated or unsupported claims are flagged in the UI.

```
PDF ─► section-aware parse ─► chunks ─► Chroma + BM25

chat ─► condense follow-up ─┬─► dense ──┐
                            └─► BM25 ───┴─► RRF ─► rerank ─► passages ─► stream answer ─► verify
```

Multi-turn: the client sends the conversation history with each message; follow-ups are
condensed into standalone retrieval queries by the cheap model.

## Setup

```bash
git clone https://github.com/AmmarNagri/Neuro-Symbolic-Multi-Agent-RAG-System-for-Research
cd Neuro-Symbolic-Multi-Agent-RAG-System-for-Research
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
table lives in `src/pipeline/deps.py`. (`GRAPH_LLM_MODEL` is the cheap-model slot, named for the
graph-extraction work it once also did.)

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
python -m unittest discover -s tests    # parsing, chunking, fusion, reranking, verification
cd web-visualizer && npm test           # Vitest
```
