from __future__ import annotations

"""Section-aware PDF parsing.

One parse feeds both consumers: Graphiti episodes (section granularity) and
Chroma chunks (blocks within a section).

The hard part is not finding sections, it is that a PDF has no paragraphs. Every
visual line arrives as its own string, words are hyphenated across line ends, and
tables, captions and display equations are interleaved with prose as if they were
sentences. Everything below exists to undo that before any text is indexed.
"""

import re
from collections import Counter
from dataclasses import dataclass, field
from hashlib import sha256
from pathlib import Path

import pymupdf

# ponytail: font-size header heuristic, tuned for arXiv/NeurIPS two-column papers.
# Knobs below. Falls back to one section per page when it finds <2 headers, so a
# weird layout degrades to the old behaviour instead of losing text. Swap for a
# layout model (GROBID / unstructured) if it misses on your corpus.
HEADER_SIZE_RATIO = 1.15
HEADER_MAX_CHARS = 80
EPISODE_MAX_CHARS = 4000
BLOCK_MAX_CHARS = 3000
# Consecutive symbolic lines needed before a run counts as a display equation
# rather than a symbol-heavy sentence inside a paragraph.
MATH_RUN_MIN = 3
# Below this a block is a fragment, not a passage; fold it into its neighbour.
MIN_BLOCK_CHARS = 200

PROSE = "prose"
MATH = "math"
REFERENCE = "reference"

_BOLD_FLAG = 1 << 4
_ARXIV_STAMP = re.compile(r"^\s*arxiv:\s*\d", re.I)

# Numbered headings: "3.1 Attention", "1.Introduction", "IV. Results".
# The trailing [A-Z] is what rejects "3.3 · 1018" and other numeric table debris.
_NUMBERED_HEADER = re.compile(r"^\s*(\d+(\.\d+)*|[IVXLC]+)\.?\s*[A-Z][A-Za-z]")

# "Table 3:", "Figure 2.", "Algorithm 1" — opens an atomic block.
_CAPTION = re.compile(r"^\s*(table|figure|fig\.?|algorithm|listing)\s*\d+", re.I)
_CAPTION_KIND = {"fig": "figure", "fig.": "figure"}

_REFERENCE_SECTION = re.compile(
    r"^\s*(references?|bibliography|acknowledge?ments?)\b", re.I
)

# Characters that mean "this line is an equation, not a sentence".
_MATH_CHARS = set("∑∏∫∂∇√±≤≥≈≠∈∀∃⊤⊆∪∩·×→←↦⟨⟩αβγδεθλμσφψωΓΔΘΛΣΦΨΩ^_{}")
_SENTENCE_END = re.compile(r"[.!?:;]['\")\]]?\s*$")
_STOPWORDS = {
    "the", "a", "an", "of", "to", "in", "is", "are", "we", "that", "for",
    "and", "with", "as", "on", "by", "this", "it", "be", "from", "which",
}

_KNOWN_HEADERS = {
    "abstract", "introduction", "background", "related work", "method", "methods",
    "methodology", "approach", "model", "architecture", "experiments",
    "experimental setup", "results", "evaluation", "discussion", "analysis",
    "ablation", "ablation study", "conclusion", "conclusions", "future work",
    "references", "appendix", "acknowledgements", "acknowledgments",
}


@dataclass
class Block:
    """A run of one kind of content inside a section. The unit of chunking."""

    content_type: str
    text: str

    @property
    def atomic(self) -> bool:
        """Tables, figures and algorithms are never split across chunks.

        References are long lists, not units — they must stay splittable, or a
        whole bibliography becomes one 26k-character chunk.
        """
        return self.content_type not in (PROSE, MATH, REFERENCE)


@dataclass
class Section:
    """A titled span of a paper. The unit of Graphiti episode ingestion."""

    paper_title: str
    section: str
    text: str
    source: str
    file_hash: str
    pages: list[int] = field(default_factory=list)
    blocks: list[Block] = field(default_factory=list)

    @property
    def paper_stem(self) -> str:
        return Path(self.source).stem

    @property
    def episode_name(self) -> str:
        """Join key between a Graphiti episode and its Chroma chunks."""
        return f"{self.paper_stem}::{self.section}"

    @property
    def is_reference(self) -> bool:
        return bool(_REFERENCE_SECTION.match(self.section))


# ---------------------------------------------------------------------------
# page -> lines
# ---------------------------------------------------------------------------


def _lines_with_sizes(page: pymupdf.Page) -> list[tuple[str, float, bool]]:
    """Flatten a page into (line_text, max_span_size, is_bold) triples, reading order."""
    lines: list[tuple[str, float, bool]] = []
    for block in page.get_text("dict").get("blocks", []):
        if block.get("type") != 0:  # skip images
            continue
        for line in block.get("lines", []):
            spans = line.get("spans", [])
            text = "".join(s.get("text", "") for s in spans).strip()
            if not text:
                continue
            size = max((float(s.get("size", 0.0)) for s in spans), default=0.0)
            # Bold if every non-blank span is bold — a heading is uniformly
            # weighted, a body sentence with one bold term is not.
            weighted = [s for s in spans if s.get("text", "").strip()]
            bold = bool(weighted) and all(
                int(s.get("flags", 0)) & _BOLD_FLAG for s in weighted
            )
            lines.append((text, size, bold))
    return lines


def _body_size(all_lines: list[tuple[str, float, bool]]) -> float:
    """Modal font size across the document = body text size."""
    counts = Counter(round(size, 1) for _, size, _ in all_lines)
    return counts.most_common(1)[0][0] if counts else 10.0


def _is_header(text: str, size: float, bold: bool, body: float) -> bool:
    if len(text) > HEADER_MAX_CHARS or _ARXIV_STAMP.match(text):
        return False
    if _CAPTION.match(text):  # "Figure 3" is a caption, not a section heading
        return False
    if size > body * HEADER_SIZE_RATIO:
        return True
    # Same-size bold headings are common (CNNpred, most Elsevier templates).
    # Weight is the signal that separates "4 Experiments" from a table cell
    # reading "4 Indian stocks", which no text-only rule can distinguish.
    stripped = text.strip().rstrip(".").lower()
    if stripped in _KNOWN_HEADERS:
        return True
    return bold and bool(_NUMBERED_HEADER.match(text)) and len(text.split()) <= 10


def _paper_title(doc: pymupdf.Document, body: float) -> str:
    """Largest text on page 1, ignoring the rotated arXiv margin stamp."""
    if doc.page_count == 0:
        return ""
    lines = [ln for ln in _lines_with_sizes(doc[0]) if not _ARXIV_STAMP.match(ln[0])]
    if not lines:
        return ""
    top_size = max(size for _, size, _ in lines)
    if top_size <= body:
        return lines[0][0]
    return " ".join(t for t, s, _ in lines if s >= top_size - 0.5).strip()[:200]


# ---------------------------------------------------------------------------
# normalisation — the single highest-value step in this file
# ---------------------------------------------------------------------------


def _is_mathy(line: str) -> bool:
    """A display equation, not a sentence.

    Deliberately conservative. An earlier, looser version classified "We use
    N = 6 layers" as math because it counted "We" as a short token, which then
    tore the surrounding paragraph into fragments. Prose misread as math costs
    far more than math misread as prose, so the bar is high: real words veto.
    """
    stripped = line.strip()
    if not stripped or len(stripped) > 200:
        return False

    words = stripped.split()
    # An alphabetic token of 3+ chars is evidence of a sentence.
    real_words = [w for w in words if len(w) >= 3 and w.strip(".,()").isalpha()]
    if len(real_words) >= 3:
        return False
    if sum(1 for w in words if w.lower().strip(".,()") in _STOPWORDS) >= 2:
        return False

    symbols = sum(1 for ch in stripped if ch in _MATH_CHARS)
    if symbols and symbols / len(stripped) > 0.05:
        return True

    # No maths glyphs, so demand near-total absence of prose plus operator or
    # digit density: a bare fraction line or a variable assignment.
    if len(real_words) > 1:
        return False
    ops = sum(1 for ch in stripped if ch in "=+-/*<>^_")
    digits = sum(1 for ch in stripped if ch.isdigit())
    return (ops + digits) / len(stripped) > 0.15


def splice_hyphens(lines: list[str]) -> list[str]:
    """Rejoin words broken across a line end, preserving the line structure.

    Used for table/figure blocks, which must keep their rows but should not keep
    "computa-\\ntional" as two unmatchable tokens.
    """
    out: list[str] = []
    for raw in lines:
        line = raw.strip()
        if not line:
            continue
        if out and out[-1].endswith("-") and len(out[-1]) > 1 \
                and out[-1][-2].isalpha() and line[:1].islower():
            out[-1] = out[-1][:-1] + line
        else:
            out.append(line)
    return out


def reflow(lines: list[str]) -> str:
    """Turn PDF display lines back into paragraphs.

    Two fixes, both of which change what gets indexed:

    * dehyphenation — "computa-\\ntional" is two unmatchable BM25 tokens, and it
      is common: 1051 occurrences across this 12-paper corpus.
    * line joining — a PDF line break is typography, not structure. Joining
      wrapped lines produces real paragraph boundaries, which is what lets the
      splitter cut between sentences instead of through them. 53% of chunks
      started mid-sentence before this existed.
    """
    paragraphs: list[str] = []
    current: list[str] = []

    for line in splice_hyphens(lines):
        if current:
            prev = current[-1]
            if _SENTENCE_END.search(prev) and line[:1].isupper():
                # Sentence ended and a new one starts: paragraph boundary.
                paragraphs.append(" ".join(current))
                current = [line]
                continue
            current.append(line)
        else:
            current.append(line)

    if current:
        paragraphs.append(" ".join(current))
    return "\n\n".join(p.strip() for p in paragraphs if p.strip())


def _caption_kind(line: str) -> str:
    word = _CAPTION.match(line).group(1).lower()
    return _CAPTION_KIND.get(word, word)


def build_blocks(lines: list[str], is_reference: bool) -> list[Block]:
    """Group a section's lines into typed blocks.

    Captions anchor atomic blocks: a table is retrievable through its caption and
    its numbers, so the caption must travel with the numbers and the pair must
    never be split. Reconstructing the cell grid buys little and costs a lot —
    PyMuPDF's own table finder returns nothing on these borderless layouts and
    hallucinates tables out of prose when forced.
    """
    if is_reference:
        return [Block(REFERENCE, reflow(lines))] if lines else []

    blocks: list[Block] = []
    prose: list[str] = []
    math: list[str] = []
    atomic: list[str] | None = None
    atomic_kind = ""

    def flush_prose() -> None:
        nonlocal prose
        if prose:
            text = reflow(prose)
            if text:
                blocks.append(Block(PROSE, text))
            prose = []

    def flush_math() -> None:
        """Collapse the vertical soup onto one line — not reconstructing LaTeX,
        just making it one unit instead of twenty fragments.

        A short run stays *inside* the paragraph it interrupts. Only a sustained
        run is really a display equation worth isolating; promoting every stray
        symbolic line to its own block is what fragmented prose into 222-char
        median chunks on the first attempt.
        """
        nonlocal math
        if not math:
            return
        collapsed = " ".join(m.strip() for m in math)
        if len(math) >= MATH_RUN_MIN:
            flush_prose()
            blocks.append(Block(MATH, collapsed))
        else:
            prose.append(collapsed)
        math = []

    def flush_atomic() -> None:
        nonlocal atomic, atomic_kind
        if atomic:
            # Keep the row structure (it is the only thing standing in for the
            # grid) but still repair words broken across line ends.
            joined = "\n".join(splice_hyphens(atomic))
            blocks.append(Block(atomic_kind, joined[:BLOCK_MAX_CHARS]))
        atomic, atomic_kind = None, ""

    for raw in lines:
        line = raw.strip()
        if not line:
            continue

        if _CAPTION.match(line):
            flush_prose(); flush_math(); flush_atomic()
            atomic, atomic_kind = [line], _caption_kind(line)
            continue

        if atomic is not None:
            # Prose has resumed when we see a long, properly-terminated sentence.
            resumed = len(line) > 100 and bool(_SENTENCE_END.search(line))
            if resumed or sum(len(x) for x in atomic) > BLOCK_MAX_CHARS:
                flush_atomic()
                prose.append(line)
            else:
                atomic.append(line)
            continue

        if _is_mathy(line):
            math.append(line)
            continue

        flush_math()
        prose.append(line)

    flush_math(); flush_prose(); flush_atomic()
    return _merge_small(blocks)


def _merge_small(blocks: list[Block], floor: int = MIN_BLOCK_CHARS) -> list[Block]:
    """Fold undersized non-atomic blocks into their neighbour.

    A 40-character chunk carries no retrievable meaning but still occupies a
    top-k slot and costs an embedding.
    """
    out: list[Block] = []
    for block in blocks:
        if (
            out
            and not block.atomic
            and not out[-1].atomic
            and len(block.text) < floor
            and out[-1].content_type in (PROSE, block.content_type)
        ):
            out[-1] = Block(out[-1].content_type, f"{out[-1].text}\n\n{block.text}")
        else:
            out.append(block)
    return [b for b in out if b.text.strip()]


# ---------------------------------------------------------------------------
# parse
# ---------------------------------------------------------------------------


def _split_long(text: str, limit: int) -> list[str]:
    """Split on paragraph boundaries so no piece exceeds `limit`."""
    if len(text) <= limit:
        return [text]
    pieces: list[str] = []
    current = ""
    for para in text.split("\n\n"):
        if len(current) + len(para) + 2 > limit and current:
            pieces.append(current.strip())
            current = ""
        if len(para) > limit:
            for i in range(0, len(para), limit):
                pieces.append(para[i : i + limit].strip())
            continue
        current = f"{current}\n\n{para}" if current else para
    if current.strip():
        pieces.append(current.strip())
    return [p for p in pieces if p]


def parse_paper(pdf_path: Path, max_pages: int | None = 200) -> list[Section]:
    """Parse a PDF into titled sections, each capped at EPISODE_MAX_CHARS."""

    pdf_path = Path(pdf_path)
    file_hash = sha256(pdf_path.read_bytes()).hexdigest()
    source = str(pdf_path)

    with pymupdf.open(pdf_path) as doc:
        pages = list(range(min(doc.page_count, max_pages or doc.page_count)))
        per_page = [(p, _lines_with_sizes(doc[p])) for p in pages]
        flat = [ln for _, lines in per_page for ln in lines]
        body = _body_size(flat)
        title = _paper_title(doc, body)

    buckets: list[dict] = []
    current: dict | None = None
    for page_no, lines in per_page:
        for text, size, bold in lines:
            if _is_header(text, size, bold, body):
                current = {"section": text.strip(), "lines": [], "pages": {page_no}}
                buckets.append(current)
                continue
            if current is None:
                current = {"section": "Front matter", "lines": [], "pages": {page_no}}
                buckets.append(current)
            current["lines"].append(text)
            current["pages"].add(page_no)

    named = [b for b in buckets if b["lines"]]
    if len(named) < 2:
        # Heuristic missed: degrade to one section per page rather than lose text.
        named = [
            {"section": f"Page {p + 1}", "lines": [t for t, _, _ in lines], "pages": {p}}
            for p, lines in per_page
            if lines
        ]

    sections: list[Section] = []
    for bucket in named:
        is_ref = bool(_REFERENCE_SECTION.match(bucket["section"]))
        blocks = build_blocks(bucket["lines"], is_ref)
        text = "\n\n".join(b.text for b in blocks).strip()
        if not text:
            continue

        # Episodes are capped for extraction quality; blocks ride with the first
        # piece since Graphiti only consumes `text`.
        for i, part in enumerate(_split_long(text, EPISODE_MAX_CHARS)):
            parts_name = bucket["section"]
            sections.append(
                Section(
                    paper_title=title,
                    section=parts_name if i == 0 and len(text) <= EPISODE_MAX_CHARS
                    else f"{parts_name} ({i + 1})",
                    text=part,
                    source=source,
                    file_hash=file_hash,
                    pages=sorted(bucket["pages"]),
                    blocks=blocks if i == 0 else [],
                )
            )
        # Re-attach blocks so every piece of a split section can still be chunked.
        if len(text) > EPISODE_MAX_CHARS:
            _redistribute_blocks(sections, blocks, len(_split_long(text, EPISODE_MAX_CHARS)))
    return sections


def _redistribute_blocks(sections: list[Section], blocks: list[Block], n_parts: int) -> None:
    """Spread a long section's blocks across the episode-sized pieces it became."""
    targets = sections[-n_parts:]
    if not targets:
        return
    per = max(1, len(blocks) // len(targets) + (1 if len(blocks) % len(targets) else 0))
    for i, sec in enumerate(targets):
        sec.blocks = blocks[i * per : (i + 1) * per]


def section_to_chunks(section: Section, chunk_size: int, chunk_overlap: int) -> list[dict]:
    """Turn a section's blocks into Chroma chunks.

    Prose is split normally. Atomic blocks (table/figure/algorithm) emit exactly
    one chunk each so a caption is never separated from its numbers.

    The embedded text carries a semantic header ("<title> — <section>"), unlike
    the old `[path#chunkN]` prefix which put a file path into every vector. The
    header is stripped again for BM25 (`retrieval.body_of`).
    """
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    base_meta = {
        "source": section.source,
        "file_hash": section.file_hash,
        "section": section.section,
        "paper_title": section.paper_title,
        "page": section.pages[0] if section.pages else 0,
    }
    blocks = section.blocks or [Block(PROSE, section.text)]

    out: list[dict] = []
    for block in blocks:
        if not block.text.strip():
            continue
        label = "" if block.content_type == PROSE else f" — {block.content_type.title()}"
        header = f"{section.paper_title} — {section.section}{label}"
        pieces = [block.text] if block.atomic else splitter.split_text(block.text)
        for piece in pieces:
            out.append(
                {
                    "text": f"{header}\n\n{piece}",
                    "metadata": {**base_meta, "content_type": block.content_type},
                }
            )
    return out


__all__ = [
    "BLOCK_MAX_CHARS",
    "Block",
    "EPISODE_MAX_CHARS",
    "MATH",
    "PROSE",
    "REFERENCE",
    "Section",
    "build_blocks",
    "parse_paper",
    "reflow",
    "section_to_chunks",
]
