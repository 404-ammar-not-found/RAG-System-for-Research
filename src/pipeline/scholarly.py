from __future__ import annotations

"""Bibliographic metadata: identity, retraction status, and citation edges.

The usual advice is "OpenAlex and Semantic Scholar give you references and
citers for free". The metadata is free; *resolution* is not. Measured against
this corpus:

  OpenAlex by arXiv DOI (10.48550/arXiv.…)  404
  Crossref by the same DOI                   404
  OpenAlex title search "Attention Is All
    You Need"                                returns a 2025 paper, not the 2017 one
  Semantic Scholar by arXiv:id               429 without an API key
  arXiv Atom API                             authoritative, no key, reliable

So this resolves in stages, cheapest and most reliable first, and degrades to
partial metadata rather than failing. Everything is cached on disk: these are
public APIs and a re-ingest should not re-hammer them.
"""

import json
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
CACHE_PATH = REPO_ROOT / "cache" / "scholarly.json"

ARXIV_RE = re.compile(r"arxiv[:\s]*(\d{4}\.\d{4,5})", re.I)
DOI_RE = re.compile(r"\b(10\.\d{4,9}/[-._;()/:A-Za-z0-9]+)\b")
TIMEOUT = 25


def _mailto() -> str | None:
    """Contact address for the OpenAlex/Crossref polite pool, if configured."""
    return os.getenv("OPENALEX_MAILTO") or None


def _agent() -> str:
    contact = _mailto()
    return f"research-rag/0.3 ({'mailto:' + contact if contact else 'no contact'})"


def _get(url: str, *, retries: int = 2) -> Any:
    """GET JSON, returning None on any failure. Never raises into the pipeline."""
    for attempt in range(retries + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": _agent()})
            with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
                return json.loads(resp.read().decode())
        except urllib.error.HTTPError as exc:
            # 429 is the normal state of the Semantic Scholar free tier.
            if exc.code == 429 and attempt < retries:
                time.sleep(3 * (attempt + 1))
                continue
            return None
        except Exception:
            return None
    return None


def _get_text(url: str) -> str | None:
    try:
        req = urllib.request.Request(url, headers={"User-Agent": _agent()})
        with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
            return resp.read().decode()
    except Exception:
        return None


# ---------------------------------------------------------------------------


@dataclass
class PaperRecord:
    """What we know about a paper beyond its text."""

    key: str
    title: str = ""
    year: int | None = None
    venue: str | None = None
    doi: str | None = None
    arxiv_id: str | None = None
    arxiv_doi: str | None = None
    citation_count: int | None = None
    is_retracted: bool = False
    retraction_note: str | None = None
    reference_ids: list[str] = field(default_factory=list)
    citers: list[dict[str, Any]] = field(default_factory=list)
    resolved_by: list[str] = field(default_factory=list)
    unresolved: bool = False
    fetched_at: str = ""

    def as_chunk_metadata(self) -> dict[str, Any]:
        """Chroma metadata must be scalars, so flatten and drop the lists."""
        return {
            "pub_year": self.year or 0,
            "venue": self.venue or "",
            "doi": self.doi or "",
            "arxiv_id": self.arxiv_id or "",
            "arxiv_doi": self.arxiv_doi or "",
            "citation_count": self.citation_count or 0,
            "is_retracted": bool(self.is_retracted),
        }


# Both id schemes: new-style `1706.03762`, pre-2007 `cs/0501001`, optional `v3`.
_ID = r"(\d{4}\.\d{4,5}|[a-z-]+(?:\.[A-Z]{2})?/\d{7})(v\d+)?"
ARXIV_LINK_RE = re.compile(rf"(?:arxiv\.org/(?:abs|pdf|html)/)?{_ID}", re.I)


def parse_arxiv_id(text: str) -> str | None:
    """Pull an arXiv id out of a link, an `arXiv:` string, or a bare id."""
    match = ARXIV_LINK_RE.search(text.strip())
    return match.group(1) + (match.group(2) or "") if match else None


def download_arxiv_pdf(arxiv_id: str) -> bytes:
    """Fetch a paper's PDF. Raises unless arXiv actually returns a PDF."""
    req = urllib.request.Request(
        f"https://arxiv.org/pdf/{arxiv_id}", headers={"User-Agent": _agent()}
    )
    with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
        data = resp.read()
    if not data.startswith(b"%PDF"):
        raise ValueError(f"arXiv returned no PDF for {arxiv_id}")
    return data


def extract_ids(pdf_path: Path, pages: int = 2) -> dict[str, str | None]:
    """Pull an arXiv id or DOI off the opening pages, or the filename."""
    import pymupdf

    try:
        with pymupdf.open(pdf_path) as doc:
            head = "\n".join(doc[i].get_text() for i in range(min(pages, doc.page_count)))
    except Exception:
        head = ""

    arxiv = ARXIV_RE.search(head)
    arxiv_id = arxiv.group(1) if arxiv else None
    if not arxiv_id:
        from_name = re.match(r"^(\d{4}\.\d{4,5})", Path(pdf_path).stem)
        arxiv_id = from_name.group(1) if from_name else None

    doi_match = DOI_RE.search(head)
    doi = doi_match.group(1).rstrip(".,;)") if doi_match else None
    # arXiv's own DOI is not registered with Crossref or OpenAlex; it resolves
    # nowhere and is worse than having no DOI at all.
    if doi and "10.48550" in doi:
        doi = None
    return {"arxiv_id": arxiv_id, "doi": doi}


# ------------------------------------------------------------------ sources


def _norm_title(title: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", title.lower()).strip()


def _title_close(a: str, b: str, threshold: float = 0.92) -> bool:
    """Guard against near-miss titles.

    An arXiv title search for "Attention Is All You Need" also returns
    "Not All Attention Is All You Need" and "Tensor Product Attention Is All
    You Need" — accepting any hit would attach the wrong paper's metadata.
    """
    from difflib import SequenceMatcher

    na, nb = _norm_title(a), _norm_title(b)
    if not na or not nb:
        return False
    return na == nb or SequenceMatcher(None, na, nb).ratio() >= threshold


def find_arxiv_by_title(title: str) -> str | None:
    """Recover an arXiv id from a title alone.

    The identifier is the key to everything else — Semantic Scholar resolves
    arXiv ids directly, and arXiv's own API is authoritative for the metadata —
    so a paper whose PDF hides its id is worth one search to rescue.
    """
    if not title:
        return None
    query = urllib.parse.quote(f'ti:"{title}"')
    xml = _get_text(
        f"http://export.arxiv.org/api/query?search_query={query}&max_results=5"
    )
    if not xml:
        return None
    for entry in re.findall(r"<entry>(.*?)</entry>", xml, re.S):
        found = re.search(r"<title>\s*(.*?)\s*</title>", entry, re.S)
        ident = re.search(r"<id>https?://arxiv\.org/abs/([^v<]+)", entry)
        if found and ident and _title_close(title, " ".join(found.group(1).split())):
            return ident.group(1)
    return None


def from_arxiv(arxiv_id: str, record: PaperRecord) -> bool:
    """Authoritative for arXiv papers, no key required."""
    xml = _get_text(f"http://export.arxiv.org/api/query?id_list={arxiv_id}")
    if not xml or "<entry>" not in xml:
        return False
    entry = xml[xml.find("<entry>") :]

    title = re.search(r"<title>([^<]+)</title>", entry)
    published = re.search(r"<published>(\d{4})-", entry)
    journal = re.search(r"<arxiv:journal_ref[^>]*>([^<]+)<", entry)
    doi = re.search(r"<arxiv:doi[^>]*>([^<]+)<", entry)

    if title:
        record.title = " ".join(title.group(1).split())
    if published:
        record.year = int(published.group(1))
    if journal:
        record.venue = " ".join(journal.group(1).split())
    if doi and not record.doi:
        record.doi = doi.group(1).strip()
    record.arxiv_id = arxiv_id
    return True


def from_openalex(record: PaperRecord) -> bool:
    """Citation counts, retraction flag and reference edges.

    Title search is ambiguous — several papers now share famous titles — so a
    candidate must agree on publication year before it is accepted.
    """
    contact = f"&mailto={urllib.parse.quote(_mailto())}" if _mailto() else ""

    if record.doi:
        data = _get(f"https://api.openalex.org/works/doi:{record.doi}?{contact.lstrip('&')}")
        if data and data.get("id"):
            _apply_openalex(record, data)
            return True

    if not record.title:
        return False

    query = urllib.parse.quote(f'"{record.title}"')
    data = _get(
        f"https://api.openalex.org/works?filter=title.search:{query}"
        f"&sort=cited_by_count:desc&per-page=5{contact}"
    )
    candidates = [
        c
        for c in ((data or {}).get("results") or [])
        # Same title, different paper: "Attention Is All You Need" title-matches
        # a 2025 work before the 2017 one.
        if not (record.year and c.get("publication_year")
                and abs(c["publication_year"] - record.year) > 1)
    ]
    if not candidates:
        return False

    # With no year to check against, prefer the earliest publication. Famous
    # papers get reprinted — AlexNet's CACM reissue outranks the 2012 NeurIPS
    # original on citation count, and dating it 2017 would be wrong.
    if not record.year:
        candidates.sort(key=lambda c: c.get("publication_year") or 9999)
    _apply_openalex(record, candidates[0])
    return True


def _apply_openalex(record: PaperRecord, work: dict[str, Any]) -> None:
    record.title = record.title or (work.get("title") or "")
    record.year = record.year or work.get("publication_year")
    record.citation_count = work.get("cited_by_count")
    if work.get("is_retracted"):
        record.is_retracted = True
        record.retraction_note = "OpenAlex reports this work as retracted."
    source = (work.get("primary_location") or {}).get("source") or {}
    record.venue = record.venue or source.get("display_name")
    record.reference_ids = list(work.get("referenced_works") or [])[:200]
    doi = (work.get("ids") or {}).get("doi")
    if doi and not record.doi:
        record.doi = doi.replace("https://doi.org/", "")
    record.resolved_by.append("openalex")


def check_retraction(record: PaperRecord) -> None:
    """Crossref carries the Retraction Watch database; absence of a DOI is not
    evidence of good standing, so that case is reported as unknown, not clean."""
    if not record.doi:
        return
    contact = f"?mailto={urllib.parse.quote(_mailto())}" if _mailto() else ""
    data = _get(f"https://api.crossref.org/works/{urllib.parse.quote(record.doi)}{contact}")
    message = (data or {}).get("message") or {}
    if not message:
        return
    record.resolved_by.append("crossref")
    if message.get("type") == "retraction" or message.get("update-to"):
        record.is_retracted = True
        record.retraction_note = "Crossref records a retraction notice for this DOI."
    for update in message.get("updated-by") or []:
        if "retract" in str(update.get("type", "")).lower():
            record.is_retracted = True
            record.retraction_note = f"Retracted by {update.get('DOI')}."


def _s2_url(path: str, params: str) -> str:
    key = os.getenv("SEMANTIC_SCHOLAR_KEY")
    url = f"https://api.semanticscholar.org/graph/v1/paper/{path}?{params}"
    return url + (f"&x-api-key={key}" if key else "")


def from_semantic_scholar(record: PaperRecord) -> bool:
    """Primary source for arXiv-native papers.

    OpenAlex has no title-searchable record for several of these — a
    year-filtered search for "attention is all you need" in 2017 returns
    count=0 — while Semantic Scholar resolves an arXiv id directly. The free
    tier rate-limits hard, so this is retried with backoff and cached.
    """
    ident = None
    if record.arxiv_id:
        ident = f"arXiv:{record.arxiv_id}"
    elif record.doi:
        ident = f"DOI:{record.doi}"
    if not ident:
        return False

    data = _get(
        _s2_url(
            urllib.parse.quote(ident, safe=":"),
            "fields=title,year,venue,citationCount,referenceCount,externalIds,isOpenAccess",
        ),
        retries=3,
    )
    if not data or not data.get("title"):
        return False

    record.title = record.title or data["title"]
    record.year = record.year or data.get("year")
    record.venue = record.venue or (data.get("venue") or None)
    if record.citation_count is None:
        record.citation_count = data.get("citationCount")
    external = data.get("externalIds") or {}
    if not record.doi and external.get("DOI"):
        record.doi = external["DOI"]
    record.resolved_by.append("semanticscholar")
    return True


def fetch_citers(record: PaperRecord, limit: int = 10) -> None:
    """Forward citations — the papers that cite this one.

    This is where a 2023 refutation of a 2019 finding comes from. Semantic
    Scholar rate-limits the free tier aggressively, so a miss here is normal
    and must not fail anything.
    """
    ident = f"arXiv:{record.arxiv_id}" if record.arxiv_id else record.doi
    if not ident:
        return
    key = os.getenv("SEMANTIC_SCHOLAR_KEY")
    url = (
        f"https://api.semanticscholar.org/graph/v1/paper/{urllib.parse.quote(ident, safe=':')}"
        f"/citations?fields=title,year,externalIds&limit={limit}"
    )
    if key:
        url += f"&x-api-key={key}"
    data = _get(url)
    for item in (data or {}).get("data") or []:
        paper = item.get("citingPaper") or {}
        if paper.get("title"):
            record.citers.append(
                {"title": paper["title"], "year": paper.get("year")}
            )
    if record.citers:
        record.resolved_by.append("semanticscholar")


# ------------------------------------------------------------------- cache


def _load_cache() -> dict[str, Any]:
    if CACHE_PATH.exists():
        try:
            return json.loads(CACHE_PATH.read_text())
        except Exception:
            return {}
    return {}


def _save_cache(cache: dict[str, Any]) -> None:
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    CACHE_PATH.write_text(json.dumps(cache, indent=2))


def _resolve_online(record: PaperRecord) -> None:
    """Staged resolution, most authoritative source first.

    The order is dictated by what actually works, not by what the docs suggest:

    1. No arXiv id? Search arXiv by title to recover one. The id is the key that
       unlocks the rest, so it is worth one request.
    2. arXiv API for title/year/venue — authoritative, no key, never rate-limited.
    3. Semantic Scholar by arXiv id or DOI. This is the primary for arXiv-native
       papers because OpenAlex has no title-searchable record for several of
       them, and it is also the only source of forward citations here.
    4. OpenAlex for citation counts, references and the retraction flag. Reliable
       for published journal work (CNNpred, AlexNet), unreliable for preprints.
    5. Crossref for retraction notices, which needs a real (non-arXiv) DOI.
    """
    if not record.arxiv_id and record.title:
        found = find_arxiv_by_title(record.title)
        if found:
            record.arxiv_id = found
            record.resolved_by.append("arxiv-title-search")

    if record.arxiv_id:
        if from_arxiv(record.arxiv_id, record):
            record.resolved_by.append("arxiv")
        # Recorded for citation display. Note it resolves at neither OpenAlex
        # nor Crossref — both 404 on 10.48550 DOIs — so it is an identifier,
        # not a lookup key.
        record.arxiv_doi = f"10.48550/arXiv.{record.arxiv_id}"

    from_semantic_scholar(record)
    from_openalex(record)
    check_retraction(record)
    fetch_citers(record)


def resolve_paper(
    pdf_path: Path, title_hint: str = "", *, refresh: bool = False, online: bool = True
) -> PaperRecord:
    """Resolve one paper's bibliographic record, cached on disk."""
    pdf_path = Path(pdf_path)
    ids = extract_ids(pdf_path)
    key = ids["arxiv_id"] or ids["doi"] or pdf_path.stem

    cache = _load_cache()
    if not refresh and key in cache:
        return PaperRecord(**cache[key])

    record = PaperRecord(key=key, arxiv_id=ids["arxiv_id"], doi=ids["doi"], title=title_hint)
    if online:
        _resolve_online(record)

    # A source can contribute twice (metadata, then citers); report it once.
    record.resolved_by = list(dict.fromkeys(record.resolved_by))
    record.unresolved = not record.resolved_by
    record.fetched_at = datetime.now(timezone.utc).isoformat(timespec="seconds")

    cache[key] = asdict(record)
    _save_cache(cache)
    return record


def warn_lines(record: PaperRecord) -> list[str]:
    """Cautions worth putting in front of the model and the reader."""
    out: list[str] = []
    if record.is_retracted:
        out.append(f"RETRACTED: {record.retraction_note or 'this work has been retracted'}")
    if not record.doi and not record.arxiv_id:
        out.append("No DOI or arXiv id found — retraction status could not be checked.")
    return out


__all__ = [
    "PaperRecord",
    "find_arxiv_by_title",
    "from_semantic_scholar",
    "check_retraction",
    "extract_ids",
    "fetch_citers",
    "from_arxiv",
    "from_openalex",
    "resolve_paper",
    "warn_lines",
]
