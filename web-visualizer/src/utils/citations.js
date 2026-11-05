import { marked } from "marked";
import katex from "katex";

// Mirrors CITATION_RE in src/pipeline/verification.py — keep the two in sync.
const CITE_RE = /\[([A-Za-z0-9._-]+chunk-\d+)(?:\s+"[^"]{0,400}")?\]/g;

export const shortSource = (source = "") => source.split("/").pop().replace(/\.pdf$/i, "");
export const paperName = (meta = {}) => meta.paper_title || shortSource(meta.source);

const escapeHtml = (s) =>
  s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");

/** Pull LaTeX out before escaping/markdown mangle it; restored after marked. */
const extractMath = (text) => {
  const spans = [];
  const stash = (src, display) => {
    spans.push({ src, display });
    return `%%MATH${spans.length - 1}%%`;
  };
  // Display forms first so $$...$$ isn't eaten by the inline rule. Inline
  // $...$ requires non-space at both ends, so "$5 and $10" is left alone.
  const out = text
    .replace(/\$\$([\s\S]+?)\$\$/g, (_, src) => stash(src, true))
    .replace(/\\\[([\s\S]+?)\\\]/g, (_, src) => stash(src, true))
    .replace(/\\\(([\s\S]+?)\\\)/g, (_, src) => stash(src, false))
    .replace(/\$(\S(?:[^$\n]*\S)?)\$/g, (_, src) => stash(src, false));
  return { out, spans };
};

/* Wrap flagged claim sentences in a verdict span. Escape-then-indexOf means
 * exact-match-or-skip; claims spanning markdown blocks (contain \n) or whose
 * text was altered by math extraction silently get no underline — they still
 * appear in the flagged list below the answer.
 * ponytail: inline-only marking; a markdown-AST walk is the upgrade path. */
const wrapClaims = (escaped, claims) => {
  let out = escaped;
  for (const c of claims) {
    if (!["partial", "unsupported", "fabricated"].includes(c.verdict)) continue;
    const target = escapeHtml(c.text || "");
    if (!target || target.includes("\n") || !out.includes(target)) continue;
    const title = escapeHtml(`${c.verdict}${c.reason ? `: ${c.reason}` : ""}`)
      .replace(/"/g, "&quot;");
    out = out.replace(
      target,
      `<span class="claim ${c.verdict}" title="${title}">${target}</span>`
    );
  }
  return out;
};

/** Render an answer to HTML: markdown + KaTeX, with [chunk-id] / [chunk-id "quote"]
 * citations replaced by numbered <button class="cite"> marks and flagged claim
 * sentences wrapped in verdict spans. Raw text is entity-escaped first so any
 * HTML in the model output renders as text. */
export const renderAnswerHtml = (answer, matches, claims = []) => {
  const known = new Set(matches.map((m) => (m.metadata || {}).id).filter(Boolean));
  const order = [];

  const { out: demathed, spans } = extractMath(answer);
  const escaped = wrapClaims(escapeHtml(demathed), claims);

  const withCites = escaped.replace(CITE_RE, (full, id) => {
    if (!known.has(id)) return full;
    if (!order.includes(id)) order.push(id);
    const n = order.indexOf(id) + 1;
    return `<button type="button" class="cite" data-cite="${id}" title="${id}">${n}</button>`;
  });

  const html = marked
    .parse(withCites, { breaks: true })
    .replace(/%%MATH(\d+)%%/g, (full, n) => {
      const m = spans[Number(n)];
      return m
        ? katex.renderToString(m.src, { displayMode: m.display, throwOnError: false })
        : full;
    });

  return { html, order };
};

/** Split text into ordered {text, hit} segments from claim spans (chunk-local
 * offsets). Overlaps merge; out-of-range offsets clamp. */
export const highlightSegments = (text, spans = []) => {
  const sorted = spans
    .map((s) => ({
      start: Math.max(0, s.start ?? 0),
      end: Math.min(text.length, s.end ?? 0),
    }))
    .filter((s) => s.end > s.start)
    .sort((a, b) => a.start - b.start);

  const merged = [];
  for (const s of sorted) {
    const last = merged[merged.length - 1];
    if (last && s.start <= last.end) last.end = Math.max(last.end, s.end);
    else merged.push({ ...s });
  }

  const segments = [];
  let pos = 0;
  for (const s of merged) {
    if (s.start > pos) segments.push({ text: text.slice(pos, s.start), hit: false });
    segments.push({ text: text.slice(s.start, s.end), hit: true });
    pos = s.end;
  }
  if (pos < text.length || !segments.length)
    segments.push({ text: text.slice(pos), hit: false });
  return segments;
};

/** Papers whose retrieved passages contradict the main line, one entry per
 * paper: {paper, claim, year}. Empty on the (common) unanimous answer. */
export const disagreements = (stances = {}, matches = []) => {
  const byPaper = new Map();
  for (const m of matches) {
    const meta = m.metadata || {};
    const s = stances[meta.id];
    if (s?.stance === "contradicts" && !byPaper.has(paperName(meta)))
      byPaper.set(paperName(meta), { claim: s.claim || "", year: meta.pub_year || null });
  }
  return [...byPaper].map(([paper, v]) => ({ paper, ...v }));
};

// ponytail: heuristic — an arXiv id plus a non-arXiv venue means the local
// file is the arXiv copy of published work; true v1-vs-vN comparison needs
// version capture in scholarly.extract_ids.
export const preprintNote = (meta = {}) =>
  meta.arxiv_id && meta.venue && !/arxiv/i.test(meta.venue)
    ? `preprint · published: ${meta.venue}${meta.pub_year ? ` ${meta.pub_year}` : ""}`
    : null;
