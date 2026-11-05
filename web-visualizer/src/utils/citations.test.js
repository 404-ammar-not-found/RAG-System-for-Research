import { describe, expect, it } from "vitest";
import {
  disagreements,
  highlightSegments,
  preprintNote,
  renderAnswerHtml,
} from "./citations.js";

const matches = [
  { metadata: { id: "paper-ab12cd34-chunk-0" } },
  { metadata: { id: "paper-ab12cd34-chunk-3" } },
];

describe("renderAnswerHtml", () => {
  it("numbers citations in order of first use", () => {
    const { html, order } = renderAnswerHtml(
      "First [paper-ab12cd34-chunk-3] then [paper-ab12cd34-chunk-0] and again [paper-ab12cd34-chunk-3].",
      matches
    );
    expect(order).toEqual(["paper-ab12cd34-chunk-3", "paper-ab12cd34-chunk-0"]);
    expect(html.match(/data-cite="paper-ab12cd34-chunk-3"/g)).toHaveLength(2);
    expect(html).toMatch(/data-cite="paper-ab12cd34-chunk-0"[^>]*>2</);
  });

  it("matches the quote-bearing form the backend emits", () => {
    const { html, order } = renderAnswerHtml(
      'Attention scales [paper-ab12cd34-chunk-0 "we scale the dot products"].',
      matches
    );
    expect(order).toEqual(["paper-ab12cd34-chunk-0"]);
    expect(html).not.toContain("we scale the dot products");
  });

  it("renders markdown and escapes raw HTML", () => {
    const { html } = renderAnswerHtml("**bold** <script>alert(1)</script>", matches);
    expect(html).toContain("<strong>bold</strong>");
    expect(html).not.toContain("<script>");
  });

  it("leaves citations to unknown ids as text", () => {
    const { html, order } = renderAnswerHtml("See [nope-chunk-9].", matches);
    expect(order).toEqual([]);
    expect(html).toContain("[nope-chunk-9]");
  });

  it("wraps flagged claim sentences in a verdict span with the reason", () => {
    const claims = [
      { text: "The sky is green.", verdict: "unsupported", reason: "not in passage" },
      { text: "Water is wet.", verdict: "supported" },
    ];
    const { html } = renderAnswerHtml("The sky is green. Water is wet.", matches, claims);
    expect(html).toContain('class="claim unsupported"');
    expect(html).toContain("unsupported: not in passage");
    expect(html).not.toContain('class="claim supported"');
  });

  it("skips block-spanning claims instead of producing broken HTML", () => {
    const claims = [{ text: "line one\nline two", verdict: "fabricated" }];
    const { html } = renderAnswerHtml("line one\nline two", matches, claims);
    expect(html).not.toContain("claim fabricated");
  });

  it("matches claims containing escaped characters without injecting HTML", () => {
    const claims = [{ text: "a < b holds.", verdict: "partial", reason: 'see "table"' }];
    const { html } = renderAnswerHtml("a < b holds.", matches, claims);
    expect(html).toContain('class="claim partial"');
    expect(html).toContain("a &lt; b holds.");
  });

  it("renders display and inline math through KaTeX", () => {
    const { html } = renderAnswerHtml("Update rule: $$x^2 + y$$ and inline $e_i$.", matches);
    expect(html).toContain("katex");
    expect(html).not.toContain("%%MATH");
  });

  it("leaves currency alone", () => {
    const { html } = renderAnswerHtml("Costs $5 and $10 per run.", matches);
    expect(html).not.toContain("katex");
    expect(html).toContain("$5 and $10");
  });

  it("does not throw on malformed LaTeX", () => {
    const { html } = renderAnswerHtml("Broken: $$\\frac{1}{$$ end.", matches);
    expect(html).toContain("katex");
  });
});

describe("highlightSegments", () => {
  it("splits around a span", () => {
    expect(highlightSegments("abcdef", [{ start: 2, end: 4 }])).toEqual([
      { text: "ab", hit: false },
      { text: "cd", hit: true },
      { text: "ef", hit: false },
    ]);
  });

  it("merges overlapping spans", () => {
    const segs = highlightSegments("abcdef", [
      { start: 1, end: 3 },
      { start: 2, end: 5 },
    ]);
    expect(segs.filter((s) => s.hit)).toEqual([{ text: "bcde", hit: true }]);
  });

  it("clamps out-of-range offsets", () => {
    expect(highlightSegments("abc", [{ start: 1, end: 99 }])).toEqual([
      { text: "a", hit: false },
      { text: "bc", hit: true },
    ]);
  });

  it("returns one non-hit segment when there are no spans", () => {
    expect(highlightSegments("abc", [])).toEqual([{ text: "abc", hit: false }]);
  });
});

describe("disagreements", () => {
  const mk = (id, paper, year) => ({
    metadata: { id, paper_title: paper, pub_year: year },
  });

  it("returns empty when nothing contradicts", () => {
    const stances = { a: { stance: "supports", claim: "x" } };
    expect(disagreements(stances, [mk("a", "P1", 2020)])).toEqual([]);
  });

  it("dedupes to one entry per paper and carries year and claim", () => {
    const stances = {
      a: { stance: "contradicts", claim: "first claim" },
      b: { stance: "contradicts", claim: "second claim" },
    };
    const out = disagreements(stances, [mk("a", "P1", 2023), mk("b", "P1", 2023)]);
    expect(out).toEqual([{ paper: "P1", claim: "first claim", year: 2023 }]);
  });
});

describe("preprintNote", () => {
  it("flags an arXiv copy of published work", () => {
    expect(
      preprintNote({ arxiv_id: "1706.03762", venue: "NeurIPS", pub_year: 2017 })
    ).toBe("preprint · published: NeurIPS 2017");
  });

  it("stays quiet for arXiv-only venues and non-arXiv papers", () => {
    expect(preprintNote({ arxiv_id: "1", venue: "arXiv.org" })).toBeNull();
    expect(preprintNote({ venue: "NeurIPS" })).toBeNull();
    expect(preprintNote({ arxiv_id: "1" })).toBeNull();
  });
});
