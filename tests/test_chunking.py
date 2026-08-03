import unittest
from pathlib import Path

from src.pipeline.ingestion import build_chunk_records
from src.pipeline.parsing import (
    MATH,
    PROSE,
    Block,
    Section,
    _is_mathy,
    build_blocks,
    parse_paper,
    reflow,
    splice_hyphens,
)
from src.pipeline.settings import PipelineSettings

REPO_ROOT = Path(__file__).resolve().parents[1]
PDF = REPO_ROOT / "data" / "1706.03762v7.pdf"


class TestReflow(unittest.TestCase):
    def test_rejoins_words_broken_across_lines(self):
        # 1051 of these across the corpus; BM25 cannot match either half.
        out = reflow(["without modifying the computa-", "tional cost of the layer."])
        self.assertIn("computational", out)
        self.assertNotIn("computa-", out)

    def test_does_not_join_a_real_trailing_hyphen(self):
        # "end-to-end" style: next line starts uppercase, so it is not a split word.
        out = reflow(["we use a state-of-the-", "Art model"])
        self.assertNotIn("state-of-theArt", out)

    def test_joins_wrapped_lines_into_one_paragraph(self):
        out = reflow(["The model computes attention over", "all positions in the sequence."])
        self.assertEqual(out, "The model computes attention over all positions in the sequence.")

    def test_starts_a_new_paragraph_after_a_sentence_ends(self):
        out = reflow(["This ends here.", "A new thought begins."])
        self.assertEqual(out.count("\n\n"), 1)

    def test_splice_hyphens_preserves_line_structure(self):
        # Table rows must keep their lines even while words are repaired.
        rows = splice_hyphens(["Model  BLEU  train-", "ing cost", "Base  27.3  3.3"])
        self.assertEqual(len(rows), 2)
        self.assertIn("training cost", rows[0])


class TestMathDetection(unittest.TestCase):
    def test_ordinary_short_sentences_are_not_math(self):
        # Regression: an earlier version called this math and shredded the
        # surrounding paragraph into fragments.
        self.assertFalse(_is_mathy("We use N = 6 layers"))
        self.assertFalse(_is_mathy("and apply a softmax function to obtain"))

    def test_symbol_dense_lines_are_math(self):
        self.assertTrue(_is_mathy("θt+1,i −θ∗,i = αt · ∑ βt−j"))

    def test_short_math_run_stays_inside_the_paragraph(self):
        blocks = build_blocks(
            ["The update rule is defined as follows.", "x = y + z", "This keeps training stable."],
            is_reference=False,
        )
        self.assertEqual([b.content_type for b in blocks], [PROSE])

    def test_sustained_math_run_becomes_its_own_block(self):
        # Realistic display-equation lengths: a handful of ~70-char fraction
        # lines, as the Adam appendix produces. A *short* run is deliberately
        # merged back into the paragraph instead (see _merge_small).
        equation = "θ∗t+1,i = θ∗t,i − αt · ∑j βt−j 2 g2 j,i / √(1 − βt 2) ≤ ∥g1:t,i∥2"
        lines = ["Consider the bound below."] + [equation] * 5
        kinds = [b.content_type for b in build_blocks(lines, is_reference=False)]
        self.assertIn(MATH, kinds)

    def test_tiny_math_run_is_folded_back_into_prose(self):
        blocks = build_blocks(
            ["Consider the bound below."] + ["x = y"] * 4, is_reference=False
        )
        self.assertEqual([b.content_type for b in blocks], [PROSE])


class TestAtomicBlocks(unittest.TestCase):
    def test_caption_opens_a_table_block_that_keeps_its_rows(self):
        blocks = build_blocks(
            ["Table 2: BLEU scores.", "Model EN-DE EN-FR", "Transformer 28.4 41.8"],
            is_reference=False,
        )
        table = [b for b in blocks if b.content_type == "table"]
        self.assertEqual(len(table), 1)
        self.assertIn("Table 2", table[0].text)
        self.assertIn("28.4", table[0].text)

    def test_atomic_block_is_never_split_by_the_chunker(self):
        section = Section(
            paper_title="P", section="Results", text="", source="data/p.pdf", file_hash="h" * 64,
            blocks=[Block("table", "Table 2: results.\n" + "row 1.23 4.56\n" * 200)],
        )
        from src.pipeline.parsing import section_to_chunks

        chunks = section_to_chunks(section, chunk_size=200, chunk_overlap=20)
        self.assertEqual(len(chunks), 1, "a table must not be split across chunks")
        self.assertIn("Table 2", chunks[0]["text"])

    def test_references_are_splittable_not_atomic(self):
        # Regression: treating references as atomic produced a 26k-char chunk.
        self.assertFalse(Block("reference", "x").atomic)
        self.assertTrue(Block("table", "x").atomic)


class TestChunkRecords(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not PDF.exists():
            raise unittest.SkipTest("sample PDF not present")
        cls.settings = PipelineSettings()
        cls.records = build_chunk_records(parse_paper(PDF), cls.settings)

    def test_neighbour_ids_form_a_chain(self):
        ids = [r["metadata"]["id"] for r in self.records]
        self.assertGreater(len(ids), 2)
        self.assertNotIn("prev_id", self.records[0]["metadata"])
        self.assertEqual(self.records[0]["metadata"]["next_id"], ids[1])
        self.assertEqual(self.records[1]["metadata"]["prev_id"], ids[0])
        self.assertNotIn("next_id", self.records[-1]["metadata"])

    def test_references_are_excluded_by_default(self):
        sections = [s.section.lower() for s in parse_paper(PDF)]
        self.assertTrue(any("reference" in s for s in sections), "corpus should have references")
        kinds = {r["metadata"]["content_type"] for r in self.records}
        self.assertNotIn("reference", kinds)

    def test_references_come_back_when_enabled(self):
        on = build_chunk_records(parse_paper(PDF), PipelineSettings(index_references=True))
        self.assertGreater(len(on), len(self.records))

    def test_no_hyphen_breaks_survive(self):
        import re

        broken = sum(len(re.findall(r"[a-z]-\n[a-z]", r["text"])) for r in self.records)
        self.assertEqual(broken, 0)

    def test_every_chunk_carries_a_content_type(self):
        for r in self.records:
            self.assertIn("content_type", r["metadata"])


class TestQuotaHandling(unittest.TestCase):
    """Retry per-minute throttling; fail fast on a per-day cap."""

    def test_per_minute_limit_is_retried(self):
        from src.pipeline.ingestion import _is_rate_limit

        self.assertTrue(_is_rate_limit(Exception("429 RESOURCE_EXHAUSTED PerMinute")))

    def test_per_day_limit_is_not_retried(self):
        # A daily bucket does not refill on a 90s backoff; retrying burns
        # wall-clock and a few more requests before failing anyway.
        from src.pipeline.ingestion import _is_rate_limit

        self.assertFalse(_is_rate_limit(Exception("429 ... PerDayPerUserPerProject ...")))

    def test_daily_exhaustion_raises_a_clear_error(self):
        from src.pipeline.ingestion import DailyQuotaExhausted, _add_batch

        class Boom:
            def add_texts(self, **kw):
                raise RuntimeError("429 quotaId EmbedContentRequestsPerDayPerUser")

        with self.assertRaises(DailyQuotaExhausted) as ctx:
            _add_batch(Boom(), ["t"], [{}], ["i"])
        self.assertIn("re-run", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
