import unittest

from src.pipeline.qa import _strip_unknown_citations
from src.pipeline.retrieval import Passage, rrf_fuse


def p(pid):
    return Passage(id=pid, text=f"text-{pid}", metadata={"id": pid})


class TestRrfFuse(unittest.TestCase):
    def test_agreement_across_channels_beats_a_single_strong_hit(self):
        # "b" is #1 in dense only; "a" is #2 in both. Agreement should win.
        dense = [p("b"), p("a"), p("c")]
        lexical = [p("d"), p("a"), p("e")]

        ranked = rrf_fuse(("dense", dense), ("bm25", lexical))
        order = [x.id for x in ranked]
        self.assertEqual(order[0], "a")
        self.assertLess(order.index("a"), order.index("b"))

    def test_records_which_channels_found_each_passage(self):
        ranked = rrf_fuse(("dense", [p("a")]), ("bm25", [p("a"), p("b")]))
        by_id = {x.id: x for x in ranked}
        self.assertEqual(sorted(by_id["a"].channels), ["bm25", "dense"])
        self.assertEqual(by_id["b"].channels, ["bm25"])

    def test_deduplicates_and_scores_descending(self):
        ranked = rrf_fuse(("dense", [p("a"), p("b")]), ("bm25", [p("b"), p("a")]))
        self.assertEqual(len(ranked), 2)
        self.assertGreaterEqual(ranked[0].score, ranked[1].score)

    def test_empty_channels_are_harmless(self):
        self.assertEqual(rrf_fuse(("dense", []), ("bm25", [])), [])


class TestCitationValidation(unittest.TestCase):
    def test_keeps_real_ids_and_drops_invented_ones(self):
        answer = "Dropout helps [paper-abc12345-chunk-2] and so does data augmentation [fake-99999999-chunk-7]."
        cleaned, used = _strip_unknown_citations(answer, {"paper-abc12345-chunk-2"})
        self.assertIn("[paper-abc12345-chunk-2]", cleaned)
        self.assertNotIn("fake-99999999-chunk-7", cleaned)
        self.assertEqual(used, ["paper-abc12345-chunk-2"])

    def test_reports_each_used_id_once(self):
        answer = "[a-11111111-chunk-0] and again [a-11111111-chunk-0]"
        _, used = _strip_unknown_citations(answer, {"a-11111111-chunk-0"})
        self.assertEqual(used, ["a-11111111-chunk-0"])


class TestBm25Index(unittest.TestCase):
    def test_lexical_match_ranks_first(self):
        from src.pipeline.retrieval import Bm25Index

        class FakeCollection:
            def get(self, include=None):
                return {
                    "ids": ["c0", "c1", "c2"],
                    "documents": [
                        "dropout regularization prevents overfitting",
                        "convolutional kernels stride pooling",
                        "machine translation with attention",
                    ],
                    "metadatas": [{"id": "c0"}, {"id": "c1"}, {"id": "c2"}],
                }

        class FakeStore:
            _collection = FakeCollection()

        index = Bm25Index()
        self.assertEqual(index.build(FakeStore()), 3)
        hits = index.search("dropout overfitting", k=3)
        self.assertTrue(hits)
        self.assertEqual(hits[0].id, "c0")

    def test_no_match_returns_nothing_rather_than_noise(self):
        from src.pipeline.retrieval import Bm25Index

        index = Bm25Index()
        self.assertEqual(index.search("anything", k=5), [])


class TestListwiseRerank(unittest.TestCase):
    """One LLM call, and never fails the question when the model misbehaves."""

    def _run(self, reply):
        import asyncio
        from src.pipeline.retrieval import rerank

        calls = []

        class FakeLLM:
            async def ainvoke(self, prompt):
                calls.append(prompt)
                if isinstance(reply, Exception):
                    raise reply
                return type("M", (), {"content": reply})()

        got = asyncio.run(rerank(FakeLLM(), "q", [p("a"), p("b"), p("c")], limit=2))
        return got, calls

    def test_reorders_by_model_output_in_a_single_call(self):
        got, calls = self._run("[3, 1, 2]")
        self.assertEqual([x.id for x in got], ["c", "a"])
        self.assertEqual(len(calls), 1, "must be listwise, not one call per passage")

    def test_tolerates_prose_around_the_array(self):
        got, _ = self._run("Sure! Here you go: [2, 1] — hope that helps.")
        self.assertEqual([x.id for x in got], ["b", "a"])

    def test_falls_back_to_rrf_order_when_the_model_errors(self):
        got, _ = self._run(RuntimeError("429 rate limit"))
        self.assertEqual([x.id for x in got], ["a", "b"])

    def test_ignores_out_of_range_and_duplicate_indices(self):
        got, _ = self._run("[99, 2, 2, 0, 1]")
        self.assertEqual([x.id for x in got], ["b", "a"])

    def test_tops_up_from_rrf_when_model_returns_too_few(self):
        got, _ = self._run("[3]")
        self.assertEqual(len(got), 2)
        self.assertEqual(got[0].id, "c")

    def test_garbage_reply_does_not_raise(self):
        got, _ = self._run("I cannot help with that.")
        self.assertEqual([x.id for x in got], ["a", "b"])


if __name__ == "__main__":
    unittest.main()
