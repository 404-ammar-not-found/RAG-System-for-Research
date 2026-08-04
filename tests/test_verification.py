import asyncio
import unittest

from src.pipeline.stance import AGREE, DISAGREE, describe_split, group_by_stance
from src.pipeline.verification import (
    FABRICATED,
    SUPPORTED,
    UNSUPPORTED,
    extract_claims,
    locate_quote,
    summarise,
    to_payload,
    verify_claims,
)

CHUNK = {
    "p-aaaa-chunk-3": {
        "text": "Reducing Overfitting\n\nWe use dropout in the first two fully-connected "
                "layers. Without dropout, our network exhibits substantial overfitting.",
        "doc_start": 5000,
    }
}


class FakeLLM:
    def __init__(self, reply):
        self.reply = reply
        self.calls = 0

    async def ainvoke(self, prompt):
        self.calls += 1
        if isinstance(self.reply, Exception):
            raise self.reply
        return type("M", (), {"content": self.reply})()


class TestLocateQuote(unittest.TestCase):
    def test_exact_match_returns_offsets(self):
        found = locate_quote("dropout in the first two", CHUNK["p-aaaa-chunk-3"]["text"])
        start, end, exact = found
        self.assertTrue(exact)
        self.assertEqual(CHUNK["p-aaaa-chunk-3"]["text"][start:end], "dropout in the first two")

    def test_whitespace_differences_still_match_but_are_marked_inexact(self):
        # A model re-typing a quote across a line break is formatting, not fraud.
        found = locate_quote("dropout in   the\nfirst two", CHUNK["p-aaaa-chunk-3"]["text"])
        self.assertIsNotNone(found)
        self.assertFalse(found[2])

    def test_absent_quote_returns_none(self):
        self.assertIsNone(locate_quote("we used a dropout rate of 0.7", CHUNK["p-aaaa-chunk-3"]["text"]))


class TestExtractClaims(unittest.TestCase):
    def test_located_quote_yields_document_offsets(self):
        answer = 'Dropout was applied [p-aaaa-chunk-3 "dropout in the first two fully-connected layers"].'
        claim = extract_claims(answer, CHUNK)[0]
        self.assertEqual(len(claim.spans), 1)
        span = claim.spans[0]
        # Document offset = chunk offset within the paper + offset within the chunk.
        self.assertEqual(span.doc_start, 5000 + span.start)
        self.assertGreater(span.doc_end, span.doc_start)

    def test_quote_not_in_chunk_is_fabricated(self):
        answer = 'The rate was 0.7 [p-aaaa-chunk-3 "we used a dropout rate of 0.7"].'
        claim = extract_claims(answer, CHUNK)[0]
        self.assertEqual(claim.verdict, FABRICATED)
        self.assertIn("does not appear", claim.reason)

    def test_citation_to_a_chunk_that_was_never_retrieved_is_fabricated(self):
        claim = extract_claims('Something [ghost-9999-chunk-1 "anything"].', CHUNK)[0]
        self.assertEqual(claim.verdict, FABRICATED)
        self.assertIn("not retrieved", claim.reason)

    def test_answer_splits_into_sentences(self):
        claims = extract_claims("First point. Second point. Third point.", CHUNK)
        self.assertEqual(len(claims), 3)

    def test_uncited_sentences_are_counted(self):
        claims = extract_claims(
            "This is a long confident assertion with no citation attached to it at all.",
            CHUNK,
        )
        self.assertEqual(summarise(claims)["uncited_sentences"], 1)


class TestEntailmentPass(unittest.TestCase):
    def _run(self, answer, reply):
        claims = extract_claims(answer, CHUNK)
        llm = FakeLLM(reply)
        return asyncio.run(verify_claims(llm, claims)), llm

    def test_marks_supported_and_unsupported(self):
        answer = (
            'Dropout was used [p-aaaa-chunk-3 "dropout in the first two fully-connected layers"]. '
            'It halved training time [p-aaaa-chunk-3 "substantial overfitting"].'
        )
        claims, llm = self._run(
            answer,
            '[{"n":1,"verdict":"supported","why":"stated directly"},'
            ' {"n":2,"verdict":"unsupported","why":"passage says nothing about time"}]',
        )
        self.assertEqual(claims[0].verdict, SUPPORTED)
        self.assertEqual(claims[1].verdict, UNSUPPORTED)
        self.assertEqual(llm.calls, 1, "entailment must be one batched call")

    def test_fabricated_claims_skip_the_llm_entirely(self):
        # The deterministic layer already decided; do not pay for a second opinion.
        claims, llm = self._run('X [p-aaaa-chunk-3 "not in the text at all"].', "[]")
        self.assertEqual(claims[0].verdict, FABRICATED)
        self.assertEqual(llm.calls, 0)

    def test_verifier_outage_leaves_claims_unchecked_not_wrongly_passed(self):
        answer = 'Dropout was used [p-aaaa-chunk-3 "dropout in the first two"].'
        claims, _ = self._run(answer, RuntimeError("429"))
        self.assertNotEqual(claims[0].verdict, SUPPORTED)

    def test_payload_carries_offsets(self):
        answer = 'Dropout was used [p-aaaa-chunk-3 "dropout in the first two"].'
        payload = to_payload(extract_claims(answer, CHUNK))
        span = payload[0]["spans"][0]
        self.assertIn("docStart", span)
        self.assertIn("quote", span)


class FakePassage:
    def __init__(self, pid, paper, year):
        self.id = pid
        self.text = f"header\n\nbody of {pid}"
        self.metadata = {"paper_title": paper, "pub_year": year}


class TestStanceSplit(unittest.TestCase):
    def test_no_disagreement_produces_no_block(self):
        ps = [FakePassage("a", "Paper A", 2019)]
        labels = {"a": {"stance": AGREE, "claim": "finds an effect"}}
        self.assertEqual(describe_split(group_by_stance(ps, labels), labels), "")

    def test_disagreement_is_reported_with_both_sides(self):
        ps = [FakePassage("a", "Paper A", 2019), FakePassage("b", "Paper B", 2023)]
        labels = {
            "a": {"stance": AGREE, "claim": "finds an effect"},
            "b": {"stance": DISAGREE, "claim": "reports no effect"},
        }
        out = describe_split(group_by_stance(ps, labels), labels)
        self.assertIn("Paper A", out)
        self.assertIn("Paper B", out)
        self.assertIn("reports no effect", out)

    def test_notes_when_the_pushback_is_more_recent(self):
        ps = [FakePassage("a", "Old", 2019), FakePassage("b", "New", 2023)]
        labels = {
            "a": {"stance": AGREE, "claim": "effect"},
            "b": {"stance": DISAGREE, "claim": "no effect"},
        }
        self.assertIn("more recent", describe_split(group_by_stance(ps, labels), labels))


if __name__ == "__main__":
    unittest.main()
