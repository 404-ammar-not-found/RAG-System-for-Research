import argparse
import unittest

from scripts.eval_answers import summarise_run
from scripts.eval_retrieval import build_settings


class RetrievalEvalSettings(unittest.TestCase):
    def test_ranking_is_scored_before_neighbour_expansion(self):
        # Expansion re-sorts passages by document position, so scoring after it
        # measures reading order rather than ranking.
        args = argparse.Namespace(k=8, no_graph=False, no_rerank=False)
        self.assertFalse(build_settings(args).neighbour_expansion)


def row(*verdicts, bare=0, evidence=True, error=None):
    return {
        "verdicts": list(verdicts),
        "bare_citations": bare,
        "cited_expected_evidence": evidence,
        "error": error,
    }


class AnswerSummary(unittest.TestCase):
    def test_quote_accuracy_counts_fabricated_quotes_against_all_quotes(self):
        report = summarise_run([row("supported", "partial", "unsupported", "fabricated")])
        self.assertEqual(report["quotes"], {"found": 3, "total": 4, "rate": 0.75})

    def test_entailment_rate_ignores_fabricated_and_unjudged_claims(self):
        report = summarise_run([row("supported", "partial", "fabricated", "judged_none")])
        self.assertEqual(report["entailment"], {"supported": 1, "judged": 2, "rate": 0.5})

    def test_failed_questions_are_reported_not_scored(self):
        report = summarise_run([row("supported"), row(error="429")])
        self.assertEqual(report["answers"], {"scored": 1, "failed": 1})

    def test_bare_citations_are_counted_as_unverifiable(self):
        self.assertEqual(summarise_run([row("supported", bare=2)])["bare_citations"], 2)

    def test_empty_run_has_no_rates(self):
        report = summarise_run([])
        self.assertIsNone(report["quotes"]["rate"])
        self.assertIsNone(report["entailment"]["rate"])
