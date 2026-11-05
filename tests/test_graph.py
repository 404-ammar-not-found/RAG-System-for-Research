import unittest

from src.pipeline.scholarly import PaperRecord, citation_edges


def rec(name, openalex_id=None, refs=()):
    return PaperRecord(key=name, openalex_id=openalex_id, reference_ids=list(refs))


class TestCitationEdges(unittest.TestCase):
    def test_edge_found_when_reference_matches_a_corpus_id(self):
        records = {
            "bert": rec("bert", "W2", refs=["W1", "W999"]),
            "transformer": rec("transformer", "W1"),
        }
        self.assertEqual(citation_edges(records), [("bert", "transformer")])

    def test_papers_without_openalex_id_produce_no_edges(self):
        records = {
            "a": rec("a", None, refs=["W1"]),
            "b": rec("b", "W1", refs=["W2"]),
        }
        # "a" can still cite; it just cannot be cited.
        self.assertEqual(citation_edges(records), [("a", "b")])

    def test_self_edges_are_excluded(self):
        records = {"a": rec("a", "W1", refs=["W1"])}
        self.assertEqual(citation_edges(records), [])

    def test_empty_corpus(self):
        self.assertEqual(citation_edges({}), [])


if __name__ == "__main__":
    unittest.main()
