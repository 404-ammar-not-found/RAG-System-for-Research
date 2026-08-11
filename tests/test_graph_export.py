import unittest
from types import SimpleNamespace

from src.pipeline.graph_export import build_chunk_layer, build_graph


def node(uuid, name, label, summary=""):
    return SimpleNamespace(uuid=uuid, name=name, labels=["Entity", label], summary=summary)


def edge(src, tgt, name, fact):
    return SimpleNamespace(uuid=f"{src}-{tgt}", source_node_uuid=src,
                           target_node_uuid=tgt, name=name, fact=fact)


class FakeCollection:
    def __init__(self, rows):
        self.rows = rows

    def get(self, include=None):
        return {
            "ids": [r[0] for r in self.rows],
            "documents": [r[1] for r in self.rows],
            "metadatas": [r[2] for r in self.rows],
        }


class FakeStore:
    def __init__(self, rows):
        self._collection = FakeCollection(rows)


ROWS = [
    ("p-aaaa-chunk-0", "intro text", {"source": "data/p.pdf", "section": "Introduction",
                                      "chunk_index": 0, "paper_title": "P"}),
    ("p-aaaa-chunk-1", "more intro", {"source": "data/p.pdf", "section": "Introduction",
                                      "chunk_index": 1, "paper_title": "P"}),
    ("p-aaaa-chunk-10", "later", {"source": "data/p.pdf", "section": "Results",
                                  "chunk_index": 10, "paper_title": "P"}),
]

NODES = [node("u1", "Transformer", "Model"), node("u2", "WMT 2014", "Dataset")]
EDGES = [edge("u1", "u2", "EvaluatedOn", "The Transformer was evaluated on WMT 2014.")]


class TestBuildGraph(unittest.TestCase):
    def setUp(self):
        self.graph = build_graph(
            NODES, EDGES, FakeStore(ROWS), {"p::Introduction": ["u1"]}
        )

    def test_entity_nodes_carry_type_not_base_label(self):
        entities = {n["id"]: n for n in self.graph["nodes"] if n["kind"] == "entity"}
        self.assertEqual(entities["u1"]["type"], "Model")
        self.assertEqual(entities["u1"]["group"], "Model")
        self.assertEqual(entities["u1"]["name"], "Transformer")

    def test_relation_links_are_labelled(self):
        rel = [l for l in self.graph["links"] if l["kind"] == "relation"]
        self.assertEqual(len(rel), 1)
        self.assertEqual(rel[0]["label"], "EvaluatedOn")
        self.assertIn("WMT", rel[0]["fact"])

    def test_every_link_endpoint_resolves_to_something_real(self):
        # A dangling endpoint is an invisible bug: the visualiser drops the
        # link and a whole relationship silently disappears from the figure.
        ids = {n["id"] for n in self.graph["nodes"]}
        papers = {n["group"] for n in self.graph["nodes"] if n["kind"] == "chunk"}
        for link in self.graph["links"]:
            self.assertIn(link["source"], ids)
            # `from_paper` names a paper, which is not a node here — the
            # visualiser synthesises one node per source path — so it resolves
            # against the paper set instead.
            expected = papers if link["kind"] == "from_paper" else ids
            self.assertIn(link["target"], expected, link["kind"])

    def test_entities_reach_their_paper_even_without_a_passage(self):
        # References are excluded from the passage store, so entities extracted
        # there can never link to a chunk. The episode still names the paper.
        graph = build_graph(
            NODES, EDGES, FakeStore(ROWS), {"p::References": ["u1", "u2"]}
        )
        from_paper = [l for l in graph["links"] if l["kind"] == "from_paper"]
        self.assertEqual({l["source"] for l in from_paper}, {"u1", "u2"})
        self.assertEqual({l["target"] for l in from_paper}, {"data/p.pdf"})
        # That section has no chunks, so there is no passage-level provenance.
        self.assertEqual([l for l in graph["links"] if l["kind"] == "mentions"], [])

    def test_no_duplicate_node_ids(self):
        # The old exporter merged into the previous file and never evicted,
        # accumulating the same chunk under two id schemes.
        ids = [n["id"] for n in self.graph["nodes"]]
        self.assertEqual(len(ids), len(set(ids)))

    def test_rebuild_is_idempotent(self):
        again = build_graph(NODES, EDGES, FakeStore(ROWS), {"p::Introduction": ["u1"]})
        self.assertEqual(len(again["nodes"]), len(self.graph["nodes"]))
        self.assertEqual(len(again["links"]), len(self.graph["links"]))

    def test_mentions_link_joins_entity_to_its_section_chunks(self):
        mentions = [l for l in self.graph["links"] if l["kind"] == "mentions"]
        self.assertEqual(
            sorted(l["target"] for l in mentions),
            ["p-aaaa-chunk-0", "p-aaaa-chunk-1"],
        )

    def test_sequence_links_follow_chunk_index_not_lexical_order(self):
        _, links, _ = build_chunk_layer(FakeStore(ROWS))
        pairs = [(l["source"], l["target"]) for l in links]
        # chunk-1 -> chunk-10, never chunk-10 -> chunk-1 (which lexical sort gives).
        self.assertIn(("p-aaaa-chunk-1", "p-aaaa-chunk-10"), pairs)

    def test_entity_only_export_when_no_vectorstore(self):
        graph = build_graph(NODES, EDGES, None)
        self.assertTrue(all(n["kind"] == "entity" for n in graph["nodes"]))

    def test_drops_edges_pointing_at_missing_nodes(self):
        graph = build_graph(NODES, EDGES + [edge("u1", "ghost", "Uses", "x")], None)
        self.assertEqual(len([l for l in graph["links"] if l["kind"] == "relation"]), 1)


if __name__ == "__main__":
    unittest.main()
