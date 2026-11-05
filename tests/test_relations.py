import asyncio
import unittest
from unittest import mock

from src.pipeline.relations import (
    RelationGraph,
    format_relations,
    graph_search,
    merge_extractions,
    norm_label,
    title_citations,
    with_relations,
)
from src.pipeline.retrieval import Bm25Index, retrieve_passages
from src.pipeline.settings import PipelineSettings


def node(id, label, kind="concept", stub=False):
    out = {"id": id, "label": label, "file_type": kind}
    if stub:
        out["metadata"] = {"graphify_reference_stub": True}
    return out


def edge(source, target, relation="implements", score=1.0):
    return {
        "source": source,
        "target": target,
        "relation": relation,
        "confidence": "EXTRACTED",
        "confidence_score": score,
        "source_location": "Section 3",
    }


TITLES = {"bert": ["BERT: Pre-training"], "vit": ["An Image is Worth 16x16 Words"], "attn": ["Attention Is All You Need"]}
EXTRACTIONS = {
    "bert": {
        "nodes": [
            node("b_paper", "BERT: Pre-training", "paper"),
            node("b_model", "BERT (Bidirectional Encoders)"),
            node("b_tf", "Transformer architecture (Vaswani et al., 2017)", stub=True),
            node("b_attn", "Attention Is All You Need (Vaswani)", "paper", stub=True),
        ],
        "edges": [edge("b_model", "b_tf"), edge("b_paper", "b_attn", "cites")],
    },
    "vit": {
        "nodes": [
            node("v_paper", "An Image is Worth 16x16 Words", "paper"),
            node("v_model", "Vision Transformer"),
            node("v_tf", "Transformer Architecture", stub=True),
        ],
        "edges": [edge("v_model", "v_tf"), edge("v_model", "v_model")],
    },
}


def graph():
    return RelationGraph.from_extraction(merge_extractions(EXTRACTIONS, TITLES))


class NormLabel(unittest.TestCase):
    def test_drops_parentheticals_and_punctuation(self):
        self.assertEqual(norm_label("Transformer architecture (Vaswani et al., 2017)"), "transformer architecture")

    def test_keeps_a_label_that_is_only_a_parenthetical(self):
        self.assertEqual(norm_label("(MLM)"), "mlm")


class Merge(unittest.TestCase):
    def test_same_concept_from_two_papers_becomes_one_node(self):
        merged = merge_extractions(EXTRACTIONS, TITLES)
        shared = [n for n in merged["nodes"] if n["label"].lower().startswith("transformer architecture")]
        self.assertEqual(len(shared), 1)
        self.assertEqual(shared[0]["papers"], ["bert", "vit"])

    def test_reference_to_a_library_paper_resolves_to_that_paper(self):
        self.assertIn(("bert", "cites", "attn"), graph().paper_links())

    def test_self_loops_are_dropped(self):
        merged = merge_extractions(EXTRACTIONS, TITLES)
        self.assertFalse([e for e in merged["edges"] if e["source"] == e["target"]])


class TitleCitations(unittest.TestCase):
    def test_finds_a_title_across_line_breaks_and_hyphens(self):
        texts = {"bert": "[12] A. Vaswani. Atten-\ntion is all\nyou need. NeurIPS", "attn": "Attention Is All You Need"}
        self.assertEqual(title_citations(texts, TITLES), [("bert", "attn")])

    def test_ignores_titles_too_short_to_be_distinctive(self):
        self.assertEqual(title_citations({"a": "we use deep nets", "b": "x"}, {"b": ["Deep nets"]}), [])


class Queries(unittest.TestCase):
    def test_question_about_a_concept_reaches_every_paper_using_it(self):
        self.assertEqual(graph().related_papers("Which models use the Transformer?"), {"bert", "vit"})

    def test_unrelated_question_finds_nothing(self):
        self.assertEqual(graph().related_papers("What is the capital of France?"), set())

    def test_facts_name_both_ends_and_the_source_paper(self):
        facts = graph().facts("How does Vision Transformer relate to the Transformer architecture?")
        self.assertIn(
            {
                "source": "Vision Transformer",
                "relation": "implements",
                "target": "Transformer architecture (Vaswani et al., 2017)",
                "paper": "vit",
                "confidence": "EXTRACTED",
            },
            facts,
        )

    def test_shared_concepts_link_papers_that_never_cite_each_other(self):
        self.assertEqual(
            graph().shared_concepts(),
            [{"a": "bert", "b": "vit", "concepts": ["Transformer architecture (Vaswani et al., 2017)"]}],
        )


class GraphSearch(unittest.TestCase):
    def test_only_returns_passages_from_related_papers(self):
        index = Bm25Index()
        index.load_texts(
            [
                ("b1", "BERT stacks transformer encoder layers", {"source": "/data/bert.pdf"}),
                ("v1", "Vision Transformer splits an image into patches", {"source": "/data/vit.pdf"}),
                ("g1", "The transformer in this generator is unrelated", {"source": "/data/gan.pdf"}),
            ]
        )
        ids = {p.id for p in graph_search(index, graph(), "Which models use the Transformer?", 5)}
        self.assertEqual(ids, {"b1", "v1"})

    def test_no_graph_match_means_no_passages(self):
        index = Bm25Index()
        index.load_texts([("b1", "France has a capital", {"source": "/data/bert.pdf"})])
        self.assertEqual(graph_search(index, graph(), "capital of France", 5), [])


class RetrievalChannel(unittest.TestCase):
    def test_passages_from_related_papers_are_tagged_with_the_graph_channel(self):
        index = Bm25Index()
        index.load_texts(
            [
                ("b1", "BERT stacks transformer encoder layers", {"source": "/data/bert.pdf"}),
                ("g1", "An unrelated generator discriminator game", {"source": "/data/gan.pdf"}),
                ("g2", "Noise samples feed the generator", {"source": "/data/gan.pdf"}),
                ("g3", "The discriminator estimates a probability", {"source": "/data/gan.pdf"}),
            ]
        )
        settings = PipelineSettings(
            use_reranker=False, neighbour_expansion=False, debug_retrieval=False, top_k=5
        )
        with mock.patch("src.pipeline.retrieval.dense_search", return_value=[]):
            passages = asyncio.run(
                retrieve_passages(None, index, "Which models use the Transformer?", settings, relations=graph())
            )
        hit = next(p for p in passages if p.id == "b1")
        self.assertIn("graph", hit.channels)


class LiteratureMapData(unittest.TestCase):
    def test_adds_missing_citations_and_shared_concepts(self):
        base = {"papers": [{"file": "bert"}], "edges": [["vit", "bert"]]}
        out = with_relations(base, graph())
        self.assertEqual(out["edges"], [["bert", "attn"], ["vit", "bert"]])
        self.assertEqual(out["shared"][0]["a"], "bert")

    def test_without_a_graph_the_map_data_is_unchanged_apart_from_shared(self):
        base = {"papers": [], "edges": [["a", "b"]]}
        self.assertEqual(with_relations(base, None), {"papers": [], "edges": [["a", "b"]], "shared": []})


class PromptContext(unittest.TestCase):
    def test_prompt_lists_relationships_and_says_they_are_not_citable(self):
        block = format_relations(graph().facts("How does Vision Transformer use the Transformer architecture?"))
        self.assertIn("Vision Transformer -[implements]-> Transformer architecture", block)
        self.assertIn("never cite", block.lower())

    def test_no_facts_means_no_block(self):
        self.assertEqual(format_relations([]), "")


if __name__ == "__main__":
    unittest.main()
