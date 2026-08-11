import unittest

from src.pipeline.deps import PROVIDERS
from src.pipeline.graph_store import EDGE_TYPE_MAP, EDGE_TYPES, ENTITY_TYPES, format_facts
from src.pipeline.settings import PipelineSettings


class FakeEdge:
    def __init__(self, name, fact, valid_at=None):
        self.name = name
        self.fact = fact
        self.valid_at = valid_at


class TestOntology(unittest.TestCase):
    def test_every_edge_type_map_entry_names_real_types(self):
        for (src, tgt), edges in EDGE_TYPE_MAP.items():
            for side in (src, tgt):
                self.assertTrue(
                    side == "Entity" or side in ENTITY_TYPES,
                    f"{side} in edge_type_map is not a declared entity type",
                )
            for edge in edges:
                self.assertIn(edge, EDGE_TYPES, f"{edge} is not a declared edge type")

    def test_has_a_wildcard_fallback(self):
        # Without this the extractor can be left with no legal edge to emit.
        self.assertIn(("Entity", "Entity"), EDGE_TYPE_MAP)


class TestFormatFacts(unittest.TestCase):
    def test_numbers_and_labels_facts(self):
        out = format_facts([FakeEdge("EvaluatedOn", "X was evaluated on Y.")])
        self.assertIn("1. [EvaluatedOn] X was evaluated on Y.", out)

    def test_empty_is_empty_not_a_crash(self):
        self.assertEqual(format_facts([]), "")


class TestQuotaCriticalSettings(unittest.TestCase):
    def test_passage_and_graph_use_different_embedding_models(self):
        # Hosted embeddings are metered per model, so pointing the passage store
        # and the graph at different ones buys two daily budgets instead of one.
        # It does not apply to the local model, which is not metered at all.
        for provider, models in PROVIDERS.items():
            if models["embed"] is None:  # Anthropic: no embedding API at all
                self.assertIsNone(models["graph_embed"], provider)
            elif models["keys"]:  # hosted, therefore quota-limited
                self.assertNotEqual(models["embed"], models["graph_embed"], provider)

    def test_concurrency_is_set_explicitly_not_via_env(self):
        # graphiti_core reads SEMAPHORE_LIMIT at import time, before load_dotenv,
        # so it must be passed to the constructor as max_coroutines instead.
        s = PipelineSettings()
        self.assertGreater(s.max_coroutines, 0)
        self.assertLessEqual(s.max_coroutines, 5)

    def test_answer_model_is_not_the_20_per_day_preview(self):
        self.assertNotEqual(PROVIDERS["gemini"]["text"], "gemini-3-flash-preview")


if __name__ == "__main__":
    unittest.main()
