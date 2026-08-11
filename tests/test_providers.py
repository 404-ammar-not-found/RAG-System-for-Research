import os
import unittest
from unittest import mock

from src.pipeline import deps
from src.pipeline.settings import PipelineSettings


def env(**pairs):
    """Isolate the process env — including from the repo's own .env file."""
    return mock.patch.multiple(
        deps, _load_env=lambda: None
    ), mock.patch.dict(os.environ, pairs, clear=True)


class ProviderSelection(unittest.TestCase):
    def _with(self, **pairs):
        patch_load, patch_env = env(**pairs)
        self.addCleanup(patch_load.stop)
        self.addCleanup(patch_env.stop)
        patch_load.start()
        patch_env.start()

    def test_single_key_picks_its_provider(self):
        for key, provider in [
            ("GEMINI_API_KEY", "gemini"),
            ("ANTHROPIC_API_KEY", "anthropic"),
            ("OPENAI_API_KEY", "openai"),
        ]:
            with self.subTest(key=key):
                patch_load, patch_env = env(**{key: "x"})
                with patch_load, patch_env:
                    self.assertEqual(deps.chat_provider(), provider)

    def test_llm_provider_overrides_key_order(self):
        self._with(GEMINI_API_KEY="x", OPENAI_API_KEY="y", LLM_PROVIDER="openai")
        self.assertEqual(deps.chat_provider(), "openai")

    def test_forcing_a_provider_without_its_key_is_an_error(self):
        self._with(GEMINI_API_KEY="x", LLM_PROVIDER="anthropic")
        with self.assertRaises(ValueError):
            deps.chat_provider()

    def test_unknown_provider_name_is_an_error(self):
        self._with(GEMINI_API_KEY="x", LLM_PROVIDER="llama")
        with self.assertRaises(ValueError):
            deps.chat_provider()

    def test_no_key_at_all_is_an_error(self):
        self._with()
        with self.assertRaises(ValueError):
            deps.chat_provider()


class Embeddings(unittest.TestCase):
    def _with(self, **pairs):
        patch_load, patch_env = env(**pairs)
        self.addCleanup(patch_load.stop)
        self.addCleanup(patch_env.stop)
        patch_load.start()
        patch_env.start()

    def test_anthropic_borrows_vectors_from_another_key(self):
        self._with(ANTHROPIC_API_KEY="x", OPENAI_API_KEY="y")
        self.assertEqual(deps.chat_provider(), "anthropic")
        self.assertEqual(deps.embed_provider(), "openai")

    def test_chat_provider_embeds_for_itself_when_it_can(self):
        self._with(OPENAI_API_KEY="y")
        self.assertEqual(deps.embed_provider(), "openai")


class LocalEmbeddings(unittest.TestCase):
    def _with(self, **pairs):
        patch_load, patch_env = env(**pairs)
        self.addCleanup(patch_load.stop)
        self.addCleanup(patch_env.stop)
        patch_load.start()
        patch_env.start()

    def test_anthropic_alone_falls_back_to_local_instead_of_failing(self):
        # Anthropic has no embedding endpoint, but that is no longer fatal:
        # on-device embeddings need no key at all.
        self._with(ANTHROPIC_API_KEY="x")
        self.assertEqual(deps.embed_provider(), "local")

    def test_local_can_be_forced_over_a_working_key(self):
        # The way out of a spent daily quota.
        self._with(GEMINI_API_KEY="x", EMBED_PROVIDER="local")
        self.assertEqual(deps.embed_provider(), "local")
        self.assertEqual(deps.chat_provider(), "gemini")

    def test_local_is_not_selectable_for_chat(self):
        self._with(GEMINI_API_KEY="x", LLM_PROVIDER="local")
        with self.assertRaises(ValueError):
            deps.chat_provider()

    def test_each_embedding_provider_gets_its_own_collection(self):
        # 384-dim local vectors cannot be written into a 3072-dim Gemini
        # collection; Chroma fixes width at creation time.
        s = PipelineSettings()
        self._with(GEMINI_API_KEY="x")
        self.assertEqual(deps.collection_for(s), s.collection_name)
        with mock.patch.dict(os.environ, {"GEMINI_API_KEY": "x", "EMBED_PROVIDER": "local"}):
            self.assertEqual(deps.collection_for(s), f"{s.collection_name}-local")


class LocalModelRoundTrip(unittest.TestCase):
    """Exercises the real ONNX model — no network once it is cached."""

    def test_embeds_text_at_the_expected_width(self):
        from src.pipeline.local_embed import DIMENSIONS, LocalEmbeddings as Embedder

        embedder = Embedder()
        vectors = embedder.embed_documents(["attention is all you need", "residual learning"])
        self.assertEqual(len(vectors), 2)
        self.assertEqual(len(vectors[0]), DIMENSIONS)
        self.assertTrue(all(isinstance(x, float) for x in vectors[0]))
        # A query embeds into the same space as a document.
        self.assertEqual(len(embedder.embed_query("what is attention?")), DIMENSIONS)


class ModelResolution(unittest.TestCase):
    def _with(self, **pairs):
        patch_load, patch_env = env(**pairs)
        self.addCleanup(patch_load.stop)
        self.addCleanup(patch_env.stop)
        patch_load.start()
        patch_env.start()

    def test_defaults_come_from_the_provider_table(self):
        self._with(ANTHROPIC_API_KEY="x", GEMINI_API_KEY="y", LLM_PROVIDER="anthropic")
        s = PipelineSettings(text_llm_model="", graph_llm_model="", embed_model="")
        self.assertEqual(deps.model_for(s, "text"), "claude-opus-5")
        # Graph work runs on the cheap model, and Graphiti always sends
        # temperature — which the Claude 5 series rejects.
        self.assertEqual(deps.model_for(s, "graph"), "claude-haiku-4-5")
        # Vectors fall through to the Gemini key.
        self.assertEqual(deps.model_for(s, "embed"), "models/gemini-embedding-2")

    def test_explicit_setting_wins(self):
        self._with(GEMINI_API_KEY="x")
        s = PipelineSettings(text_llm_model="my-model")
        self.assertEqual(deps.model_for(s, "text"), "my-model")


if __name__ == "__main__":
    unittest.main()
