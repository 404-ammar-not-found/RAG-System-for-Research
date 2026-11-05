import asyncio
import unittest

from src.pipeline.qa import QaEngine, _suggest_followups
from src.pipeline.settings import PipelineSettings


class FakeLLM:
    def __init__(self, reply="", tokens=()):
        self.reply, self.tokens = reply, tokens

    async def ainvoke(self, prompt):
        return self.reply

    async def astream(self, prompt):
        for t in self.tokens:
            yield t


def make_engine(**settings):
    engine = QaEngine.__new__(QaEngine)  # skip __init__: no real clients in tests
    engine.settings = PipelineSettings(
        cluster_stance=False, verify_claims=False, debug_retrieval=False, **settings
    )
    engine.llm = FakeLLM(tokens=["Hello ", "world."])
    engine.rerank_llm = FakeLLM(reply='["A?", "B?", "C?"]')

    async def no_passages(query):
        return []

    engine.retrieve = no_passages
    return engine


class TestAnswerStream(unittest.TestCase):
    def collect(self, engine):
        async def run():
            return [e async for e in engine.answer_stream("what is attention?")]

        return asyncio.run(run())

    def test_stage_events_in_order_then_tokens_then_done(self):
        events = self.collect(make_engine())
        stages = [e["name"] for e in events if e["type"] == "stage"]
        self.assertEqual(stages, ["retrieving", "writing", "verifying"])
        tokens = "".join(e["text"] for e in events if e["type"] == "token")
        self.assertEqual(tokens, "Hello world.")
        self.assertEqual(events[-1]["type"], "done")

    def test_done_carries_suggestions(self):
        events = self.collect(make_engine(suggest_followups=True))
        self.assertEqual(events[-1]["suggestions"], ["A?", "B?", "C?"])

    def test_suggestions_off(self):
        events = self.collect(make_engine(suggest_followups=False))
        self.assertEqual(events[-1]["suggestions"], [])


class TestSuggestFollowups(unittest.TestCase):
    def test_degrades_to_empty_on_garbage(self):
        out = asyncio.run(_suggest_followups(FakeLLM(reply="not json"), "q", "a"))
        self.assertEqual(out, [])

    def test_caps_at_three_strings(self):
        llm = FakeLLM(reply='["1?", "2?", "3?", "4?", 5, ""]')
        out = asyncio.run(_suggest_followups(llm, "q", "a"))
        self.assertEqual(out, ["1?", "2?", "3?"])


if __name__ == "__main__":
    unittest.main()
